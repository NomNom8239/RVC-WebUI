import logging
import os
import tempfile
import traceback
from io import BytesIO

import ffmpeg
import numpy as np
import soundfile as sf
import torch

logger = logging.getLogger(__name__)

from infer.audio import clean_path, load_audio, wav2
from infer.module.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from infer.vc.pipeline import Pipeline
from infer.vc.utils import *
from i18n.i18n import I18nAuto
from tools.progress import batch_status, should_report
from tools.cuda_graph import clear_cuda_graph_cache


i18n = I18nAuto()

LONG_AUDIO_THRESHOLD_SECONDS = 120.0
LONG_AUDIO_CHUNK_SECONDS = 60.0
LONG_AUDIO_OVERLAP_SECONDS = 1.0
LONG_AUDIO_MIN_TAIL_SECONDS = 5.0


def _probe_audio_duration(path):
    info = ffmpeg.probe(clean_path(os.fspath(path)), cmd="ffprobe")
    stream = next(
        item for item in info.get("streams", [])
        if item.get("codec_type") == "audio"
    )

    for container in (stream, info.get("format", {})):
        value = container.get("duration")
        if value not in (None, "", "N/A"):
            duration = float(value)
            if duration > 0:
                return duration

    duration_ts = stream.get("duration_ts")
    time_base = stream.get("time_base")
    if duration_ts not in (None, "", "N/A") and time_base:
        numerator, denominator = (int(part) for part in time_base.split("/", 1))
        if denominator:
            duration = float(duration_ts) * numerator / denominator
            if duration > 0:
                return duration

    raise RuntimeError("Could not determine audio duration.")


def _load_audio_segment(path, start_seconds, duration_seconds, sample_rate=16000):
    input_stream = ffmpeg.input(
        clean_path(os.fspath(path)),
        ss=max(0.0, float(start_seconds)),
    )
    out, _ = (
        input_stream.output(
            "-",
            format="f32le",
            acodec="pcm_f32le",
            ac=1,
            ar=int(sample_rate),
            t=max(0.0, float(duration_seconds)),
        )
        .run(
            cmd=["ffmpeg", "-nostdin"],
            capture_stdout=True,
            capture_stderr=True,
        )
    )
    return np.frombuffer(out, np.float32).copy()


def _long_audio_ranges(total_seconds):
    ranges = []
    start = 0.0
    total_seconds = float(total_seconds)

    while start < total_seconds:
        end = min(start + LONG_AUDIO_CHUNK_SECONDS, total_seconds)
        remaining = total_seconds - end
        if 0 < remaining < LONG_AUDIO_MIN_TAIL_SECONDS:
            end = total_seconds

        ranges.append((start, end))
        if end >= total_seconds:
            break
        start = max(0.0, end - LONG_AUDIO_OVERLAP_SECONDS)

    return ranges


def inference_status(title, state, detail=""):
    lines = ["【%s】" % i18n(title), "%s：%s" % (i18n("状态"), i18n(state))]
    if detail:
        lines.extend(["", str(detail).strip()])
    return "\n".join(lines)


def normalized_speaker_info(checkpoint, n_spk):
    speaker_info = []
    seen = set()
    for item in checkpoint.get("speaker_info", []):
        try:
            speaker_id = int(item["id"])
            speaker_name = str(item["name"])
        except (KeyError, TypeError, ValueError):
            continue
        if (
            speaker_id < 0
            or speaker_id >= n_spk
            or not speaker_name
            or speaker_id in seen
        ):
            continue
        seen.add(speaker_id)
        speaker_info.append({"id": speaker_id, "name": speaker_name})
    speaker_info.sort(key=lambda item: item["id"])
    return speaker_info


def speaker_selector_updates(checkpoint, n_spk):
    speaker_info = normalized_speaker_info(checkpoint, n_spk)
    if speaker_info:
        choices = [
            i18n("说话人：%s（ID：%s）") % (item["name"], item["id"])
            for item in speaker_info
        ]
        return (
            {
                "visible": False,
                "value": speaker_info[0]["id"],
                "__type__": "update",
            },
            {
                "visible": True,
                "choices": choices,
                "value": choices[0],
                "__type__": "update",
            },
        )
    return (
        {
            "visible": True,
            "maximum": max(n_spk - 1, 0),
            "__type__": "update",
        },
        {"visible": False, "value": None, "__type__": "update"},
    )


class VC:
    def __init__(self, config):
        self.n_spk = None
        self.tgt_sr = None
        self.net_g = None
        self.pipeline = None
        self.cpt = None
        self.version = None
        self.if_f0 = None
        self.version = None
        self.hubert_model = None

        self.config = config

    def get_vc(self, sid, *to_return_protect):
        logger.info("%s: %s", i18n("选择模型"), sid)

        to_return_protect0 = {
            "visible": self.if_f0 != 0,
            "value": (
                to_return_protect[0] if self.if_f0 != 0 and to_return_protect else 0.5
            ),
            "__type__": "update",
        }
        to_return_protect1 = {
            "visible": self.if_f0 != 0,
            "value": (
                to_return_protect[1] if self.if_f0 != 0 and to_return_protect else 0.33
            ),
            "__type__": "update",
        }

        if sid == "" or sid == []:
            if (
                self.hubert_model is not None
            ):  # 考虑到轮询, 需要加个判断看是否 sid 是由有模型切换到无模型的
                logger.info(i18n("清理模型缓存"))
                clear_cuda_graph_cache(self.net_g)
                clear_cuda_graph_cache(self.hubert_model)
                del (self.net_g, self.n_spk, self.hubert_model, self.tgt_sr)  # ,cpt
                self.hubert_model = self.net_g = self.n_spk = self.hubert_model = (
                    self.tgt_sr
                ) = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                ###楼下不这么折腾清理不干净
                self.if_f0 = self.cpt.get("f0", 1)
                self.version = self.cpt.get("version", "v1")
                if self.version == "v1":
                    if self.if_f0 == 1:
                        self.net_g = SynthesizerTrnMs256NSFsid(
                            *self.cpt["config"], is_half=self.config.is_half
                        )
                    else:
                        self.net_g = SynthesizerTrnMs256NSFsid_nono(*self.cpt["config"])
                elif self.version == "v2":
                    if self.if_f0 == 1:
                        self.net_g = SynthesizerTrnMs768NSFsid(
                            *self.cpt["config"], is_half=self.config.is_half
                        )
                    else:
                        self.net_g = SynthesizerTrnMs768NSFsid_nono(*self.cpt["config"])
                del self.net_g, self.cpt
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            return (
                {"visible": False, "__type__": "update"},
                {"visible": False, "value": None, "__type__": "update"},
                {
                    "visible": True,
                    "value": to_return_protect0,
                    "__type__": "update",
                },
                {
                    "visible": True,
                    "value": to_return_protect1,
                    "__type__": "update",
                },
                "",
                "",
            )
        person = f'{os.getenv("weight_root")}/{sid}'
        logger.info("%s: %s", i18n("正在加载模型"), person)

        if self.net_g is not None:
            clear_cuda_graph_cache(self.net_g)

        self.cpt = torch.load(person, map_location="cpu")
        self.tgt_sr = self.cpt["config"][-1]
        self.cpt["config"][-3] = self.cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
        self.if_f0 = self.cpt.get("f0", 1)
        self.version = self.cpt.get("version", "v1")

        synthesizer_class = {
            ("v1", 1): SynthesizerTrnMs256NSFsid,
            ("v1", 0): SynthesizerTrnMs256NSFsid_nono,
            ("v2", 1): SynthesizerTrnMs768NSFsid,
            ("v2", 0): SynthesizerTrnMs768NSFsid_nono,
        }

        self.net_g = synthesizer_class.get(
            (self.version, self.if_f0), SynthesizerTrnMs256NSFsid
        )(*self.cpt["config"], is_half=self.config.is_half)

        del self.net_g.enc_q

        self.net_g.load_state_dict(self.cpt["weight"], strict=False)
        self.net_g.eval().to(self.config.device)
        if self.config.is_half:
            self.net_g = self.net_g.half()
        else:
            self.net_g = self.net_g.float()

        self.pipeline = Pipeline(self.tgt_sr, self.config)
        n_spk = self.cpt["config"][-3]
        speaker_info = normalized_speaker_info(self.cpt, n_spk)
        speaker_slider_update, speaker_dropdown_update = speaker_selector_updates(
            self.cpt, n_spk
        )
        default_speaker_id = speaker_info[0]["id"] if speaker_info else 0
        index = {
            "value": get_index_path_from_model(sid, default_speaker_id),
            "__type__": "update",
        }
        logger.info("%s: %s", i18n("选择索引"), index["value"])

        return (
            (
                speaker_slider_update,
                speaker_dropdown_update,
                to_return_protect0,
                to_return_protect1,
                index,
                index,
            )
            if to_return_protect
            else speaker_slider_update
        )

    def _normalize_index_path(self, file_index):
        if not file_index:
            return ""
        return (
            str(file_index)
            .strip(" ")
            .strip('"')
            .strip("\n")
            .strip('"')
            .strip(" ")
            .replace("trained", "added")
        )

    def _run_single_audio(
        self,
        sid,
        audio,
        f0_up_key,
        f0_method,
        file_index,
        index_rate,
        resample_sr,
        rms_mix_rate,
        protect,
    ):
        audio = np.asarray(audio, dtype=np.float32)
        if audio.size == 0:
            raise ValueError("Input audio is empty.")

        audio_max = np.abs(audio).max() / 0.95
        if audio_max > 1:
            audio = audio / audio_max

        if self.hubert_model is None:
            self.hubert_model = load_hubert(self.config)

        file_index = self._normalize_index_path(file_index)
        times = [0.0, 0.0, 0.0]
        audio_opt = self.pipeline.pipeline(
            self.hubert_model,
            self.net_g,
            sid,
            audio,
            times,
            int(f0_up_key),
            f0_method,
            file_index,
            index_rate,
            self.if_f0,
            self.tgt_sr,
            resample_sr,
            rms_mix_rate,
            self.version,
            protect,
        )
        tgt_sr = (
            resample_sr
            if self.tgt_sr != resample_sr >= 16000
            else self.tgt_sr
        )
        return tgt_sr, audio_opt, times, file_index

    def _index_info(self, file_index):
        return (
            "%s：%s" % (i18n("索引"), file_index)
            if file_index and os.path.exists(file_index)
            else "%s：%s" % (i18n("索引"), i18n("未使用"))
        )

    def vc_single(
        self,
        sid,
        input_audio_path,
        f0_up_key,
        f0_method,
        file_index,
        index_rate,
        resample_sr,
        rms_mix_rate,
        protect,
    ):
        if input_audio_path is None:
            return inference_status("单次推理", "等待输入", i18n("请上传音频文件")), None
        try:
            audio = load_audio(input_audio_path, 16000)
            tgt_sr, audio_opt, times, file_index = self._run_single_audio(
                sid,
                audio,
                f0_up_key,
                f0_method,
                file_index,
                index_rate,
                resample_sr,
                rms_mix_rate,
                protect,
            )
            return (
                inference_status(
                    "单次推理",
                    "成功",
                    "%s\n%s：%s %.2fs | F0 %.2fs | %s %.2fs"
                    % (
                        self._index_info(file_index),
                        i18n("耗时"),
                        i18n("特征"),
                        times[0],
                        times[1],
                        i18n("合成"),
                        times[2],
                    ),
                ),
                (tgt_sr, audio_opt),
            )
        except Exception:
            info = traceback.format_exc()
            logger.warning(info)
            return inference_status("单次推理", "失败", info), (None, None)

    def vc_single_chunked(
        self,
        sid,
        input_audio_path,
        f0_up_key,
        f0_method,
        file_index,
        index_rate,
        resample_sr,
        rms_mix_rate,
        protect,
        duration_seconds=None,
    ):
        if input_audio_path is None:
            return inference_status("单次推理", "等待输入", i18n("请上传音频文件")), None

        output_path = None
        try:
            if duration_seconds is None:
                duration_seconds = _probe_audio_duration(input_audio_path)

            ranges = _long_audio_ranges(duration_seconds)
            if not ranges:
                raise RuntimeError("No long-audio chunks were generated.")

            total_times = [0.0, 0.0, 0.0]
            output_parts = []
            pending = None
            tgt_sr = None
            normalized_index = self._normalize_index_path(file_index)

            logger.info(
                "Long audio inference: %.2fs, %d chunks (%.1fs, %.1fs overlap)",
                duration_seconds,
                len(ranges),
                LONG_AUDIO_CHUNK_SECONDS,
                LONG_AUDIO_OVERLAP_SECONDS,
            )

            for chunk_index, (start, end) in enumerate(ranges, start=1):
                logger.info(
                    "Long audio chunk %d/%d: %.2f-%.2fs",
                    chunk_index,
                    len(ranges),
                    start,
                    end,
                )
                audio = _load_audio_segment(
                    input_audio_path,
                    start,
                    end - start,
                    sample_rate=16000,
                )
                chunk_sr, audio_opt, times, normalized_index = self._run_single_audio(
                    sid,
                    audio,
                    f0_up_key,
                    f0_method,
                    normalized_index,
                    index_rate,
                    resample_sr,
                    rms_mix_rate,
                    protect,
                )
                del audio

                if tgt_sr is None:
                    tgt_sr = int(chunk_sr)
                elif int(chunk_sr) != tgt_sr:
                    raise RuntimeError(
                        "Long-audio chunk sample rate changed unexpectedly."
                    )

                for index in range(3):
                    total_times[index] += float(times[index])

                current = np.asarray(audio_opt, dtype=np.float32)
                del audio_opt

                if pending is None:
                    pending = current
                    continue

                overlap_samples = min(
                    int(round(LONG_AUDIO_OVERLAP_SECONDS * tgt_sr)),
                    len(pending),
                    len(current),
                )
                if overlap_samples <= 0:
                    output_parts.append(pending)
                    pending = current
                    continue

                fade_in = np.linspace(
                    0.0,
                    1.0,
                    overlap_samples,
                    endpoint=True,
                    dtype=np.float32,
                )
                fade_out = 1.0 - fade_in
                crossfade = (
                    pending[-overlap_samples:] * fade_out
                    + current[:overlap_samples] * fade_in
                )
                if len(pending) > overlap_samples:
                    output_parts.append(pending[:-overlap_samples])
                output_parts.append(crossfade)
                pending = current[overlap_samples:]

            if pending is not None:
                output_parts.append(pending)
            if not output_parts or tgt_sr is None:
                raise RuntimeError("Long-audio inference produced no output.")

            combined = np.concatenate(output_parts)
            combined = np.clip(
                np.rint(combined),
                -32768,
                32767,
            ).astype(np.int16)

            fd, output_path = tempfile.mkstemp(
                prefix="rvc_long_",
                suffix=".wav",
            )
            os.close(fd)
            sf.write(
                output_path,
                combined,
                tgt_sr,
                subtype="PCM_16",
            )
            del combined, output_parts, pending

            detail = (
                "%s\n"
                "Long audio: %.1fs / %d chunks (%.0fs each, %.1fs overlap)\n"
                "%s：%s %.2fs | F0 %.2fs | %s %.2fs"
                % (
                    self._index_info(normalized_index),
                    duration_seconds,
                    len(ranges),
                    LONG_AUDIO_CHUNK_SECONDS,
                    LONG_AUDIO_OVERLAP_SECONDS,
                    i18n("耗时"),
                    i18n("特征"),
                    total_times[0],
                    total_times[1],
                    i18n("合成"),
                    total_times[2],
                )
            )
            return (
                inference_status("单次推理", "成功", detail),
                output_path,
            )
        except Exception:
            if output_path and os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except OSError:
                    pass
            info = traceback.format_exc()
            logger.warning(info)
            return inference_status("单次推理", "失败", info), None

    def vc_single_auto(
        self,
        sid,
        input_audio_path,
        f0_up_key,
        f0_method,
        file_index,
        index_rate,
        resample_sr,
        rms_mix_rate,
        protect,
    ):
        if input_audio_path is None:
            return inference_status("单次推理", "等待输入", i18n("请上传音频文件")), None

        try:
            duration_seconds = _probe_audio_duration(input_audio_path)
        except Exception:
            logger.warning(
                "Could not probe input duration; falling back to normal inference.\n%s",
                traceback.format_exc(),
            )
            return self.vc_single(
                sid,
                input_audio_path,
                f0_up_key,
                f0_method,
                file_index,
                index_rate,
                resample_sr,
                rms_mix_rate,
                protect,
            )

        if duration_seconds <= LONG_AUDIO_THRESHOLD_SECONDS:
            return self.vc_single(
                sid,
                input_audio_path,
                f0_up_key,
                f0_method,
                file_index,
                index_rate,
                resample_sr,
                rms_mix_rate,
                protect,
            )

        return self.vc_single_chunked(
            sid,
            input_audio_path,
            f0_up_key,
            f0_method,
            file_index,
            index_rate,
            resample_sr,
            rms_mix_rate,
            protect,
            duration_seconds=duration_seconds,
        )

    def vc_multi(
        self,
        sid,
        dir_path,
        opt_root,
        paths,
        f0_up_key,
        f0_method,
        file_index,
        index_rate,
        resample_sr,
        rms_mix_rate,
        protect,
        format1,
    ):
        try:
            dir_path = (
                (dir_path or "")
                .strip(" ")
                .strip('"')
                .strip("\n")
                .strip('"')
                .strip(" ")
            )  # 防止小白拷路径头尾带了空格和"和回车
            opt_root = (
                (opt_root or "")
                .strip(" ")
                .strip('"')
                .strip("\n")
                .strip('"')
                .strip(" ")
            )
            if not opt_root:
                yield inference_status(
                    "批量推理", "等待输入", i18n("请填写输出文件夹路径")
                )
                return
            os.makedirs(opt_root, exist_ok=True)
            try:
                if dir_path != "":
                    paths = [
                        os.path.join(dir_path, name) for name in os.listdir(dir_path)
                    ]
                else:
                    paths = [path if isinstance(path, str) else path.name for path in (paths or [])]
            except Exception:
                traceback.print_exc()
                paths = [
                    path if isinstance(path, str) else path.name for path in (paths or [])
                ]
            total = len(paths)
            if total == 0:
                yield batch_status(i18n("批量推理"), 0, 0, 0, 0)
                return
            success = 0
            failed = 0
            failures = []
            for idx, path in enumerate(paths):
                item_failed = False
                info, opt = self.vc_single(
                    sid,
                    path,
                    f0_up_key,
                    f0_method,
                    file_index,
                    index_rate,
                    resample_sr,
                    rms_mix_rate,
                    protect,
                )
                if opt and opt[0] is not None and opt[1] is not None:
                    try:
                        tgt_sr, audio_opt = opt
                        if format1 in ["wav", "flac"]:
                            sf.write(
                                "%s/%s.%s"
                                % (
                                    opt_root,
                                    os.path.splitext(os.path.basename(path))[0],
                                    format1,
                                ),
                                audio_opt,
                                tgt_sr,
                            )
                        else:
                            path = "%s/%s.%s" % (
                                opt_root,
                                os.path.splitext(os.path.basename(path))[0],
                                format1,
                            )
                            with BytesIO() as wavf:
                                sf.write(wavf, audio_opt, tgt_sr, format="wav")
                                wavf.seek(0, 0)
                                with open(path, "wb") as outf:
                                    wav2(wavf, outf, format1)
                        success += 1
                    except Exception:
                        info = "%s\n%s" % (info, traceback.format_exc())
                        failed += 1
                        item_failed = True
                        failures.append("%s：%s" % (os.path.basename(path), info))
                else:
                    failed += 1
                    item_failed = True
                    failures.append("%s：%s" % (os.path.basename(path), info))
                if should_report(idx, total) or item_failed:
                    yield batch_status(
                        i18n("批量推理"),
                        idx + 1,
                        total,
                        success,
                        failed,
                        os.path.basename(path),
                        failures,
                    )
        except Exception:
            yield inference_status("批量推理", "失败", traceback.format_exc())
