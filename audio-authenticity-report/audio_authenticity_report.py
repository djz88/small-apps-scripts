import argparse
import hashlib
import json
import math
import os
import statistics
import struct
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_WINDOW_MS = 250
DEFAULT_RMS_RATIO_THRESHOLD = 8.0
DEFAULT_SAMPLE_RATE = 16000
MAX_DECODE_SECONDS = 1800
STT_BATCH_ENDPOINT = "https://modulate-developer-apis.com/api/velma-2-stt-batch"
AUDIO_SUFFIXES = {
    ".aac",
    ".aiff",
    ".flac",
    ".m4a",
    ".mov",
    ".mp3",
    ".mp4",
    ".ogg",
    ".opus",
    ".wav",
    ".webm",
}
LANGUAGE_NAMES = {
    "cs": "Czech",
    "cz": "Czech",
    "en": "English",
    "sk": "Slovak",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "pl": "Polish",
    "uk": "Ukrainian",
    "ru": "Russian",
}


@dataclass(frozen=True)
class SignalFinding:
    kind: str
    severity: str
    message: str
    start_ms: int | None = None
    end_ms: int | None = None
    evidence: dict[str, Any] | None = None


class ModulateAPIError(RuntimeError):
    def __init__(self, service: str, status_code: int, body: str):
        self.service = service
        self.status_code = status_code
        self.body = body
        super().__init__(
            f"{service} returned HTTP {status_code}: {body or '<empty response body>'}"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "service": self.service,
            "status_code": self.status_code,
            "body": self.body,
        }


def raise_for_modulate_error(response: Any, service: str) -> None:
    status_code = int(getattr(response, "status_code", 0))
    if status_code < 400:
        return
    body = str(getattr(response, "text", ""))
    raise ModulateAPIError(service, status_code, body)


def json_text(value: Any, **kwargs: Any) -> str:
    return json.dumps(value, ensure_ascii=False, **kwargs)


def language_name(language_code: Any) -> str | None:
    if language_code is None:
        return None
    normalized = str(language_code).split("-")[0].lower()
    return LANGUAGE_NAMES.get(normalized)


def compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classify_overall(findings: list[SignalFinding]) -> str:
    strong = [finding for finding in findings if finding.severity == "strong"]
    warnings = [finding for finding in findings if finding.severity == "warning"]

    if len(strong) > 1 or (strong and warnings):
        return "multiple_strong_indications"
    if any(finding.kind == "deepfake" for finding in strong):
        return "possible_synthetic_voice_detected"
    if strong or warnings:
        return "possible_editing_detected"
    return "inconclusive"


def find_audio_anomalies(
    samples: list[int],
    sample_rate: int,
    window_ms: int = DEFAULT_WINDOW_MS,
    ratio_threshold: float = DEFAULT_RMS_RATIO_THRESHOLD,
    silence_floor: float = 100.0,
) -> list[SignalFinding]:
    window_size = max(1, int(sample_rate * window_ms / 1000))
    rms_values: list[float] = []

    for offset in range(0, len(samples), window_size):
        window = samples[offset : offset + window_size]
        if not window:
            continue
        square_mean = sum(sample * sample for sample in window) / len(window)
        rms_values.append(math.sqrt(square_mean))

    findings: list[SignalFinding] = []
    for index in range(1, len(rms_values)):
        previous = max(rms_values[index - 1], 1.0)
        current = max(rms_values[index], 1.0)
        if min(previous, current) < silence_floor:
            continue
        ratio = max(previous, current) / min(previous, current)

        if ratio >= ratio_threshold:
            start_ms = (index - 1) * window_ms
            end_ms = (index + 1) * window_ms
            findings.append(
                SignalFinding(
                    kind="continuity",
                    severity="warning",
                    message="Abrupt RMS energy change between adjacent windows",
                    start_ms=start_ms,
                    end_ms=end_ms,
                    evidence={
                        "previous_rms": round(previous, 2),
                        "current_rms": round(current, 2),
                        "ratio": round(ratio, 2),
                    },
                )
            )

    return findings


def run_json_command(command: list[str]) -> dict[str, Any] | None:
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError:
        return None


def probe_media(path: Path) -> dict[str, Any] | None:
    return run_json_command(
        [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ]
    )


def metadata_findings(path: Path, probe: dict[str, Any] | None) -> list[SignalFinding]:
    findings: list[SignalFinding] = []
    if probe is None:
        return [
            SignalFinding(
                "metadata",
                "info",
                "ffprobe metadata was not available; container analysis skipped",
            )
        ]

    format_info = probe.get("format", {})
    format_name = str(format_info.get("format_name", ""))
    tags = format_info.get("tags", {}) or {}
    encoder = str(tags.get("encoder", ""))

    suffix = path.suffix.lower().lstrip(".")
    if suffix and suffix not in format_name.lower():
        common_aliases = {"mp3": "mp3", "wav": "wav", "m4a": "mov,mp4,m4a"}
        expected = common_aliases.get(suffix, suffix)
        if expected not in format_name.lower():
            findings.append(
                SignalFinding(
                    "metadata",
                    "warning",
                    "File extension does not clearly match container format",
                    evidence={"extension": suffix, "format_name": format_name},
                )
            )

    if encoder:
        lowered = encoder.lower()
        if any(name in lowered for name in ("lavf", "ffmpeg", "audacity")):
            findings.append(
                SignalFinding(
                    "metadata",
                    "warning",
                    "Encoder metadata indicates the file may be an export or re-encode",
                    evidence={"encoder": encoder},
                )
            )

    streams = probe.get("streams", [])
    audio_streams = [
        stream for stream in streams if stream.get("codec_type") == "audio"
    ]
    if len(audio_streams) != 1:
        findings.append(
            SignalFinding(
                "metadata",
                "warning",
                "Expected one audio stream",
                evidence={"audio_stream_count": len(audio_streams)},
            )
        )

    return findings


def decode_audio_samples(
    path: Path,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    max_seconds: int = MAX_DECODE_SECONDS,
) -> list[int] | None:
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-t",
        str(max_seconds),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "s16le",
        "-",
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    raw = completed.stdout
    if len(raw) < 2:
        return []
    if len(raw) % 2:
        raw = raw[:-1]
    return [sample[0] for sample in struct.iter_unpack("<h", raw)]


def convert_to_wav(
    input_file: Path,
    output_file: Path,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> dict[str, Any]:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        str(input_file),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        str(output_file),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError:
        return {
            "status": "skipped",
            "reason": "ffmpeg executable was not found",
            "path": str(output_file),
        }
    except subprocess.CalledProcessError as exc:
        return {
            "status": "failed",
            "reason": exc.stderr.strip() or str(exc),
            "path": str(output_file),
        }

    return {
        "status": "converted",
        "path": str(output_file),
        "sample_rate": sample_rate,
        "num_channels": 1,
        "codec": "pcm_s16le",
        "size_bytes": output_file.stat().st_size,
        "sha256": compute_sha256(output_file),
    }


def audio_stats(samples: list[int]) -> dict[str, Any]:
    if not samples:
        return {"sample_count": 0}

    abs_values = [abs(sample) for sample in samples]
    clipping_count = sum(1 for value in abs_values if value >= 32760)
    mean_abs = statistics.fmean(abs_values)
    peak = max(abs_values)
    return {
        "sample_count": len(samples),
        "mean_abs": round(mean_abs, 2),
        "peak_abs": peak,
        "clipping_samples": clipping_count,
    }


def inputs_from_args(
    input_files: list[Path],
    all_local_audio: bool,
    cwd: Path = Path("."),
) -> list[Path]:
    if input_files:
        return input_files
    if not all_local_audio:
        return []
    return sorted(
        path
        for path in cwd.iterdir()
        if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES
    )


def run_modulate_batch(
    path: Path,
    api_key: str,
    confidence_threshold: float,
) -> tuple[dict[str, Any] | None, list[SignalFinding]]:
    from deepfake_detect import detect_batch_result, summarize_frames

    result = detect_batch_result(path, api_key)
    frames = result["frames"]
    summary = summarize_frames(frames, confidence_threshold)

    findings: list[SignalFinding] = []
    for frame in frames:
        if (
            frame.get("verdict") == "synthetic"
            and float(frame.get("confidence", 0.0)) >= confidence_threshold
        ):
            findings.append(
                SignalFinding(
                    "deepfake",
                    "strong",
                    "Synthetic voice frame above confidence threshold",
                    start_ms=int(frame["start_time_ms"]),
                    end_ms=int(frame["end_time_ms"]),
                    evidence={"confidence": float(frame["confidence"])},
                )
            )

    return {
        "summary": asdict(summary),
        "frames": frames,
    }, findings


def run_modulate_stt_enrichment(path: Path, api_key: str) -> dict[str, Any]:
    import requests

    with path.open("rb") as file_obj:
        response = requests.post(
            STT_BATCH_ENDPOINT,
            headers={"X-API-Key": api_key},
            data={
                "speaker_diarization": True,
                "emotion_signal": True,
                "accent_signal": True,
                "deepfake_signal": True,
                "pii_phi_tagging": False,
            },
            files={"upload_file": file_obj},
            timeout=180,
        )
    raise_for_modulate_error(response, "stt-enrichment")
    return response.json()


def _extract_score(payload: Any) -> tuple[str | None, float | None]:
    if isinstance(payload, dict):
        label = (
            payload.get("verdict")
            or payload.get("label")
            or payload.get("value")
            or payload.get("prediction")
        )
        confidence = (
            payload.get("confidence")
            or payload.get("score")
            or payload.get("probability")
        )
        return str(label) if label is not None else None, (
            float(confidence) if confidence is not None else None
        )
    if isinstance(payload, str):
        return payload, None
    return None, None


def stt_enrichment_findings(
    result: dict[str, Any],
    confidence_threshold: float,
) -> list[SignalFinding]:
    findings: list[SignalFinding] = []
    for utterance in result.get("utterances", []):
        start_ms = int(utterance.get("start_ms", 0))
        duration_ms = int(
            utterance.get("duration_ms")
            or utterance.get("end_ms", start_ms) - start_ms
        )
        end_ms = int(utterance.get("end_ms", start_ms + duration_ms))
        speaker = utterance.get("speaker")
        language = utterance.get("language")
        utterance_context = {
            "speaker": speaker,
            "language": language,
            "language_name": language_name(language),
            "text": utterance.get("text", ""),
        }

        deepfake_payload = (
            utterance.get("deepfake")
            or utterance.get("deepfake_signal")
            or utterance.get("synthetic_voice")
        )
        verdict, confidence = _extract_score(deepfake_payload)
        if verdict and verdict == "synthetic" and (
            confidence is None or confidence >= confidence_threshold
        ):
            findings.append(
                SignalFinding(
                    "deepfake",
                    "strong",
                    "STT enrichment marked utterance as synthetic",
                    start_ms=start_ms,
                    end_ms=end_ms,
                    evidence={
                        **utterance_context,
                        "confidence": confidence,
                    },
                )
            )

        emotion_payload = utterance.get("emotion") or utterance.get("emotion_signal")
        emotion_label, emotion_confidence = _extract_score(emotion_payload)
        accent_payload = utterance.get("accent") or utterance.get("accent_signal")
        accent_label, accent_confidence = _extract_score(accent_payload)

        if emotion_label or accent_label or utterance_context["text"]:
            signal_parts = []
            if emotion_label:
                signal_parts.append(f"emotion={emotion_label}")
            if accent_label:
                signal_parts.append(f"accent={accent_label}")
            if not signal_parts:
                signal_parts.append("transcript")

            findings.append(
                SignalFinding(
                    "stt-utterance",
                    "info",
                    "STT utterance signals: " + ", ".join(signal_parts),
                    start_ms=start_ms,
                    end_ms=end_ms,
                    evidence={
                        **utterance_context,
                        "emotion": emotion_label,
                        "emotion_confidence": emotion_confidence,
                        "accent": accent_label,
                        "accent_confidence": accent_confidence,
                    },
                )
            )

    return findings


def build_report(
    input_file: Path,
    include_deepfake: bool,
    include_stt_enrichment: bool = False,
    api_key: str | None = None,
    confidence_threshold: float = 0.8,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    window_ms: int = DEFAULT_WINDOW_MS,
    converted_wav: dict[str, Any] | None = None,
) -> dict[str, Any]:
    input_file = input_file.resolve()
    findings: list[SignalFinding] = []

    probe = probe_media(input_file)
    findings.extend(metadata_findings(input_file, probe))

    samples = decode_audio_samples(input_file, sample_rate=sample_rate)
    if samples is None:
        audio_analysis = {
            "status": "skipped",
            "reason": "ffmpeg decode was not available or failed",
        }
        findings.append(
            SignalFinding(
                "continuity",
                "info",
                "Audio continuity analysis skipped because decoding failed",
            )
        )
    else:
        audio_analysis = {
            "status": "analyzed",
            "sample_rate": sample_rate,
            "window_ms": window_ms,
            "stats": audio_stats(samples),
        }
        findings.extend(
            find_audio_anomalies(
                samples,
                sample_rate=sample_rate,
                window_ms=window_ms,
            )
        )

    deepfake_result = None
    stt_enrichment_result = None
    stt_enrichment_error = None
    if include_deepfake:
        if not api_key:
            findings.append(
                SignalFinding(
                    "deepfake",
                    "info",
                    "Deepfake analysis requested but no API key was provided",
                )
            )
        else:
            try:
                deepfake_result, deepfake_findings = run_modulate_batch(
                    input_file,
                    api_key,
                    confidence_threshold,
                )
                findings.extend(deepfake_findings)
            except Exception as exc:
                findings.append(
                    SignalFinding(
                        "deepfake",
                        "info",
                        f"Deepfake analysis failed: {exc}",
                    )
                )

    if include_stt_enrichment:
        if not api_key:
            findings.append(
                SignalFinding(
                    "stt-enrichment",
                    "info",
                    "STT enrichment requested but no API key was provided",
                )
            )
        else:
            try:
                stt_input = input_file
                if converted_wav and converted_wav.get("status") == "converted":
                    stt_input = Path(str(converted_wav["path"]))
                stt_enrichment_result = run_modulate_stt_enrichment(stt_input, api_key)
                findings.extend(
                    stt_enrichment_findings(
                        stt_enrichment_result,
                        confidence_threshold,
                    )
                )
            except ModulateAPIError as exc:
                stt_enrichment_error = exc.as_dict()
                findings.append(
                    SignalFinding(
                        "stt-enrichment",
                        "info",
                        f"STT enrichment failed: {exc}",
                        evidence=exc.as_dict(),
                    )
                )
            except Exception as exc:
                findings.append(
                    SignalFinding(
                        "stt-enrichment",
                        "info",
                        f"STT enrichment failed: {exc}",
                    )
                )

    report = {
        "input": {
            "path": str(input_file),
            "filename": input_file.name,
            "size_bytes": input_file.stat().st_size,
            "sha256": compute_sha256(input_file),
        },
        "metadata": probe,
        "audio_analysis": audio_analysis,
        "converted_wav": converted_wav,
        "deepfake_analysis": deepfake_result,
        "stt_enrichment": stt_enrichment_result,
        "stt_enrichment_error": stt_enrichment_error,
        "findings": [asdict(finding) for finding in findings],
        "overall_conclusion": classify_overall(findings),
        "limitations": [
            "This report identifies technical indications; it does not prove authenticity by itself.",
            "A forensic conclusion depends on access to the original recording and chain of custody.",
            "Absence of detected anomalies is not proof that no editing occurred.",
        ],
    }
    return report


def print_progress(message: str) -> None:
    print(message, flush=True)


def print_progress_summary(report: dict[str, Any], output=sys.stdout) -> None:
    print(f"File:     {report['input']['filename']}", file=output)
    print(f"Size:     {report['input']['size_bytes']} bytes", file=output)
    print(f"SHA-256:  {report['input']['sha256']}", file=output)
    print(f"Audio:    {report['audio_analysis']['status']}", file=output)
    if report["audio_analysis"].get("stats"):
        stats = report["audio_analysis"]["stats"]
        print(
            "Stats:    "
            f"{stats.get('sample_count', 0)} samples, "
            f"peak={stats.get('peak_abs')}, "
            f"clipping={stats.get('clipping_samples')}",
            file=output,
        )

    print(f"Conclusion: {report['overall_conclusion']}", file=output)
    print("Findings:", file=output)
    findings = report.get("findings", [])
    if not findings:
        print("  none", file=output)
    for finding in findings:
        time_part = ""
        if finding.get("start_ms") is not None:
            time_part = f" {finding['start_ms']}ms-{finding['end_ms']}ms"
        print(
            f"  [{finding['severity']}] {finding['kind']}{time_part}: "
            f"{finding['message']}",
            file=output,
        )
        if finding.get("evidence"):
            print(
                f"    evidence={json_text(finding['evidence'], sort_keys=True)}",
                file=output,
            )
    print("", file=output)
    output.flush()


def modulate_verdict_label(verdict: str) -> str:
    labels = {
        "synthetic": "Synthetic",
        "non-synthetic": "Human",
        "no-content": "No content",
    }
    return labels.get(verdict, verdict)


def print_modulate_deepfake_table(
    report: dict[str, Any],
    output=sys.stdout,
) -> None:
    deepfake_analysis = report.get("deepfake_analysis")
    if not deepfake_analysis:
        return

    frames = deepfake_analysis.get("frames", [])
    duration_ms = deepfake_analysis.get("duration_ms")
    frame_count = deepfake_analysis.get("frame_count", len(frames))

    print("Modulate Deepfake Batch:", file=output)
    print(f"File:     {report['input']['filename']}", file=output)
    if duration_ms is not None:
        print(f"Duration: {duration_ms} ms", file=output)
    print(f"Frames:   {frame_count}", file=output)
    print("", file=output)

    for frame in frames:
        label = modulate_verdict_label(str(frame["verdict"]))
        confidence = float(frame["confidence"])
        print(
            f"  {frame['start_time_ms']:>7}ms - "
            f"{frame['end_time_ms']:>7}ms  "
            f"{label:<12} confidence={confidence:.2%}",
            file=output,
        )
    print("", file=output)
    output.flush()


def print_modulate_outputs(report: dict[str, Any], output=sys.stdout) -> None:
    deepfake_analysis = report.get("deepfake_analysis")
    stt_enrichment = report.get("stt_enrichment")
    stt_enrichment_error = report.get("stt_enrichment_error")

    if deepfake_analysis:
        print("Raw Modulate deepfake batch response:", file=output)
        print(json_text(deepfake_analysis, indent=2, sort_keys=True), file=output)
        print("", file=output)

    if stt_enrichment:
        print("Raw Modulate STT enrichment response:", file=output)
        print(json_text(stt_enrichment, indent=2, sort_keys=True), file=output)
        print("", file=output)

    if stt_enrichment_error:
        print("Raw Modulate STT enrichment error:", file=output)
        print(json_text(stt_enrichment_error, indent=2, sort_keys=True), file=output)
        print("", file=output)

    output.flush()


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Audio Authenticity Report",
        "",
        "## Evidence Manifest",
        "",
        f"- File: `{report['input']['filename']}`",
        f"- Size: `{report['input']['size_bytes']}` bytes",
        f"- SHA-256: `{report['input']['sha256']}`",
        "",
        "## Overall Conclusion",
        "",
        f"`{report['overall_conclusion']}`",
        "",
        "## Findings",
        "",
    ]

    findings = report["findings"]
    if not findings:
        lines.append("- No technical findings were generated.")
    for finding in findings:
        time_part = ""
        if finding.get("start_ms") is not None:
            time_part = f" ({finding['start_ms']}ms-{finding['end_ms']}ms)"
        lines.append(
            f"- `{finding['severity']}` `{finding['kind']}`{time_part}: "
            f"{finding['message']}"
        )
        if finding.get("evidence"):
            lines.append(f"  Evidence: `{json_text(finding['evidence'], sort_keys=True)}`")

    lines.extend(
        [
            "",
            "## Audio Analysis",
            "",
            f"- Status: `{report['audio_analysis']['status']}`",
        ]
    )
    if report["audio_analysis"].get("stats"):
        lines.append(
            f"- Stats: `{json_text(report['audio_analysis']['stats'], sort_keys=True)}`"
        )
    if report.get("converted_wav"):
        lines.extend(
            [
                "",
                "## Converted WAV Artifact",
                "",
                f"- Status: `{report['converted_wav']['status']}`",
                f"- Path: `{report['converted_wav'].get('path')}`",
            ]
        )
        if report["converted_wav"].get("sha256"):
            lines.append(f"- SHA-256: `{report['converted_wav']['sha256']}`")

    if report.get("deepfake_analysis"):
        summary = report["deepfake_analysis"].get("summary")
        frames = report["deepfake_analysis"].get("frames", [])
        table_lines = [
            "| Start | End | Verdict | Confidence |",
            "| ---: | ---: | --- | ---: |",
        ]
        for frame in frames:
            start_ms = frame.get("start_time_ms", "?")
            end_ms = frame.get("end_time_ms", "?")
            verdict = modulate_verdict_label(str(frame.get("verdict", "?")))
            confidence = frame.get("confidence")
            confidence_text = (
                f"{float(confidence):.2%}" if confidence is not None else "?"
            )
            table_lines.append(
                "| "
                f"{start_ms}ms | "
                f"{end_ms}ms | "
                f"{verdict} | "
                f"{confidence_text} |"
            )
        lines.extend(
            [
                "",
                "## Deepfake Analysis",
                "",
                f"- Summary: `{json_text(summary, sort_keys=True)}`",
                "",
                *table_lines,
                "",
                "### Raw Deepfake Batch Response",
                "",
                "```json",
                json_text(report["deepfake_analysis"], indent=2, sort_keys=True),
                "```",
            ]
        )

    if report.get("stt_enrichment"):
        stt = report["stt_enrichment"]
        lines.extend(
            [
                "",
                "## STT Enrichment",
                "",
                f"- Transcript length: `{len(stt.get('text', ''))}` characters",
                f"- Utterances: `{len(stt.get('utterances', []))}`",
                "",
                "### Raw STT Enrichment Response",
                "",
                "```json",
                json_text(stt, indent=2, sort_keys=True),
                "```",
            ]
        )

    if report.get("stt_enrichment_error"):
        lines.extend(
            [
                "",
                "## STT Enrichment Error",
                "",
                "```json",
                json_text(
                    report["stt_enrichment_error"],
                    indent=2,
                    sort_keys=True,
                ),
                "```",
            ]
        )

    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {limitation}" for limitation in report["limitations"])
    lines.append("")
    return "\n".join(lines)


def write_report(
    report: dict[str, Any],
    output_dir: Path,
    write_wav: bool = True,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if write_wav and not report.get("converted_wav"):
        input_path = report.get("input", {}).get("path")
        if input_path:
            report["converted_wav"] = convert_to_wav(
                Path(input_path),
                output_dir / "source.wav",
            )

    (output_dir / "report.json").write_text(
        json_text(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(render_markdown(report), encoding="utf-8")
    if report.get("deepfake_analysis"):
        (output_dir / "modulate_deepfake_batch.json").write_text(
            json_text(report["deepfake_analysis"], indent=2, sort_keys=True),
            encoding="utf-8",
        )
    if report.get("stt_enrichment"):
        (output_dir / "modulate_stt_enrichment.json").write_text(
            json_text(report["stt_enrichment"], indent=2, sort_keys=True),
            encoding="utf-8",
        )
    if report.get("stt_enrichment_error"):
        (output_dir / "modulate_stt_enrichment_error.json").write_text(
            json_text(report["stt_enrichment_error"], indent=2, sort_keys=True),
            encoding="utf-8",
        )


def resolve_api_key(api_key_file: Path | None) -> str | None:
    api_key = os.environ.get("MODULATE_API_KEY")
    if api_key:
        return api_key.strip()
    if api_key_file and api_key_file.exists():
        return api_key_file.read_text(encoding="utf-8").strip()
    return None


def existing_file(value: str) -> Path:
    path = Path(value)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"file does not exist: {value}")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a forensic-style audio authenticity report.",
        epilog=(
            "Examples:\n"
            "  python3 audio_authenticity_report.py test-voice.mp3 --out reports/test-voice\n"
            "  python3 audio_authenticity_report.py test-voice.mp3 pocasi_dialog.wav --out reports/batch\n"
            "  python3 audio_authenticity_report.py --all-local-audio --out reports/all-local\n"
            "  python3 audio_authenticity_report.py --all-local-audio --modulate --out reports/full-modulate\n"
            "  python3 audio_authenticity_report.py test-voice.mp3 --modulate --api-key-file token --out reports/test-voice-full"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input_file", nargs="*", type=existing_file)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory. Default: reports/<input-stem>",
    )
    parser.add_argument(
        "--deepfake",
        action="store_true",
        help="Include Modulate batch synthetic voice detection.",
    )
    parser.add_argument(
        "--stt-enrichment",
        action="store_true",
        help="Include Modulate STT with diarization, emotion, accent, and deepfake signal.",
    )
    parser.add_argument(
        "--modulate",
        action="store_true",
        help="Run all Modulate-backed analyses: deepfake batch and STT enrichment.",
    )
    parser.add_argument(
        "--all-local-audio",
        action="store_true",
        help="Analyze all supported audio files in the current directory.",
    )
    parser.add_argument(
        "--show-raw-modulate",
        action="store_true",
        help="Print raw Modulate JSON responses/errors to the terminal.",
    )
    parser.add_argument("--api-key-file", type=Path, default=Path("token"))
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--window-ms", type=int, default=DEFAULT_WINDOW_MS)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    raw_args = sys.argv[1:] if argv is None else argv
    if not raw_args:
        parser.print_help()
        return 2
    args = parser.parse_args(raw_args)

    input_files = inputs_from_args(
        args.input_file,
        all_local_audio=args.all_local_audio,
        cwd=Path("."),
    )
    if not input_files:
        parser.error("provide at least one input_file or use --all-local-audio")

    include_deepfake = args.deepfake or args.modulate
    include_stt_enrichment = args.stt_enrichment or args.modulate
    api_key = resolve_api_key(args.api_key_file)
    base_output_dir = args.out or Path("reports")

    for input_file in input_files:
        print_progress(f"==> Analyzing {input_file}")
        print_progress("    local: hashing, metadata, and audio continuity")
        if include_deepfake and api_key:
            print_progress("    modulate: deepfake batch enabled")
        elif include_deepfake:
            print_progress("    modulate: deepfake batch requested without API key")
        if include_stt_enrichment and api_key:
            print_progress("    modulate: STT enrichment enabled")
        elif include_stt_enrichment:
            print_progress("    modulate: STT enrichment requested without API key")

        output_dir = (
            base_output_dir / input_file.stem
            if len(input_files) > 1 or args.all_local_audio
            else base_output_dir
        )
        print_progress(f"    local: converting to {output_dir / 'source.wav'}")
        converted_wav = convert_to_wav(
            input_file,
            output_dir / "source.wav",
            sample_rate=args.sample_rate,
        )
        if converted_wav.get("status") == "converted":
            print_progress("    stt-enrichment input: converted WAV")
        elif include_stt_enrichment:
            print_progress(
                "    stt-enrichment input: original file "
                f"(WAV conversion {converted_wav.get('status')})"
            )
        report = build_report(
            input_file,
            include_deepfake=include_deepfake,
            include_stt_enrichment=include_stt_enrichment,
            api_key=api_key,
            confidence_threshold=args.threshold,
            sample_rate=args.sample_rate,
            window_ms=args.window_ms,
            converted_wav=converted_wav,
        )
        print_progress_summary(report)
        print_modulate_deepfake_table(report)
        if args.show_raw_modulate:
            print_modulate_outputs(report)
        write_report(report, output_dir)
        print(f"Wrote {output_dir / 'report.md'}")
        print(f"Wrote {output_dir / 'report.json'}")
        if report.get("deepfake_analysis"):
            print(f"Wrote {output_dir / 'modulate_deepfake_batch.json'}")
        if report.get("stt_enrichment"):
            print(f"Wrote {output_dir / 'modulate_stt_enrichment.json'}")
        if report.get("stt_enrichment_error"):
            print(f"Wrote {output_dir / 'modulate_stt_enrichment_error.json'}")
        if report.get("converted_wav"):
            converted = report["converted_wav"]
            if converted.get("status") == "converted":
                print(f"Wrote {converted['path']}")
            else:
                print(
                    "WAV conversion "
                    f"{converted.get('status')}: {converted.get('reason')}"
                )
        print(f"Conclusion for {input_file}: {report['overall_conclusion']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
