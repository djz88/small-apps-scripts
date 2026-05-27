import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlencode


BATCH_ENDPOINT = (
    "https://modulate-developer-apis.com/api/"
    "velma-2-synthetic-voice-detection-batch"
)
STREAMING_ENDPOINT = (
    "wss://modulate-developer-apis.com/api/"
    "velma-2-synthetic-voice-detection-streaming"
)
DEFAULT_CHUNK_SIZE = 8192
DEFAULT_CONFIDENCE_THRESHOLD = 0.8
VERDICT_LABELS = {
    "synthetic": "Synthetic",
    "non-synthetic": "Human",
    "no-content": "No content",
}


@dataclass(frozen=True)
class DetectionSummary:
    overall_verdict: str
    should_flag: bool
    synthetic_frames: int
    speech_frames: int
    total_frames: int
    confidence_threshold: float


def summarize_frames(
    frames: list[dict[str, Any]], confidence_threshold: float
) -> DetectionSummary:
    speech_frames = [
        frame for frame in frames if frame.get("verdict") != "no-content"
    ]
    synthetic_frames = [
        frame
        for frame in speech_frames
        if frame.get("verdict") == "synthetic"
        and float(frame.get("confidence", 0.0)) >= confidence_threshold
    ]

    if synthetic_frames:
        overall_verdict = "synthetic"
    elif speech_frames:
        overall_verdict = "non-synthetic"
    else:
        overall_verdict = "no-content"

    return DetectionSummary(
        overall_verdict=overall_verdict,
        should_flag=bool(synthetic_frames),
        synthetic_frames=len(synthetic_frames),
        speech_frames=len(speech_frames),
        total_frames=len(frames),
        confidence_threshold=confidence_threshold,
    )


def resolve_api_key(api_key_file: Path | None) -> str:
    api_key = os.environ.get("MODULATE_API_KEY")
    if api_key:
        return api_key.strip()

    if api_key_file and api_key_file.exists():
        return api_key_file.read_text(encoding="utf-8").strip()

    raise RuntimeError(
        "Missing API key. Set MODULATE_API_KEY or pass --api-key-file."
    )


def build_stream_url(
    api_key: str,
    audio_format: str,
    sample_rate: int,
    num_channels: int,
) -> str:
    query = urlencode(
        {
            "api_key": api_key,
            "audio_format": audio_format,
            "sample_rate": str(sample_rate),
            "num_channels": str(num_channels),
        }
    )
    return f"{STREAMING_ENDPOINT}?{query}"


def print_frame(frame: dict[str, Any]) -> None:
    label = VERDICT_LABELS.get(str(frame["verdict"]), str(frame["verdict"]))
    confidence = float(frame["confidence"])
    print(
        f"  {frame['start_time_ms']:>7}ms - {frame['end_time_ms']:>7}ms  "
        f"{label:<12} confidence={confidence:.2%}"
    )


def print_summary(summary: DetectionSummary) -> None:
    label = VERDICT_LABELS.get(summary.overall_verdict, summary.overall_verdict)
    print()
    print(f"Overall: {label}")
    print(
        "Flag:    "
        f"{'yes' if summary.should_flag else 'no'} "
        f"(threshold={summary.confidence_threshold:.0%})"
    )
    print(
        "Frames:  "
        f"{summary.total_frames} total, "
        f"{summary.speech_frames} speech, "
        f"{summary.synthetic_frames} synthetic above threshold"
    )


def detect_batch(
    audio_file: Path,
    api_key: str,
    confidence_threshold: float,
) -> int:
    result = detect_batch_result(audio_file, api_key)
    frames = result["frames"]

    print(f"File:     {result.get('filename') or audio_file.name}")
    print(f"Duration: {result['duration_ms']} ms")
    print(f"Frames:   {len(frames)}")
    print()
    for frame in frames:
        print_frame(frame)

    summary = summarize_frames(frames, confidence_threshold)
    print_summary(summary)
    return 2 if summary.should_flag else 0


def detect_batch_result(audio_file: Path, api_key: str) -> dict[str, Any]:
    import requests

    with audio_file.open("rb") as file_obj:
        response = requests.post(
            BATCH_ENDPOINT,
            headers={"X-API-Key": api_key},
            files={"upload_file": file_obj},
            timeout=120,
        )
    response.raise_for_status()
    return response.json()


async def detect_stream(
    audio_file: Path,
    api_key: str,
    confidence_threshold: float,
    audio_format: str,
    sample_rate: int,
    num_channels: int,
    chunk_size: int,
) -> int:
    import websockets

    frames: list[dict[str, Any]] = []
    url = build_stream_url(api_key, audio_format, sample_rate, num_channels)

    async with websockets.connect(url) as ws:
        async def send_audio() -> None:
            with audio_file.open("rb") as file_obj:
                while chunk := file_obj.read(chunk_size):
                    await ws.send(chunk)
                    await asyncio.sleep(0)
            await ws.send("")

        async def receive_results() -> None:
            async for message in ws:
                msg = json.loads(message)
                msg_type = msg.get("type")

                if msg_type == "frame":
                    frame = msg["frame"]
                    frames.append(frame)
                    print_frame(frame)
                elif msg_type == "done":
                    print()
                    print(
                        f"Done: {msg['duration_ms']} ms, "
                        f"{msg['frame_count']} frames"
                    )
                    break
                elif msg_type == "error":
                    raise RuntimeError(f"Modulate error: {msg['error']}")

        await asyncio.gather(send_audio(), receive_results())

    summary = summarize_frames(frames, confidence_threshold)
    print_summary(summary)
    return 2 if summary.should_flag else 0


def existing_file(value: str) -> Path:
    path = Path(value)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"file does not exist: {value}")
    return path


def confidence_threshold(value: str) -> float:
    threshold = float(value)
    if threshold < 0 or threshold > 1:
        raise argparse.ArgumentTypeError("confidence threshold must be 0..1")
    return threshold


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect synthetic voice with Modulate Velma-2."
    )
    parser.add_argument(
        "--api-key-file",
        type=Path,
        default=Path("token"),
        help="Fallback file for the API key when MODULATE_API_KEY is unset.",
    )
    parser.add_argument(
        "--threshold",
        type=confidence_threshold,
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        help="Minimum synthetic confidence to flag a frame. Default: 0.8",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    batch = subparsers.add_parser(
        "batch",
        help="Analyze a regular audio file such as MP3, WAV, FLAC, OGG, or WebM.",
    )
    batch.add_argument("audio_file", type=existing_file)

    stream = subparsers.add_parser(
        "stream",
        help="Analyze raw PCM audio over the streaming WebSocket API.",
    )
    stream.add_argument("audio_file", type=existing_file)
    stream.add_argument("--audio-format", default="s16le")
    stream.add_argument("--sample-rate", type=int, default=16000)
    stream.add_argument("--num-channels", type=int, default=1)
    stream.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        api_key = resolve_api_key(args.api_key_file)

        if args.command == "batch":
            return detect_batch(args.audio_file, api_key, args.threshold)
        if args.command == "stream":
            return asyncio.run(
                detect_stream(
                    args.audio_file,
                    api_key,
                    args.threshold,
                    args.audio_format,
                    args.sample_rate,
                    args.num_channels,
                    args.chunk_size,
                )
            )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    parser.error(f"unknown command: {args.command}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
