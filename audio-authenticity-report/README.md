# Audio Authenticity Report

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![Modulate.ai](https://img.shields.io/badge/Modulate.ai-Developer%20API-purple)
![FFmpeg](https://img.shields.io/badge/FFmpeg-required-green)
![JSON](https://img.shields.io/badge/Reports-JSON%20%2B%20Markdown-lightgrey)

`audio_authenticity_report.py` generates a forensic-style technical report for
audio recordings. It helps identify indications that a recording may be edited,
re-encoded, synthetic, or otherwise unsuitable to treat as unquestionably
authentic.

The script does not prove authenticity by itself. It collects repeatable
technical evidence: hashes, media metadata, normalized WAV artifacts, continuity
signals, and optional Modulate.ai model outputs.

## What It Uses

- Python 3
- `ffprobe` for media/container metadata
- `ffmpeg` for decoding and normalized WAV conversion
- Modulate Developer APIs from `modulate-developer-apis.com`
- Modulate documentation: <https://docs.modulate.ai/>
- Local JSON and Markdown report generation

The Modulate-backed checks require an API key. The key is read from
`MODULATE_API_KEY` or from a local file passed with `--api-key-file`; by default
the script looks for `token`.

## Main Checks

- Evidence manifest: original path, file size, SHA-256 hash
- Container metadata: codec, format, encoder, stream count
- Normalized WAV artifact: `source.wav`, mono, 16 kHz, PCM s16le
- Audio continuity analysis: abrupt RMS changes across adjacent windows
- Modulate synthetic voice detection batch output
- Modulate STT enrichment output: transcript, speaker, language, emotion,
  accent, and STT deepfake signal

## Setup

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Install FFmpeg tools:

```bash
sudo apt install ffmpeg
```

Set the Modulate API key in one of these ways:

```bash
export MODULATE_API_KEY="your-api-key"
```

or put the key into a local file named `token`. The script uses `token`
automatically by default when `MODULATE_API_KEY` is not set. A template is
provided as `token.example`; copy it to `token` and replace the placeholder with
your Modulate Developer API key. Do not commit the real `token` file.

## Usage

Show help and examples:

```bash
python3 audio_authenticity_report.py
```

Analyze one file with local checks only:

```bash
python3 audio_authenticity_report.py test-voice.mp3 --out reports/test-voice
```

Analyze multiple files:

```bash
python3 audio_authenticity_report.py test-voice.mp3 pocasi_dialog.wav --out reports/batch
```

Analyze all supported local audio files in the current directory:

```bash
python3 audio_authenticity_report.py --all-local-audio --out reports/all-local
```

Run full local + Modulate checks:

```bash
python3 audio_authenticity_report.py --all-local-audio --modulate --out reports/full-modulate
```

Run full checks for a single file:

```bash
python3 audio_authenticity_report.py test-voice.mp3 --modulate --out reports/test-voice-full
```

Print raw Modulate JSON to the terminal as well as saving it:

```bash
python3 audio_authenticity_report.py test-voice.mp3 --modulate --show-raw-modulate --out reports/test-voice-full
```

## Outputs

Each analyzed file gets its own report directory containing:

- `report.md` - readable Markdown report
- `report.json` - structured report data
- `source.wav` - normalized WAV copy used for STT enrichment
- `modulate_deepfake_batch.json` - raw Modulate synthetic voice detection output, when available
- `modulate_stt_enrichment.json` - raw Modulate STT enrichment output, when available
- `modulate_stt_enrichment_error.json` - raw Modulate STT error response, when the API rejects a request

The terminal output shows progress, local findings, Modulate deepfake frame
tables, and paths to generated artifacts. Raw Modulate JSON is not printed by
default; use `--show-raw-modulate` when debugging API output.

## Interpretation

The overall conclusion is conservative. Typical values include:

- `inconclusive`
- `possible_editing_detected`
- `possible_synthetic_voice_detected`
- `multiple_strong_indications`

These are technical indicators, not legal conclusions. For formal proceedings,
preserve the original file, keep chain-of-custody notes, and treat this report as
supporting technical evidence rather than a standalone proof.

## Related Script

`deepfake_detect.py` is a smaller utility focused only on Modulate synthetic
voice detection:

```bash
python3 deepfake_detect.py batch test-voice.mp3
```

It can also use Modulate's streaming WebSocket endpoint for raw PCM audio.

## Troubleshooting

If STT enrichment returns an unsupported format error, use the generated
`source.wav` artifact. The main report script already sends this converted WAV
to Modulate STT enrichment.

If Modulate returns HTTP 400, inspect:

```text
modulate_stt_enrichment_error.json
```

If FFmpeg is missing, install it before running local analysis.


Created with help of Codex.
