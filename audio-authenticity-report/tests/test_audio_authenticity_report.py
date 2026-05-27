import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from audio_authenticity_report import (
    ModulateAPIError,
    SignalFinding,
    build_report,
    build_parser,
    classify_overall,
    compute_sha256,
    find_audio_anomalies,
    inputs_from_args,
    print_modulate_deepfake_table,
    print_modulate_outputs,
    raise_for_modulate_error,
    convert_to_wav,
    run_modulate_stt_enrichment,
    print_progress_summary,
    stt_enrichment_findings,
    write_report,
)


class AudioAuthenticityReportTests(unittest.TestCase):
    def test_compute_sha256_hashes_file_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.bin"
            path.write_bytes(b"abc")

            self.assertEqual(
                compute_sha256(path),
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            )

    def test_classify_overall_prefers_multiple_strong_indications(self):
        findings = [
            SignalFinding("metadata", "warning", "container was re-encoded"),
            SignalFinding("deepfake", "strong", "synthetic voice detected"),
        ]

        self.assertEqual(
            classify_overall(findings), "multiple_strong_indications"
        )

    def test_classify_overall_reports_inconclusive_without_findings(self):
        self.assertEqual(classify_overall([]), "inconclusive")

    def test_find_audio_anomalies_flags_abrupt_rms_changes(self):
        samples = [800] * 1600 + [20000] * 1600 + [900] * 1600

        findings = find_audio_anomalies(
            samples,
            sample_rate=16000,
            window_ms=100,
            ratio_threshold=8.0,
        )

        self.assertTrue(any(f.kind == "continuity" for f in findings))

    def test_find_audio_anomalies_ignores_speech_silence_boundaries(self):
        samples = [0] * 1600 + [20000] * 1600 + [0] * 1600

        findings = find_audio_anomalies(
            samples,
            sample_rate=16000,
            window_ms=100,
            ratio_threshold=8.0,
        )

        self.assertEqual(findings, [])

    def test_build_report_contains_evidence_manifest_without_external_tools(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.raw"
            path.write_bytes(b"\x00" * 32)

            report = build_report(path, include_deepfake=False)

            self.assertEqual(report["input"]["filename"], "sample.raw")
            self.assertEqual(report["input"]["size_bytes"], 32)
            self.assertIn("sha256", report["input"])
            self.assertIn("overall_conclusion", report)

    def test_convert_to_wav_writes_pcm_wav_with_ffmpeg(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "sample.wav"
            target = root / "source.wav"
            source.write_bytes(
                b"RIFF$\x00\x00\x00WAVEfmt "
                b"\x10\x00\x00\x00\x01\x00\x01\x00"
                b"\x40\x1f\x00\x00\x80>\x00\x00"
                b"\x02\x00\x10\x00data\x00\x00\x00\x00"
            )

            result = convert_to_wav(source, target)

            self.assertEqual(result["status"], "converted")
            self.assertTrue(target.exists())
            self.assertGreater(result["size_bytes"], 0)

    def test_write_report_saves_converted_wav_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "sample.wav"
            source.write_bytes(
                b"RIFF$\x00\x00\x00WAVEfmt "
                b"\x10\x00\x00\x00\x01\x00\x01\x00"
                b"\x40\x1f\x00\x00\x80>\x00\x00"
                b"\x02\x00\x10\x00data\x00\x00\x00\x00"
            )
            report = {
                "input": {
                    "filename": "sample.wav",
                    "size_bytes": source.stat().st_size,
                    "sha256": compute_sha256(source),
                    "path": str(source),
                },
                "audio_analysis": {"status": "analyzed"},
                "deepfake_analysis": None,
                "stt_enrichment": None,
                "findings": [],
                "overall_conclusion": "inconclusive",
                "limitations": [],
            }
            output_dir = root / "report"

            write_report(report, output_dir, write_wav=True)

            self.assertTrue((output_dir / "source.wav").exists())
            report_json = json.loads((output_dir / "report.json").read_text())
            self.assertEqual(report_json["converted_wav"]["status"], "converted")

    def test_build_report_uses_converted_wav_for_stt_enrichment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "sample.wav"
            converted = root / "source.wav"
            source.write_bytes(
                b"RIFF$\x00\x00\x00WAVEfmt "
                b"\x10\x00\x00\x00\x01\x00\x01\x00"
                b"\x40\x1f\x00\x00\x80>\x00\x00"
                b"\x02\x00\x10\x00data\x00\x00\x00\x00"
            )
            converted.write_bytes(source.read_bytes())
            calls = []

            def fake_stt(path, api_key):
                calls.append(path)
                return {"text": "ok", "utterances": []}

            with patch(
                "audio_authenticity_report.run_modulate_stt_enrichment",
                side_effect=fake_stt,
            ):
                report = build_report(
                    source,
                    include_deepfake=False,
                    include_stt_enrichment=True,
                    api_key="secret",
                    converted_wav={
                        "status": "converted",
                        "path": str(converted),
                    },
                )

            self.assertEqual(calls, [converted])
            self.assertEqual(report["stt_enrichment"]["text"], "ok")

    def test_inputs_from_args_expands_all_local_audio(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mp3 = root / "voice.mp3"
            wav = root / "dialog.wav"
            txt = root / "notes.txt"
            mp3.write_bytes(b"mp3")
            wav.write_bytes(b"wav")
            txt.write_text("ignore", encoding="utf-8")

            inputs = inputs_from_args([], all_local_audio=True, cwd=root)

            self.assertEqual(inputs, [wav, mp3])

    def test_stt_enrichment_findings_merges_utterance_emotion_and_accent(self):
        result = {
            "utterances": [
                {
                    "start_ms": 1000,
                    "duration_ms": 2500,
                    "speaker": 2,
                    "language": "en",
                    "text": "hello",
                    "deepfake": {"verdict": "synthetic", "confidence": 0.93},
                    "emotion": {"label": "angry", "confidence": 0.81},
                    "accent": {"label": "en-US", "confidence": 0.76},
                }
            ]
        }

        findings = stt_enrichment_findings(result, confidence_threshold=0.8)

        self.assertEqual(findings[0].kind, "deepfake")
        self.assertEqual(findings[0].severity, "strong")
        self.assertEqual(findings[0].start_ms, 1000)
        self.assertEqual(findings[0].end_ms, 3500)
        self.assertEqual(findings[0].evidence["language"], "en")
        self.assertEqual(findings[0].evidence["text"], "hello")
        utterance = next(f for f in findings if f.kind == "stt-utterance")
        self.assertEqual(utterance.evidence["language"], "en")
        self.assertEqual(utterance.evidence["text"], "hello")
        self.assertEqual(utterance.evidence["emotion"], "angry")
        self.assertEqual(utterance.evidence["emotion_confidence"], 0.81)
        self.assertEqual(utterance.evidence["accent"], "en-US")
        self.assertEqual(utterance.evidence["accent_confidence"], 0.76)
        self.assertFalse(any(f.kind == "stt-emotion" for f in findings))
        self.assertFalse(any(f.kind == "stt-accent" for f in findings))

    def test_stt_evidence_keeps_czech_text_and_language_name(self):
        result = {
            "utterances": [
                {
                    "start_ms": 600,
                    "end_ms": 14220,
                    "speaker": 1,
                    "language": "cs",
                    "text": "Já teda zjistím, jak ty ceny",
                    "emotion": {"label": "Neutral"},
                }
            ]
        }

        findings = stt_enrichment_findings(result, confidence_threshold=0.8)
        emotion = next(f for f in findings if f.kind == "stt-utterance")

        self.assertEqual(emotion.evidence["language"], "cs")
        self.assertEqual(emotion.evidence["language_name"], "Czech")
        self.assertEqual(emotion.evidence["text"], "Já teda zjistím, jak ty ceny")

    def test_progress_summary_prints_czech_text_without_unicode_escapes(self):
        report = {
            "input": {
                "filename": "voice.wav",
                "size_bytes": 10,
                "sha256": "abcdef123456",
            },
            "audio_analysis": {"status": "analyzed"},
            "findings": [
                {
                    "kind": "stt-emotion",
                    "severity": "info",
                    "message": "Emotion signal: Neutral",
                    "start_ms": 600,
                    "end_ms": 14220,
                    "evidence": {
                        "confidence": None,
                        "language": "cs",
                        "language_name": "Czech",
                        "speaker": 1,
                        "text": "Já teda zjistím, jak ty ceny",
                    },
                }
            ],
            "overall_conclusion": "inconclusive",
        }

        with tempfile.TemporaryFile("w+") as output:
            print_progress_summary(report, output=output)
            output.seek(0)
            text = output.read()

        self.assertIn("Já teda zjistím", text)
        self.assertNotIn("\\u00e1", text)

    def test_print_progress_summary_includes_findings(self):
        report = {
            "input": {
                "filename": "voice.mp3",
                "size_bytes": 10,
                "sha256": "abcdef123456",
            },
            "audio_analysis": {
                "status": "analyzed",
                "stats": {"sample_count": 16000},
            },
            "findings": [
                {
                    "kind": "continuity",
                    "severity": "warning",
                    "message": "Abrupt RMS energy change",
                    "start_ms": 1000,
                    "end_ms": 1250,
                    "evidence": {"ratio": 9.1},
                }
            ],
            "overall_conclusion": "possible_editing_detected",
        }

        with tempfile.TemporaryFile("w+") as output:
            print_progress_summary(report, output=output)
            output.seek(0)
            text = output.read()

        self.assertIn("voice.mp3", text)
        self.assertIn("possible_editing_detected", text)
        self.assertIn("continuity", text)
        self.assertIn("1000ms-1250ms", text)

    def test_print_modulate_outputs_includes_raw_json(self):
        report = {
            "deepfake_analysis": {
                "frames": [
                    {
                        "start_time_ms": 0,
                        "end_time_ms": 3000,
                        "verdict": "synthetic",
                        "confidence": 0.93,
                    }
                ]
            },
            "stt_enrichment": {
                "text": "hello",
                "utterances": [
                    {
                        "start_ms": 0,
                        "duration_ms": 1200,
                        "deepfake": {"verdict": "synthetic", "confidence": 0.91},
                    }
                ],
            },
        }

        with tempfile.TemporaryFile("w+") as output:
            print_modulate_outputs(report, output=output)
            output.seek(0)
            text = output.read()

        self.assertIn("Raw Modulate deepfake batch response", text)
        self.assertIn('"verdict": "synthetic"', text)
        self.assertIn("Raw Modulate STT enrichment response", text)
        self.assertIn('"text": "hello"', text)

    def test_print_modulate_deepfake_table_matches_original_batch_style(self):
        report = {
            "input": {"filename": "test-voice.mp3"},
            "deepfake_analysis": {
                "duration_ms": 55379,
                "frame_count": 2,
                "frames": [
                    {
                        "start_time_ms": 0,
                        "end_time_ms": 4000,
                        "verdict": "no-content",
                        "confidence": 1.0,
                    },
                    {
                        "start_time_ms": 28000,
                        "end_time_ms": 32000,
                        "verdict": "non-synthetic",
                        "confidence": 0.9084,
                    },
                ],
            },
        }

        with tempfile.TemporaryFile("w+") as output:
            print_modulate_deepfake_table(report, output=output)
            output.seek(0)
            text = output.read()

        self.assertIn("Modulate Deepfake Batch:", text)
        self.assertIn("File:     test-voice.mp3", text)
        self.assertIn("Duration: 55379 ms", text)
        self.assertIn("Frames:   2", text)
        self.assertIn("0ms -    4000ms  No content", text)
        self.assertIn("28000ms -   32000ms  Human", text)
        self.assertIn("confidence=90.84%", text)

    def test_write_report_saves_raw_modulate_outputs(self):
        report = {
            "input": {
                "filename": "voice.mp3",
                "size_bytes": 10,
                "sha256": "abcdef123456",
            },
            "audio_analysis": {"status": "analyzed"},
            "deepfake_analysis": {"frames": [{"verdict": "non-synthetic"}]},
            "stt_enrichment": {"text": "hello", "utterances": []},
            "findings": [],
            "overall_conclusion": "inconclusive",
            "limitations": [],
        }

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            write_report(report, output_dir)

            self.assertTrue((output_dir / "modulate_deepfake_batch.json").exists())
            self.assertTrue((output_dir / "modulate_stt_enrichment.json").exists())

    def test_help_includes_usage_examples(self):
        help_text = build_parser().format_help()

        self.assertIn("Examples:", help_text)
        self.assertIn("--all-local-audio --modulate", help_text)
        self.assertIn("--show-raw-modulate", help_text)

    def test_run_modulate_stt_enrichment_uses_documented_bool_params(self):
        class FakeResponse:
            status_code = 200
            text = '{"text": "ok", "utterances": []}'

            def json(self):
                return {"text": "ok", "utterances": []}

        captured = {}

        def fake_post(url, headers, data, files, timeout):
            captured["data"] = data
            return FakeResponse()

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.mp3"
            path.write_bytes(b"audio")

            with patch("requests.post", side_effect=fake_post):
                result = run_modulate_stt_enrichment(path, "secret")

        self.assertEqual(result["text"], "ok")
        self.assertIs(captured["data"]["speaker_diarization"], True)
        self.assertIs(captured["data"]["emotion_signal"], True)
        self.assertIs(captured["data"]["accent_signal"], True)
        self.assertIs(captured["data"]["deepfake_signal"], True)
        self.assertIs(captured["data"]["pii_phi_tagging"], False)

    def test_raise_for_modulate_error_includes_response_body(self):
        class FakeResponse:
            status_code = 400
            text = '{"detail":"bad parameter"}'

        with self.assertRaises(ModulateAPIError) as context:
            raise_for_modulate_error(FakeResponse(), "stt-enrichment")

        self.assertIn("400", str(context.exception))
        self.assertIn("bad parameter", str(context.exception))


if __name__ == "__main__":
    unittest.main()
