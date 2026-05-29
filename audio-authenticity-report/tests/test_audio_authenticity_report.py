import asyncio
import json
import sys
import tempfile
import types
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
    reality_defender_findings,
    resolve_reality_defender_api_key,
    convert_to_wav,
    run_modulate_stt_enrichment,
    print_progress_summary,
    print_reality_defender_outputs,
    print_reality_defender_table,
    poll_completion_metadata,
    poll_reality_defender_media_detail,
    render_markdown,
    resolve_output_dir,
    select_modulate_stt_input,
    select_reality_defender_input,
    run_reality_defender_detection,
    resolve_api_key,
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

    def test_build_report_uses_original_for_supported_stt_enrichment(self):
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

            self.assertEqual(calls, [source.resolve()])
            self.assertEqual(report["stt_enrichment"]["text"], "ok")

    def test_build_report_uses_converted_wav_for_amr_stt_enrichment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "sample.amr"
            converted = root / "source.wav"
            source.write_bytes(b"amr")
            converted.write_bytes(b"wav")
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

    def test_select_modulate_stt_input_falls_back_only_for_blacklisted_formats(self):
        converted = {"status": "converted", "path": "/tmp/source.wav"}

        self.assertEqual(
            select_modulate_stt_input(Path("voice.mp3"), converted),
            Path("voice.mp3"),
        )
        self.assertEqual(
            select_modulate_stt_input(Path("voice.amr"), converted),
            Path("/tmp/source.wav"),
        )

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

    def test_resolve_output_dir_groups_by_input_then_report_stem(self):
        output_dir = resolve_output_dir(
            base_output_dir=Path("reports/full-modulate"),
            input_file=Path("voice.mp3"),
            input_count=1,
            all_local_audio=False,
        )

        self.assertEqual(output_dir, Path("reports/voice/full-modulate"))

    def test_resolve_output_dir_uses_report_stem_for_each_batch_input(self):
        output_dir = resolve_output_dir(
            base_output_dir=Path("reports/full-modulate"),
            input_file=Path("dialog.wav"),
            input_count=2,
            all_local_audio=False,
        )

        self.assertEqual(output_dir, Path("reports/dialog/full-modulate"))

    def test_resolve_output_dir_default_reports_to_report_stem(self):
        output_dir = resolve_output_dir(
            base_output_dir=Path("reports"),
            input_file=Path("voice.mp3"),
            input_count=1,
            all_local_audio=False,
        )

        self.assertEqual(output_dir, Path("reports/voice/report"))

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

    def test_resolve_api_keys_reads_named_token_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            token_file = Path(tmp) / "token"
            token_file.write_text(
                "modulate:mod-secret\nrealitydefender:rd-secret\n",
                encoding="utf-8",
            )

            with patch.dict("os.environ", {}, clear=True):
                self.assertEqual(resolve_api_key(token_file), "mod-secret")
                self.assertEqual(
                    resolve_reality_defender_api_key(token_file),
                    "rd-secret",
                )

    def test_reality_defender_findings_flags_fake_ensemble_result(self):
        result = {
            "request_id": "req-1",
            "status": "FAKE",
            "score": 0.91,
            "models": [{"name": "audio", "status": "FAKE", "score": 0.88}],
            "raw": {
                "resultsSummary": {
                    "status": "FAKE",
                    "metadata": {"finalScore": 91},
                }
            },
        }

        findings = reality_defender_findings(result)

        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].kind, "deepfake")
        self.assertEqual(findings[0].severity, "strong")
        self.assertIn("Reality Defender", findings[0].message)
        self.assertEqual(findings[0].evidence["provider"], "reality_defender")
        self.assertEqual(findings[0].evidence["status"], "FAKE")
        self.assertEqual(findings[0].evidence["score"], 0.91)
        self.assertEqual(findings[0].evidence["model_votes"], "1 fake")
        self.assertNotIn("models", findings[0].evidence)

    def test_reality_defender_findings_records_not_applicable_reason(self):
        result = {
            "status": "NOT_APPLICABLE",
            "score": None,
            "raw": {
                "resultsSummary": {
                    "status": "NOT_APPLICABLE",
                    "metadata": {
                        "reasons": [
                            {
                                "code": "duration",
                                "message": "audio too short (<1.5s)",
                            }
                        ]
                    },
                }
            },
        }

        findings = reality_defender_findings(result)

        self.assertEqual(findings[0].kind, "deepfake")
        self.assertEqual(findings[0].severity, "info")
        self.assertIn("NOT_APPLICABLE", findings[0].message)
        self.assertEqual(findings[0].evidence["reasons"][0]["code"], "duration")

    def test_build_report_includes_reality_defender_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "voice.mp3"
            path.write_bytes(b"audio")

            with patch(
                "audio_authenticity_report.run_reality_defender_detection",
                return_value={"status": "SUSPICIOUS", "score": 0.67},
            ):
                report = build_report(
                    path,
                    include_deepfake=False,
                    include_reality_defender=True,
                    reality_defender_api_key="rd-secret",
                )

        self.assertEqual(report["reality_defender_analysis"]["status"], "SUSPICIOUS")
        self.assertEqual(report["reality_defender_analysis"]["score"], 0.67)
        self.assertEqual(
            report["reality_defender_analysis"]["input"]["source"],
            "original",
        )
        finding = next(
            item for item in report["findings"] if item["kind"] == "deepfake"
        )
        self.assertEqual(finding["severity"], "warning")

    def test_build_report_passes_reality_defender_polling_options(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "voice.mp3"
            path.write_bytes(b"audio")
            calls = []

            def fake_reality_defender(
                path,
                api_key,
                *,
                max_attempts,
                polling_interval_ms,
            ):
                calls.append((path, api_key, max_attempts, polling_interval_ms))
                return {"status": "AUTHENTIC", "score": 0.1}

            with patch(
                "audio_authenticity_report.run_reality_defender_detection",
                side_effect=fake_reality_defender,
            ):
                build_report(
                    path,
                    include_deepfake=False,
                    include_reality_defender=True,
                    reality_defender_api_key="rd-secret",
                    reality_defender_poll_attempts=120,
                    reality_defender_poll_interval_ms=5000,
                )

        self.assertEqual(calls, [(path.resolve(), "rd-secret", 120, 5000)])

    def test_reality_defender_detection_uploads_original_file_without_sdk_upload(self):
        calls = []

        class FakeResponse:
            status = 200

            async def text(self):
                return ""

        class FakePut:
            async def __aenter__(self):
                return FakeResponse()

            async def __aexit__(self, exc_type, exc, traceback):
                return None

        class FakeSession:
            def put(self, signed_url, data, headers):
                calls.append(("put", signed_url, data, headers))
                return FakePut()

        class FakeClient:
            def __init__(self):
                self.result_calls = 0

            async def post(self, path, data):
                calls.append(("post", path, data))
                return {
                    "requestId": "req-1",
                    "mediaId": "media-1",
                    "response": {"signedUrl": "https://upload.example/file"},
                }

            async def get(self, path):
                self.result_calls += 1
                calls.append(("get", path))
                if self.result_calls == 1:
                    return {
                        "requestId": "req-1",
                        "resultsSummary": {
                            "status": "ANALYZING",
                            "metadata": {},
                        },
                        "models": [],
                    }
                return {
                    "requestId": "req-1",
                    "resultsSummary": {
                        "status": "AUTHENTIC",
                        "metadata": {"finalScore": 10},
                    },
                    "models": [],
                }

            async def ensure_session(self):
                return FakeSession()

        class FakeRealityDefender:
            def __init__(self, api_key):
                calls.append(("init", api_key))
                self.client = FakeClient()

            def detect_file(self, file_path):
                raise AssertionError("detect_file must not run inside asyncio.run")

            async def upload(self, file_path):
                raise AssertionError("upload validates file extensions in the SDK")

            async def cleanup(self):
                calls.append(("cleanup", None))

        fake_module = types.SimpleNamespace(RealityDefender=FakeRealityDefender)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "voice.amr"
            path.write_bytes(b"audio")
            with patch.dict(sys.modules, {"realitydefender": fake_module}):
                result = run_reality_defender_detection(path, "rd-secret")

        self.assertEqual(result["status"], "AUTHENTIC")
        self.assertIn(
            ("post", "/api/files/aws-presigned", {"fileName": "voice.amr"}),
            calls,
        )
        self.assertEqual(
            calls[2],
            (
                "put",
                "https://upload.example/file",
                b"audio",
                {"Content-Type": "audio/amr"},
            ),
        )
        self.assertEqual(calls.count(("get", "/api/media/users/req-1")), 2)
        self.assertIn(("cleanup", None), calls)

    def test_reality_defender_polling_waits_for_audio_models_after_overall_status(self):
        calls = []

        class FakeClient:
            def __init__(self):
                self.result_calls = 0

            async def get(self, path):
                self.result_calls += 1
                calls.append(path)
                if self.result_calls == 1:
                    return {
                        "requestId": "req-1",
                        "resultsSummary": {"status": "AUTHENTIC"},
                        "models": [
                            {"name": "rd-slim-aud", "status": "AUTHENTIC"},
                            {"name": "rd-alethia-aud", "status": "ANALYZING"},
                        ],
                    }
                return {
                    "requestId": "req-1",
                    "resultsSummary": {"status": "AUTHENTIC"},
                    "models": [
                        {"name": "rd-slim-aud", "status": "AUTHENTIC"},
                        {"name": "rd-alethia-aud", "status": "AUTHENTIC"},
                    ],
                }

        result = asyncio.run(
            poll_reality_defender_media_detail(
                FakeClient(),
                "req-1",
                polling_interval_ms=0,
            )
        )

        self.assertEqual(len(calls), 2)
        self.assertEqual(result["models"][1]["status"], "AUTHENTIC")
        self.assertEqual(
            result["polling"],
            {
                "attempts": 2,
                "complete": True,
                "pending_audio_models": [],
            },
        )

    def test_reality_defender_polling_marks_partial_when_audio_models_remain_pending(self):
        class FakeClient:
            async def get(self, path):
                return {
                    "requestId": "req-1",
                    "resultsSummary": {"status": "AUTHENTIC"},
                    "models": [{"name": "rd-alethia-aud", "status": "ANALYZING"}],
                }

        result = asyncio.run(
            poll_reality_defender_media_detail(
                FakeClient(),
                "req-1",
                max_attempts=2,
                polling_interval_ms=0,
            )
        )

        self.assertEqual(
            result["polling"],
            {
                "attempts": 2,
                "complete": False,
                "pending_audio_models": ["rd-alethia-aud"],
            },
        )

    def test_build_report_sends_original_amr_to_reality_defender(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "voice.amr"
            converted = root / "source.wav"
            source.write_bytes(b"audio")
            converted.write_bytes(b"wav")
            calls = []

            def fake_reality_defender(path, api_key, **kwargs):
                calls.append(path)
                return {"status": "AUTHENTIC", "score": 0.1}

            with patch(
                "audio_authenticity_report.run_reality_defender_detection",
                side_effect=fake_reality_defender,
            ):
                report = build_report(
                    source,
                    include_deepfake=False,
                    include_reality_defender=True,
                    reality_defender_api_key="rd-secret",
                    converted_wav={
                        "status": "converted",
                        "path": str(converted),
                    },
                )

        self.assertEqual(calls, [source.resolve()])
        self.assertEqual(report["reality_defender_analysis"]["status"], "AUTHENTIC")

    def test_reality_defender_falls_back_to_converted_when_original_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "voice.xyz"
            converted = root / "source.wav"
            source.write_bytes(b"unsupported")
            converted.write_bytes(b"wav")
            calls = []

            def fake_reality_defender(path, api_key, **kwargs):
                calls.append(path)
                if path == source.resolve():
                    raise RuntimeError("Unsupported file type: .xyz")
                return {"status": "AUTHENTIC", "score": 0.1}

            with patch(
                "audio_authenticity_report.run_reality_defender_detection",
                side_effect=fake_reality_defender,
            ):
                report = build_report(
                    source,
                    include_deepfake=False,
                    include_reality_defender=True,
                    reality_defender_api_key="rd-secret",
                    converted_wav={
                        "status": "converted",
                        "path": str(converted),
                    },
                )

        self.assertEqual(calls, [source.resolve(), converted])
        self.assertEqual(report["reality_defender_analysis"]["status"], "AUTHENTIC")
        self.assertEqual(
            report["reality_defender_analysis"]["input"]["fallback_reason"],
            "Unsupported file type: .xyz",
        )

    def test_select_reality_defender_input_prefers_original(self):
        converted = {"status": "converted", "path": "/tmp/source.wav"}

        self.assertEqual(
            select_reality_defender_input(Path("voice.amr"), converted),
            Path("voice.amr"),
        )

    def test_write_report_saves_raw_reality_defender_outputs(self):
        report = {
            "input": {
                "filename": "voice.mp3",
                "size_bytes": 10,
                "sha256": "abcdef123456",
            },
            "audio_analysis": {"status": "analyzed"},
            "deepfake_analysis": None,
            "stt_enrichment": None,
            "reality_defender_analysis": {"status": "AUTHENTIC", "score": 0.04},
            "reality_defender_error": {
                "service": "reality-defender",
                "message": "temporary failure",
            },
            "findings": [],
            "overall_conclusion": "inconclusive",
            "limitations": [],
        }

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            write_report(report, output_dir, write_wav=False)

            self.assertTrue((output_dir / "reality_defender_detection.json").exists())
            self.assertTrue((output_dir / "reality_defender_error.json").exists())

    def test_render_markdown_includes_reality_defender_section(self):
        report = {
            "input": {
                "filename": "voice.mp3",
                "size_bytes": 10,
                "sha256": "abcdef123456",
            },
            "audio_analysis": {"status": "analyzed"},
            "findings": [],
            "overall_conclusion": "inconclusive",
            "reality_defender_analysis": {"status": "AUTHENTIC", "score": 0.04},
            "limitations": [],
        }

        text = render_markdown(report)

        self.assertIn("## Reality Defender Analysis", text)
        self.assertIn('"status": "AUTHENTIC"', text)

    def test_render_markdown_summarizes_reality_defender_model_votes(self):
        report = {
            "input": {
                "filename": "voice.wav",
                "size_bytes": 10,
                "sha256": "abcdef123456",
            },
            "audio_analysis": {"status": "analyzed"},
            "findings": [],
            "overall_conclusion": "possible_synthetic_voice_detected",
            "reality_defender_analysis": {
                "request_id": "req-1",
                "status": "MANIPULATED",
                "score": 0.86,
                "models": [
                    {
                        "name": "rd-slim-aud",
                        "status": "MANIPULATED",
                        "predictionNumber": 0.983,
                    },
                    {
                        "name": "rd-aud-ensemble",
                        "status": "MANIPULATED",
                        "normalizedPredictionNumber": 86,
                    },
                    {
                        "name": "rd-alethia-aud",
                        "status": "ANALYZING",
                        "score": None,
                    },
                ],
            },
            "limitations": [],
        }

        text = render_markdown(report)

        self.assertIn("- Status: `MANIPULATED`", text)
        self.assertIn("- Score: `86.00%`", text)
        self.assertIn("- Model votes: `2 manipulated`, `1 analyzing`", text)
        self.assertIn("| Model | Status | Score |", text)
        self.assertIn("| rd-slim-aud | MANIPULATED | 98.30% |", text)
        self.assertIn(
            "Reality Defender does not provide a time-localized audio cause",
            text,
        )

    def test_print_reality_defender_table_summarizes_models(self):
        report = {
            "input": {"filename": "voice.wav"},
            "reality_defender_analysis": {
                "request_id": "req-1",
                "status": "MANIPULATED",
                "score": 0.86,
                "models": [
                    {
                        "name": "rd-slim-aud",
                        "status": "MANIPULATED",
                        "score": 0.983,
                    },
                    {
                        "name": "rd-alethia-aud",
                        "status": "ANALYZING",
                        "score": None,
                    },
                ],
            },
        }

        with tempfile.TemporaryFile("w+") as output:
            print_reality_defender_table(report, output=output)
            output.seek(0)
            text = output.read()

        self.assertIn("Reality Defender:", text)
        self.assertIn("Status:   MANIPULATED", text)
        self.assertIn("Score:    86.00%", text)
        self.assertIn("rd-slim-aud", text)
        self.assertIn("98.30%", text)

    def test_print_reality_defender_outputs_includes_raw_json(self):
        report = {
            "reality_defender_analysis": {"status": "AUTHENTIC", "score": 0.04},
            "reality_defender_error": {"message": "failed"},
        }

        with tempfile.TemporaryFile("w+") as output:
            print_reality_defender_outputs(report, output=output)
            output.seek(0)
            text = output.read()

        self.assertIn("Raw Reality Defender response", text)
        self.assertIn('"status": "AUTHENTIC"', text)
        self.assertIn("Raw Reality Defender error", text)

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
        self.assertIn("--all-checks", help_text)
        self.assertIn("--reality-defender-wait-seconds", help_text)

    def test_all_checks_enables_modulate_and_reality_defender(self):
        parser = build_parser()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "voice.mp3"
            path.write_bytes(b"audio")
            args = parser.parse_args([str(path), "--all-checks"])

        self.assertTrue(args.all_checks)

    def test_parser_accepts_reality_defender_wait_seconds(self):
        parser = build_parser()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "voice.mp3"
            path.write_bytes(b"audio")
            args = parser.parse_args(
                [
                    str(path),
                    "--reality-defender",
                    "--reality-defender-wait-seconds",
                    "300",
                ]
            )

        self.assertEqual(args.reality_defender_wait_seconds, 300)

    def test_parser_defaults_reality_defender_wait_seconds_to_120(self):
        parser = build_parser()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "voice.mp3"
            path.write_bytes(b"audio")
            args = parser.parse_args([str(path), "--reality-defender"])

        self.assertEqual(args.reality_defender_wait_seconds, 120)

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
