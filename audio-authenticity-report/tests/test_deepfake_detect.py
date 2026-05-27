import unittest

from deepfake_detect import summarize_frames


class SummarizeFramesTests(unittest.TestCase):
    def test_flags_high_confidence_synthetic_frames(self):
        summary = summarize_frames(
            [
                {"verdict": "non-synthetic", "confidence": 0.92},
                {"verdict": "synthetic", "confidence": 0.91},
                {"verdict": "synthetic", "confidence": 0.74},
                {"verdict": "no-content", "confidence": 1.0},
            ],
            confidence_threshold=0.8,
        )

        self.assertEqual(summary.overall_verdict, "synthetic")
        self.assertEqual(summary.synthetic_frames, 1)
        self.assertEqual(summary.speech_frames, 3)
        self.assertTrue(summary.should_flag)

    def test_ignores_no_content_when_classifying_clean_audio(self):
        summary = summarize_frames(
            [
                {"verdict": "no-content", "confidence": 1.0},
                {"verdict": "non-synthetic", "confidence": 0.87},
            ],
            confidence_threshold=0.8,
        )

        self.assertEqual(summary.overall_verdict, "non-synthetic")
        self.assertEqual(summary.speech_frames, 1)
        self.assertFalse(summary.should_flag)

    def test_returns_no_content_when_no_speech_frames_exist(self):
        summary = summarize_frames(
            [{"verdict": "no-content", "confidence": 1.0}],
            confidence_threshold=0.8,
        )

        self.assertEqual(summary.overall_verdict, "no-content")
        self.assertEqual(summary.speech_frames, 0)
        self.assertFalse(summary.should_flag)


if __name__ == "__main__":
    unittest.main()
