import unittest

from ui.state import derive_readiness_state


class TestUIState(unittest.TestCase):
    def test_missing_model_state_is_blocking(self):
        readiness = derive_readiness_state(
            model_available=False,
            active_patient_id=None,
            uploaded_scan_name=None,
            analysis_result=None,
        )

        self.assertEqual(readiness["state"], "model_missing")
        self.assertEqual(readiness["tone"], "critical")

    def test_no_patient_state(self):
        readiness = derive_readiness_state(
            model_available=True,
            active_patient_id=None,
            uploaded_scan_name=None,
            analysis_result=None,
        )

        self.assertEqual(readiness["state"], "no_patient")
        self.assertEqual(readiness["tone"], "warning")

    def test_patient_ready_without_upload(self):
        readiness = derive_readiness_state(
            model_available=True,
            active_patient_id="MS-001",
            uploaded_scan_name=None,
            analysis_result=None,
        )

        self.assertEqual(readiness["state"], "patient_ready")
        self.assertEqual(readiness["tone"], "safe")

    def test_uploaded_scan_waiting_for_analysis(self):
        readiness = derive_readiness_state(
            model_available=True,
            active_patient_id="MS-001",
            uploaded_scan_name="scan-a.png",
            analysis_result=None,
        )

        self.assertEqual(readiness["state"], "scan_ready")
        self.assertEqual(readiness["tone"], "warning")

    def test_analysis_complete_for_current_scan(self):
        readiness = derive_readiness_state(
            model_available=True,
            active_patient_id="MS-001",
            uploaded_scan_name="scan-a.png",
            analysis_result={"image_name": "scan-a.png"},
        )

        self.assertEqual(readiness["state"], "analysis_complete")
        self.assertEqual(readiness["tone"], "safe")

    def test_new_upload_replaces_previous_analysis_state(self):
        readiness = derive_readiness_state(
            model_available=True,
            active_patient_id="MS-001",
            uploaded_scan_name="scan-b.png",
            analysis_result={"image_name": "scan-a.png"},
        )

        self.assertEqual(readiness["state"], "scan_ready")
        self.assertEqual(readiness["tone"], "warning")


if __name__ == "__main__":
    unittest.main()
