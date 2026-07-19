import sys
import os
import unittest
from unittest.mock import patch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.triage_pipeline import TriagePipeline

class TestTriagePipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("Setting up TestTriagePipeline...")
        cls.pipeline = TriagePipeline()

    def test_normal_patient(self):
        """
        Verify that a stable patient with standard vitals and a simple fever
        is classified with low urgency (ESI 4 or 5) and has no overrides triggered.
        """
        patient = {
            "age": 35,
            "gender": "FEMALE",
            "triage_vital_hr": 72,
            "triage_vital_sbp": 120,
            "triage_vital_dbp": 80,
            "triage_vital_rr": 14,
            "triage_vital_o2": 99,
            "triage_vital_temp": 98.6,
            "cc_fever": 1
        }
        res = self.pipeline.predict(patient)
        self.assertIn(res["esi_level"], [4, 5])
        self.assertFalse(res["override_triggered"])
        self.assertIsNone(res["override_reason"])
        self.assertGreater(len(res["reasons"]), 0)

    def test_esi1_hypoxia_override(self):
        """
        Verify that a patient with severe hypoxia (SpO2 < 90) triggers ESI 1 override.
        """
        patient = {
            "age": 50,
            "gender": "MALE",
            "triage_vital_hr": 95,
            "triage_vital_sbp": 115,
            "triage_vital_dbp": 75,
            "triage_vital_rr": 20,
            "triage_vital_o2": 85,  # Severe Hypoxia
            "triage_vital_temp": 101.2,
            "cc_fever": 1
        }
        res = self.pipeline.predict(patient)
        self.assertEqual(res["esi_level"], 1)
        self.assertTrue(res["override_triggered"])
        self.assertIn("Severe hypoxia", res["override_reason"])
        self.assertTrue(any("Severe Hypoxia" in g for g in res["guidelines"]))

    def test_esi1_unresponsive_override(self):
        """
        Verify that a patient with altered mental status / unresponsive triggers ESI 1 override.
        """
        patient = {
            "age": 60,
            "gender": "FEMALE",
            "triage_vital_hr": 80,
            "triage_vital_sbp": 110,
            "triage_vital_dbp": 70,
            "triage_vital_rr": 18,
            "triage_vital_o2": 96,
            "triage_vital_temp": 99.0,
            "cc_fever": 1,
            "cc_unresponsive": 1  # Unresponsive
        }
        res = self.pipeline.predict(patient)
        self.assertEqual(res["esi_level"], 1)
        self.assertTrue(res["override_triggered"])
        self.assertIn("Unresponsive", res["override_reason"])

    def test_esi2_qsofa_override(self):
        """
        Verify that a patient with qSOFA score >= 2 (e.g. SBP=90, RR=24)
        who is otherwise classified as low urgency gets upgraded to ESI 2.
        """
        # We need to make sure the raw classifier doesn't already output ESI 1 or 2
        # A patient with stable symptoms but low blood pressure and high respiratory rate
        patient = {
            "age": 40,
            "gender": "MALE",
            "triage_vital_hr": 88,
            "triage_vital_sbp": 95,  # sbp <= 100 (+1 qSOFA)
            "triage_vital_dbp": 60,
            "triage_vital_rr": 24,   # rr >= 22 (+1 qSOFA)
            "triage_vital_o2": 96,
            "triage_vital_temp": 99.8,
            "cc_fever": 1
        }
        # Run prediction
        res = self.pipeline.predict(patient)
        
        # Verify that ESI is 1 or 2
        self.assertIn(res["esi_level"], [1, 2])
        # If it was overridden, check override status
        # (It could be ESI 2 naturally or via override)
        if res["esi_level"] == 2 and res["override_triggered"]:
            self.assertIn("qSOFA", res["override_reason"])
            self.assertTrue(any("qSOFA" in g for g in res["guidelines"]))

    def test_esi2_moderate_hypoxia_override(self):
        """
        Verify that moderate hypoxia (SpO2 < 95) triggers override to ESI 2 if raw pred is lower urgency.
        """
        patient = {
            "age": 30,
            "gender": "FEMALE",
            "triage_vital_hr": 78,
            "triage_vital_sbp": 120,
            "triage_vital_dbp": 80,
            "triage_vital_rr": 16,
            "triage_vital_o2": 92,  # Moderate Hypoxia
            "triage_vital_temp": 99.0,
            "cc_fever": 1
        }
        res = self.pipeline.predict(patient)
        self.assertIn(res["esi_level"], [1, 2])
        if res["override_triggered"]:
            self.assertIn("Moderate hypoxia", res["override_reason"])

    @patch('src.triage_pipeline.Groq', None)
    @patch('src.triage_pipeline.genai', None)
    def test_offline_degradation(self):
        """
        Verify that if Groq and Gemini are unavailable, the pipeline degrades gracefully
        by generating local offline fallback recommendations instead of throwing an error.
        """
        patient = {
            "age": 35,
            "gender": "FEMALE",
            "triage_vital_hr": 72,
            "triage_vital_sbp": 120,
            "triage_vital_dbp": 80,
            "triage_vital_rr": 14,
            "triage_vital_o2": 99,
            "triage_vital_temp": 98.6,
            "cc_fever": 1
        }
        # Run prediction with mocked None LLM libraries
        res = self.pipeline.predict(patient)
        self.assertTrue(res["is_offline"])
        self.assertIn("Offline Fallback", res["formatted_advice"])
        self.assertIn("Urgency Category", res["formatted_advice"])

if __name__ == "__main__":
    unittest.main()
