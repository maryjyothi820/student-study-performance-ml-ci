
import json
import os
import unittest

import joblib
import pandas as pd


class TestMLPipeline(unittest.TestCase):

    def test_dataset_created(self):
        self.assertTrue(os.path.exists("student_results.csv"))

    def test_model_created(self):
        self.assertTrue(os.path.exists("student_result_model.pkl"))

    def test_metrics_created(self):
        self.assertTrue(os.path.exists("metrics.json"))

    def test_accuracy_is_valid(self):
        with open("metrics.json", "r") as file:
            metrics = json.load(file)

        self.assertGreaterEqual(metrics["accuracy"], 0.0)
        self.assertLessEqual(metrics["accuracy"], 1.0)

    def test_model_prediction(self):
        model = joblib.load("student_result_model.pkl")

        sample = pd.DataFrame([{
            "study_hours": 6,
            "attendance": 85,
            "assignments_completed": 8,
            "sleep_hours": 7,
            "social_media_hours": 2,
            "previous_score": 78
        }])

        prediction = model.predict(sample)[0]
        self.assertEqual(prediction, "Pass")

    def test_high_performance_student(self):
        model = joblib.load("student_result_model.pkl")

        sample = pd.DataFrame([{
            "study_hours": 9,
            "attendance": 95,
            "assignments_completed": 10,
            "sleep_hours": 8,
            "social_media_hours": 1,
            "previous_score": 90
        }])

        self.assertEqual(model.predict(sample)[0], "Pass")

    def test_low_performance_student(self):
        model = joblib.load("student_result_model.pkl")

        sample = pd.DataFrame([{
            "study_hours": 1,
            "attendance": 55,
            "assignments_completed": 1,
            "sleep_hours": 5,
            "social_media_hours": 7,
            "previous_score": 35
        }])

        self.assertEqual(model.predict(sample)[0], "Fail")


if __name__ == "__main__":
    unittest.main()
