import unittest
import numpy as np
import tensorflow as tf
from src.preprocess import clean_text
from src.models import get_model

class TestSpamClassifierPipeline(unittest.TestCase):
    
    def test_clean_text(self):
        # Test case: Normal text with Subject
        raw_text_1 = "Subject: Hello! How are you doing?   "
        self.assertEqual(clean_text(raw_text_1), "Hello! How are you doing?")
        
        # Test case: Text with newlines and multiple spaces
        raw_text_2 = "Subject: Get cash now!\n\nCheck out   this website\tnow."
        self.assertEqual(clean_text(raw_text_2), "Get cash now! Check out this website now.")

    def test_models_building(self):
        # Create a mock text vectorization layer adapted on tiny dataset
        texts = ["hello world", "win free money", "congratulations lottery winner", "meeting tomorrow at ten"]
        vectorize_layer = tf.keras.layers.TextVectorization(
            max_tokens=100,
            output_mode='int',
            output_sequence_length=10
        )
        vectorize_layer.adapt(texts)
        
        # Build and check each model
        for model_name in ['dense', 'bilstm', 'conv1d']:
            with self.subTest(model_name=model_name):
                model = get_model(model_name, vectorize_layer)
                
                # Check compilation/input/output shapes
                model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                
                # Model output shape check
                test_input = tf.constant(["win free money!", "hey there"])
                pred = model.predict(test_input, verbose=0)
                self.assertEqual(pred.shape, (2, 1))
                self.assertTrue(np.all(pred >= 0.0) and np.all(pred <= 1.0))

class TestFastAPIEndpoints(unittest.TestCase):
    def test_health_and_predict_endpoints(self):
        from fastapi.testclient import TestClient
        from src.app import app
        
        with TestClient(app) as client:
            # Test /health endpoint
            response = client.get("/health")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["status"], "healthy")
            self.assertIn("model_loaded", data)
            
            # Test /predict endpoint if model loaded
            if data["model_loaded"]:
                pred_resp = client.post("/predict", json={"text": "Congratulations, you won a lottery ticket! Click here."})
                self.assertEqual(pred_resp.status_code, 200)
                pred_data = pred_resp.json()
                self.assertIn("label", pred_data)
                self.assertIn("confidence", pred_data)
                self.assertIn("spam_probability", pred_data)

if __name__ == '__main__':
    unittest.main()

