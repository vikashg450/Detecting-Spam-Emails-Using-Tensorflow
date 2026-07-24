import sys
import os
# Add parent directory of 'src' to python path to resolve absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import tensorflow as tf
from src import config
from src.preprocess import clean_text

def predict_spam(model, raw_text):
    """
    Cleans the input text and predicts whether it is Spam or Ham.
    Returns: label (str), confidence (float), and raw probability of spam (float)
    """
    # Clean the input text using the same cleaning rules as training
    cleaned = clean_text(raw_text)
    
    # Predict (model takes a batch of raw strings since TextVectorization is integrated)
    prediction_prob = model.predict(tf.constant([cleaned]), verbose=0)[0][0]
    
    label = "Spam" if prediction_prob > 0.5 else "Ham"
    confidence = prediction_prob if label == "Spam" else (1.0 - prediction_prob)
    
    return label, confidence, prediction_prob

def main():
    parser = argparse.ArgumentParser(description="Predict if an email is spam or ham using a trained TensorFlow model")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--text', type=str, help="Text of the email to classify")
    group.add_argument('--file', type=str, help="Path to a text file containing the email")
    group.add_argument('--interactive', action='store_true', help="Run an interactive command-line loop")
    
    args = parser.parse_args()
    
    # Check if saved model exists
    if not os.path.exists(config.MODEL_SAVE_PATH):
        print(f"Error: Model not found at '{config.MODEL_SAVE_PATH}'.")
        print("Please train the model first by running: python src/train.py")
        return
        
    print("Loading trained TensorFlow model (this may take a few seconds)...")
    # Load model (including vectorizer layer)
    model = tf.keras.models.load_model(config.MODEL_SAVE_PATH)
    print("Model loaded successfully.\n")
    
    if args.text:
        label, conf, prob = predict_spam(model, args.text)
        print(f"--- Input ---")
        print(args.text)
        print(f"-------------")
        print(f"Result: {label}")
        print(f"Confidence: {conf * 100:.2f}%")
        print(f"Spam Probability: {prob:.4f}")
        
    elif args.file:
        if not os.path.exists(args.file):
            print(f"Error: File '{args.file}' does not exist.")
            return
        with open(args.file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        label, conf, prob = predict_spam(model, content)
        print(f"--- File: {args.file} ---")
        print(content[:300] + ("..." if len(content) > 300 else ""))
        print(f"-------------------------")
        print(f"Result: {label}")
        print(f"Confidence: {conf * 100:.2f}%")
        print(f"Spam Probability: {prob:.4f}")
        
    elif args.interactive:
        print("Interactive Mode. Type 'exit' or 'quit' to end.")
        while True:
            try:
                text = input("\nEnter email text: ")
                if text.strip().lower() in ['exit', 'quit']:
                    print("Exiting...")
                    break
                if not text.strip():
                    continue
                label, conf, prob = predict_spam(model, text)
                print(f"Result: {label} | Confidence: {conf*100:.2f}% | Spam Prob: {prob:.4f}")
            except KeyboardInterrupt:
                print("\nExiting...")
                break

if __name__ == '__main__':
    main()
