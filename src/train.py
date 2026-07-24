import sys
import os
# Add parent directory of 'src' to python path to resolve absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

from src import config
from src.preprocess import load_and_preprocess_data, get_vectorization_layer, prepare_datasets
from src.models import get_model

def plot_history(history, save_path):
    """
    Plots training and validation loss/accuracy curves.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='train', linewidth=2)
    if 'val_accuracy' in history.history:
        ax1.plot(history.history['val_accuracy'], label='val', linewidth=2)
    ax1.set_title('Model Accuracy', fontsize=14)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='train', linewidth=2)
    if 'val_loss' in history.history:
        ax2.plot(history.history['val_loss'], label='val', linewidth=2)
    ax2.set_title('Model Loss', fontsize=14)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved training curves to: {save_path}")

def plot_confusion_matrix(y_true, y_pred, save_path):
    """
    Generates and saves a confusion matrix heatmap.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'],
                annot_kws={"size": 14})
    plt.title('Confusion Matrix', fontsize=16)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix heatmap to: {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Train email spam classifier using TensorFlow")
    parser.add_argument('--model', type=str, default='bilstm', choices=['dense', 'bilstm', 'conv1d'],
                        help="Neural network architecture to train (default: bilstm)")
    parser.add_argument('--epochs', type=int, default=config.EPOCHS,
                        help="Number of epochs to train (default: config.EPOCHS)")
    parser.add_argument('--batch-size', type=int, default=config.BATCH_SIZE,
                        help="Batch size (default: config.BATCH_SIZE)")
    args = parser.parse_args()

    print("\n--- Step 1: Loading and Preprocessing Dataset ---")
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    print(f"Train samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    print("\n--- Step 2: Adapting TextVectorization Layer ---")
    vectorize_layer = get_vectorization_layer(X_train)
    print("TextVectorization layer adapted successfully.")
    
    print("\n--- Step 3: Preparing TensorFlow Datasets ---")
    # Set config batch size dynamically based on arg parser
    config.BATCH_SIZE = args.batch_size
    train_dataset, test_dataset = prepare_datasets(X_train, y_train, X_test, y_test)
    
    print(f"\n--- Step 4: Building model: {args.model} ---")
    model = get_model(args.model, vectorize_layer)
    model.summary()
    
    # Compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=['accuracy']
    )
    
    # Early Stopping callback to prevent overfitting
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )
    
    print("\n--- Step 5: Starting Training ---")
    history = model.fit(
        train_dataset,
        epochs=args.epochs,
        validation_data=test_dataset,
        callbacks=[early_stopping]
    )
    
    # Save the trained model (includes the vectorization layer)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    model.save(config.MODEL_SAVE_PATH)
    print(f"\nModel saved successfully to: {config.MODEL_SAVE_PATH}")
    
    print("\n--- Step 6: Evaluation ---")
    test_loss, test_acc = model.evaluate(test_dataset)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Predict probabilities and apply threshold
    y_pred_probs = model.predict(test_dataset)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))
    
    # Generate and save figures
    history_plot_path = os.path.join(config.OUTPUT_DIR, 'training_history.png')
    cm_plot_path = os.path.join(config.OUTPUT_DIR, 'confusion_matrix.png')
    
    plot_history(history, history_plot_path)
    plot_confusion_matrix(y_test, y_pred, cm_plot_path)
    print("\nTraining and evaluation pipeline complete.")

if __name__ == '__main__':
    main()
