import tensorflow as tf
from src import config

def build_dense_model(vectorize_layer):
    """
    Builds a simple Feedforward network with global average pooling.
    Very fast to train and serves as a good baseline.
    """
    model = tf.keras.Sequential([
        vectorize_layer,
        tf.keras.layers.Embedding(
            input_dim=config.MAX_VOCAB_SIZE,
            output_dim=config.EMBEDDING_DIM,
            mask_zero=True
        ),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ], name="Dense_Baseline_Model")
    return model

def build_bilstm_model(vectorize_layer):
    """
    Builds a Bidirectional LSTM recurrent neural network.
    Excellent for capturing long-term sequential dependencies in text.
    """
    model = tf.keras.Sequential([
        vectorize_layer,
        tf.keras.layers.Embedding(
            input_dim=config.MAX_VOCAB_SIZE,
            output_dim=config.EMBEDDING_DIM,
            mask_zero=True
        ),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=False)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ], name="BiLSTM_Model")
    return model

def build_conv1d_model(vectorize_layer):
    """
    Builds a 1D Convolutional Neural Network.
    Efficiently extracts local patterns and n-grams from sentences.
    """
    model = tf.keras.Sequential([
        vectorize_layer,
        tf.keras.layers.Embedding(
            input_dim=config.MAX_VOCAB_SIZE,
            output_dim=config.EMBEDDING_DIM,
            mask_zero=False  # Conv1D layers generally do not support masking directly
        ),
        tf.keras.layers.Conv1D(128, 5, activation='relu'),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ], name="Conv1D_Model")
    return model

def get_model(model_name, vectorize_layer):
    """
    Factory function to retrieve model by name.
    """
    name = model_name.lower()
    if name == 'dense':
        return build_dense_model(vectorize_layer)
    elif name == 'bilstm':
        return build_bilstm_model(vectorize_layer)
    elif name == 'conv1d':
        return build_conv1d_model(vectorize_layer)
    else:
        raise ValueError(f"Unknown model name: {model_name}. Choose from 'dense', 'bilstm', 'conv1d'.")
