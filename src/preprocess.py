import re
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from src import config

def clean_text(text):
    """
    Cleans raw email text:
    - Removes 'Subject:' prefix if present
    - Replaces newlines, carriage returns, and tabs with a single space
    - Shrinks multiple consecutive spaces into one
    """
    if isinstance(text, str):
        # Remove "Subject: " prefix (case-insensitive)
        text = re.sub(r'^Subject:\s*', '', text, flags=re.IGNORECASE)
        # Replace newlines/tabs with space
        text = re.sub(r'[\r\n\t]+', ' ', text)
        # Remove consecutive spaces
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    return ""

def load_and_preprocess_data():
    """
    Loads raw CSV dataset, cleans email text, and splits into train/test sets.
    """
    df = pd.read_csv(config.DATA_PATH)
    
    # Apply text cleaning
    df['clean_text'] = df['text'].apply(clean_text)
    
    X = df['clean_text'].values
    y = df['label_num'].values
    
    # Perform stratified split to maintain class ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config.TEST_SIZE, 
        random_state=config.RANDOM_STATE, 
        stratify=y
    )
    
    return X_train, X_test, y_train, y_test

def get_vectorization_layer(train_texts):
    """
    Creates and adapts a TextVectorization layer on training texts.
    """
    vectorize_layer = tf.keras.layers.TextVectorization(
        max_tokens=config.MAX_VOCAB_SIZE,
        output_mode='int',
        output_sequence_length=config.MAX_SEQUENCE_LENGTH
    )
    # Adapt to adapt the vocabulary index
    vectorize_layer.adapt(train_texts)
    return vectorize_layer

def prepare_datasets(X_train, y_train, X_test, y_test):
    """
    Converts numpy arrays into prefetching tf.data.Dataset objects.
    """
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    
    # Shuffle, batch, and prefetch for performance
    train_dataset = train_dataset.shuffle(buffer_size=len(X_train)).batch(config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    return train_dataset, test_dataset
