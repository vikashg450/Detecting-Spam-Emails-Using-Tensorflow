import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'spam_ham_dataset.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, 'spam_classifier_model.keras')

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Hyperparameters
MAX_VOCAB_SIZE = 10000
MAX_SEQUENCE_LENGTH = 150
TEST_SIZE = 0.2
RANDOM_STATE = 42

BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 10
EMBEDDING_DIM = 64
