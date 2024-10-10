import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd


def one_hot_encode_sequence(sequence):
    # Define a mapping of nucleotides to indices
    nucleotide_to_index = {"A": 0, "C": 1, "G": 2, "T": 3}
    # Initialize a zero matrix of size (sequence length, 4)
    one_hot_encoded = np.zeros((len(sequence), 4), dtype=int)

    # Fill the matrix with 1s according to the nucleotide positions
    for i, nucleotide in enumerate(sequence):
        if nucleotide in nucleotide_to_index:
            index = nucleotide_to_index[nucleotide]
            one_hot_encoded[i, index] = 1

    # Flatten the matrix into a single array
    return one_hot_encoded.flatten()


df = pd.read_csv("./3.csv")

# Sample dataset: sequences and their labels
X = np.array([one_hot_encode_sequence(seq) for seq in df["sequence"]])
y = np.array([0, 1, 0])  # Binary labels for 'contains_tss': FALSE=0, TRUE=1

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize and fit the logistic regression model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = log_reg.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
