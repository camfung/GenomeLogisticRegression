from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

from utils import combine_dataframes


class GenomeDataSet:
    def __init__(
        self,
        tss: pd.DataFrame,
        no_tss: pd.DataFrame,
        tss_non_tss_ratio: float = 0.01,
        train_test_ratio: float = 0.75,
        genome_seq_len: int = 5,
        accept_less=False,
    ) -> None:
        self.tss_non_tss_ratio = tss_non_tss_ratio
        self.train_test_ratio = train_test_ratio
        self.genome_seq_len = genome_seq_len
        self.total_num_data_points: int = int(tss_non_tss_ratio * len(tss))
        self.tss: pd.DataFrame = tss
        self.no_tss: pd.DataFrame = no_tss
        if len(tss) < 100 and not accept_less:
            raise Exception("Data must have at least 100 datapoints")
        if len(no_tss) < 100 and not accept_less:
            raise Exception("Data must have at least 100 datapoints")

    def preprocess(self):
        self.cut_genome_seq("sequence")
        self.encode_seq("sequence")
        self.ratio_tss_to_no_tss()
        return self.split_test_train()

    def split_test_train(self, labels_col="contains_tss"):
        if self.df is None:
            raise Exception("must call ratio_tss_to_no_tss before split_test_train")
        labels = self.df[labels_col]
        features = self.df.drop(columns=[labels_col])

        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, train_size=self.train_test_ratio, random_state=42
        )
        return (X_train, X_test, y_train, y_test)

    def cut_genome_seq(self, col_name: str):

        for i, item in enumerate(self.no_tss[col_name]):
            new_seq = self.cut(item, self.genome_seq_len)
            self.no_tss[i, col_name] = new_seq

        for i, item in enumerate(self.tss[col_name]):
            new_seq = self.cut(item, self.genome_seq_len)
            self.tss[i, col_name] = new_seq

    def encode_seq(self, col_name: str):
        for i, item in enumerate(self.no_tss[col_name]):
            new_seq = self.one_hot_encode_sequence(item)
            self.no_tss[i, col_name] = new_seq

        for i, item in enumerate(self.tss[col_name]):
            new_seq = self.one_hot_encode_sequence(item)
            self.tss[i, col_name] = new_seq

    def one_hot_encode_sequence(self, sequence):
        nucleotide_to_index = {"A": 0, "C": 1, "G": 2, "T": 3}
        one_hot_encoded = np.zeros((len(sequence), 4), dtype=int)

        for i, nucleotide in enumerate(sequence):
            if nucleotide in nucleotide_to_index:
                index = nucleotide_to_index[nucleotide]
                one_hot_encoded[i, index] = 1

        return one_hot_encoded.flatten()

    def ratio_tss_to_no_tss(self):
        combined_data = combine_dataframes(
            self.tss, self.no_tss, self.tss_non_tss_ratio, self.total_num_data_points
        )
        self.df = combined_data

    def cut(self, s, n):
        if n >= len(s):
            return s

        to_remove = (len(s) - n) // 2
        return s[to_remove : len(s) - to_remove]

    def create_descriptive_filename(self):
        """
        Create a descriptive file name based on the given parameters.

        Args:
            tss_non_tss_ratio (float): Ratio of TSS to non-TSS.
            train_test_ratio (float): Ratio of training to testing dataset.
            genome_seq_len (int): Length of the genome sequence.

        Returns:
            str: A descriptive file name.
        """
        filename = (
            f"tss_ratio_{self.tss_non_tss_ratio:.2f}_"
            f"train_test_ratio_{self.train_test_ratio:.2f}_"
            f"genome_len_{self.genome_seq_len}"
        ).replace(".", "-")

        return filename


class GenomeModel:
    def __init__(self) -> None:
        self.model = None
        self.persistence = ModelPersistence()

    def train(self, features, labels):
        # Train logistic regression model
        model = LogisticRegression()
        model.fit(features, labels)
        self.model = model

    def save_model(self, filepath):
        self.persistence.save_model(self.model, filepath)

    def load_model(self, filepath):
        self.model = self.persistence.load_model(filepath)

    def predict(self, X):
        if self.model is not None:
            return self.model.predict(X)
        else:
            print("No model loaded!")
            return None

    def get_mae(self, x_test, y_test):
        predictions = self.predict(x_test)
        return np.mean(np.abs(y_test - predictions))


class ModelPersistence:
    def save_model(self, model, filepath):
        if model is not None:
            with open(filepath, "wb") as f:
                pickle.dump(model, f)
            print(f"Model saved to {filepath}.")
        else:
            print("No model to save!")

    def load_model(self, filepath):
        with open(filepath, "rb") as f:
            model = pickle.load(f)
        print(f"Model loaded from {filepath}.")
        return model
