from abc import ABC, abstractmethod
import pandas as pd
import numpy as np


class DataSet(ABC):
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    @abstractmethod
    def preprocess(self):
        pass


class Model(ABC):
    def __init__(self, coefs) -> None:
        self.coefs = coefs

    @abstractmethod
    def train(self, data: pd.DataFrame):
        pass

    @abstractmethod
    def is_trained(self) -> bool:
        pass


class TrainedModel(ABC):
    def __init__(self, coefs: np.ndarray) -> None:
        self.coefs: np.ndarray = coefs

    @abstractmethod
    def predict(self):
        pass
