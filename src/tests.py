import unittest
import pandas as pd
from sklearn.model_selection import train_test_split

from DataStructures import GenomeDataSet, GenomeModel


class GenomeDatasetTests(unittest.TestCase):
    def test_cut_cut(self):
        df = pd.read_csv("small.csv")
        a = GenomeDataSet(df, df, accept_less=True)
        b = a.cut("12345678", 4)
        assert b == "3456"

    def test_cut(self):
        # Load the dataset
        df = pd.read_csv("small.csv")
        a = GenomeDataSet(df, df, accept_less=True)

        # Test case 1 (Provided test case)
        b = a.cut("12345678", 4)
        assert b == "3456"

        # Test case 2: String is shorter than n
        b = a.cut("abc", 5)
        assert b == "abc"  # String is already shorter than n

        # Test case 3: String has an odd number of characters
        b = a.cut("abcdefgh", 6)
        assert b == "bcdefg"  # Cut equally from both sides, returns middle 6 characters

        # Test case 4: n is the same as the length of the string
        b = a.cut("abcdefgh", 8)
        assert b == "abcdefgh"  # The string is already the desired length

    def test_process(self):
        away = pd.read_csv("./101away.csv")
        tss = pd.read_csv(("./101.csv"))

        a = GenomeDataSet(tss, away, tss_non_tss_ratio=0.5, train_test_ratio=0.5)
        (X_train, X_test, y_train, y_test) = a.preprocess()

        print("")

    def test_file_name(self):
        away = pd.read_csv("./small-away.csv")
        tss = pd.read_csv(("./small-tss.csv"))

        a = GenomeDataSet(
            tss, away, tss_non_tss_ratio=0.5, train_test_ratio=0.5, accept_less=True
        )
        assert a.create_descriptive_filename() is not None

    def test_fit_Model(self):
        away = pd.read_csv("./101away.csv")
        tss = pd.read_csv(("./101.csv"))

        a = GenomeDataSet(tss, away, tss_non_tss_ratio=0.5, train_test_ratio=0.5)
        (X_train, X_test, y_train, y_test) = a.preprocess()
        b = GenomeModel()
        X = X_train[["sequence_start", "sequence_end", "sequence"]]
        Y = y_train
        b.train(X, Y)

    def test_one_hot(self):
        away = pd.read_csv("./101away.csv")
        tss = pd.read_csv(("./101.csv"))
        a = GenomeDataSet(tss, away, tss_non_tss_ratio=0.5, train_test_ratio=0.5)
        b = a.one_hot_encode_sequence("ACTAAAGTTAG")
        print(b)

    def test_one_hot_all_data(self):
        away = pd.read_csv("./101away.csv")
        tss = pd.read_csv(("./101.csv"))
        a = GenomeDataSet(tss, away, tss_non_tss_ratio=0.5, train_test_ratio=0.5)


if __name__ == "__main__":
    unittest.main()
