import csv
import pandas as pd
from DataStructures import GenomeDataSet, GenomeModel


class ModelRunner:
    def __init__(self, tss, no_tss, params_grid, output_file="output.csv"):
        self.tss = tss
        self.no_tss = no_tss
        self.params_grid = params_grid
        self.models = []
        self.output_file = output_file

    def run(self):
        results = []

        for tss_ratio in self.params_grid["tss_non_tss_ratio"]:
            for train_test_ratio in self.params_grid["train_test_ratio"]:
                for genome_seq_len in self.params_grid["genome_seq_len"]:
                    dataset = GenomeDataSet(
                        tss=self.tss,
                        no_tss=self.no_tss,
                        tss_non_tss_ratio=tss_ratio,
                        train_test_ratio=train_test_ratio,
                        genome_seq_len=genome_seq_len,
                    )

                    X_train, X_test, y_train, y_test = dataset.preprocess()
                    model = GenomeModel()
 p                   model.train(X_train, y_train)

                    # Save the model
                    filename = dataset.create_descriptive_filename()
                    model.save_model(f"{filename}.pkl")
                    self.models.append(filename)

                    # Evaluate the model
                    mae = model.get_mae(X_test, y_test)
                    results.append((filename, mae))

        # Output results to CSV
        self.save_results_to_csv(results)

        # Output list of models
        self.save_model_list()

    def save_results_to_csv(self, results):
        with open(self.output_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Model Filename", "MAE"])
            writer.writerows(results)
        print(f"Results saved to {self.output_file}")

    def save_model_list(self, list_file="model_list.txt"):
        with open(list_file, mode="w") as file:
            for model in self.models:
                file.write(f"{model}\n")
        print(f"List of models saved to {list_file}")


# Example usage:
if __name__ == "__main__":
    tss_data = pd.read_csv(
        "/home/camer/Documents/bcitfall2024/ai/GenomicCaseStudy/cameronscode/tss.csv"
    )
    no_tss_data = pd.read_csv(
        "/home/camer/Documents/bcitfall2024/ai/GenomicCaseStudy/cameronscode/away.csv"
    )

    params_grid = {
        "tss_non_tss_ratio": [0.1, 0.2],  # Add more ratios as needed
        "train_test_ratio": [0.75, 0.80],  # Add more ratios as needed
        "genome_seq_len": [5, 10],  # Add more lengths as needed
    }

    runner = ModelRunner(tss_data, no_tss_data, params_grid)
    runner.run()
