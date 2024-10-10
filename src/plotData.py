import matplotlib.pyplot as plt
import pandas as pd


class GenePlotter:
    def __init__(self, dataframe):
        """
        Initialize the GenePlotter with a pandas DataFrame.

        Parameters:
        dataframe (pd.DataFrame): DataFrame containing gene data with columns:
                                  'gene_name', 'sequence_start', 'sequence_end', 'contains_tss'
        """
        self.df = dataframe

    def plot_gene_length(self):
        """
        Creates a scatter plot of sequence_start vs gene_length.
        Points are colored based on 'contains_tss'.
        """
        self.df["gene_length"] = self.df["sequence_end"] - self.df["sequence_start"]
        tss_true = self.df[self.df["contains_tss"] == True]
        tss_false = self.df[self.df["contains_tss"] == False]

        plt.figure(figsize=(10, 6))
        plt.scatter(
            tss_true["sequence_start"],
            tss_true["gene_length"],
            color="blue",
            label="Contains TSS",
            alpha=0.6,
        )
        plt.scatter(
            tss_false["sequence_start"],
            tss_false["gene_length"],
            color="red",
            label="No TSS",
            alpha=0.6,
        )

        plt.xlabel("Sequence Start")
        plt.ylabel("Gene Length")
        plt.title("Gene Length Scatter Plot")
        plt.legend()

        plt.show()

    def plot_gene_length_boxplot(self):
        """
        Creates a box plot comparing the gene lengths based on whether they contain a TSS or not.
        """
        plt.figure(figsize=(10, 6))
        self.df.boxplot(
            column="gene_length",
            by="contains_tss",
            grid=False,
            patch_artist=True,
            boxprops=dict(facecolor="lightblue"),
        )

        plt.title("Gene Length by TSS Status")
        plt.suptitle("")  # Remove default title
        plt.xlabel("Contains TSS")
        plt.ylabel("Gene Length")

        plt.show()

    def plot_heatmap_start_end(self):
        """
        Creates a heatmap to show the density of sequence start and end positions.
        """
        plt.figure(figsize=(10, 6))
        plt.hexbin(
            self.df["sequence_start"],
            self.df["sequence_end"],
            gridsize=50,
            cmap="Blues",
        )

        plt.xlabel("Sequence Start")
        plt.ylabel("Sequence End")
        plt.title("Heatmap of Sequence Start vs End")

        plt.colorbar(label="Counts")
        plt.show()

    def plot_sequences(self):
        """
        Creates a scatter plot of sequence_start vs sequence_end.
        Points are colored based on 'contains_tss':
        - Blue for 'True' (TSS present)
        - Red for 'False' (TSS not present)
        """
        # Separate data based on 'contains_tss'
        tss_true = self.df[self.df["contains_tss"] == True]
        tss_false = self.df[self.df["contains_tss"] == False]

        # Create scatter plot
        plt.figure(figsize=(10, 6))
        plt.scatter(
            tss_true["sequence_start"],
            tss_true["sequence_end"],
            color="blue",
            label="Contains TSS",
            alpha=0.6,
        )
        plt.scatter(
            tss_false["sequence_start"],
            tss_false["sequence_end"],
            color="red",
            label="No TSS",
            alpha=0.6,
        )

        # Add labels and title
        plt.xlabel("Sequence Start")
        plt.ylabel("Sequence End")
        plt.title("Gene Sequence Scatter Plot")
        plt.legend()

        # Show plot
        plt.show()


def toss_salad(data_set_1, data_set_2):

    df = pd.concat([data_set_1, data_set_2], ignore_index=True)
    shuffled_data = df.sample(frac=1).reset_index(drop=True)

    return shuffled_data


away = pd.read_csv("./away.csv")
tss = pd.read_csv("./tss.csv")
df = toss_salad(away, tss)
gene_plotter = GenePlotter(df)
gene_plotter.plot_heatmap_start_end()
