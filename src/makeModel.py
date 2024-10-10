# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pylab as plt
import sys


make_graph = False
csv_files = [
    "./tss.csv",
    "./away.csv",
    "./test_sh_script_tss_sequences.csv",
    "./test_sh_script_away_sequences.csv",
]
away_df = None
tss_df = None
if len(sys.argv) < 2:
    away_df = pd.read_csv(csv_files[0])
    tss_df = pd.read_csv(csv_files[1])
else:
    away_df = pd.read_csv(sys.argv[1])
    tss_df = pd.read_csv(sys.argv[2])

df = pd.concat([away_df, tss_df], ignore_index=True)
shuffled_data = df.sample(frac=1).reset_index(drop=True)


# Preprocess data
# Convert categorical `gene_name` to numerical
label_encoder = LabelEncoder()
shuffled_data["gene_name_encoded"] = label_encoder.fit_transform(
    shuffled_data["gene_name"]
)

# Select features and target variable
# Assume we generate numerical features from `sequence` somehow, here taken as `sequence_features`
sequence_features = pd.get_dummies(
    shuffled_data["sequence"]
)  # Example of one-hot encoding

# Extract the features and target
features = pd.concat(
    [
        shuffled_data[["sequence_start", "sequence_end", "gene_name_encoded"]],
        sequence_features,
    ],
    axis=1,
)
target = shuffled_data["contains_tss"]

# Split the shuffled_data
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42
)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)
print(model.coef_)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

if make_graph:
    # Scatter plot
    plt.figure(figsize=(10, 6))

    # Plot data points
    # Red for does not contain tss (0), blue for contains tss (1)
    colors = {0: "red", 1: "blue"}
    edge_colors = shuffled_data["contains_tss"].apply(lambda tss: colors[tss])

    plt.scatter(
        shuffled_data["sequence_start"],
        shuffled_data["sequence_end"],
        c=edge_colors,
        alpha=0.5,
    )

    # Add labels and title
    plt.xlabel("Sequence Start")
    plt.ylabel("Sequence End")
    plt.title("Scatter plot of Sequence Start vs. End")
    plt.legend(
        handles=[
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="Does not contain TSS",
                markerfacecolor="red",
                markersize=10,
            ),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="Contains TSS",
                markerfacecolor="blue",
                markersize=10,
            ),
        ]
    )
    plt.grid(True)
    plt.show()
