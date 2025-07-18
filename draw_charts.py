import matplotlib.pyplot as plt
import os

# Data from the previous step
normalized_results_with_invalid = {
    "imdb_sentiment": {
        "base": {"negative": 12803, "positive": 11441, "invalid": 516 + 240},  # mixed + neutral 归为 invalid
        "f":    {"negative": 11775, "positive": 11826, "invalid": 1325 + 74},   # mixed + neutral 归为 invalid
        "t":    {"negative": 13253, "positive": 10543, "invalid": 244 + 960},   # mixed + neutral 归为 invalid
    },
    "mental_sentiment": {
        "base": {"depression": 27041, "normal": 4445, "invalid": 261},
        "f":    {"depression": 27934, "normal": 3575, "invalid": 238},
        "t":    {"depression": 27317, "normal": 4309, "invalid": 121},
    },
    "financial_sentiment": {
        "base": {"bearish": 2840, "bullish": 5226, "neutral": 3863, "invalid": 2},
        "f":    {"bearish": 2732, "bullish": 5779, "neutral": 3415, "invalid": 5},
        "t":    {"bearish": 2503, "bullish": 4386, "neutral": 5041, "invalid": 1},
    },
    "fiqasa_sentiment": {
        "base": {"negative": 587, "positive": 238, "neutral": 348, "invalid": 0},
        "f":    {"negative": 511, "positive": 432, "neutral": 226, "invalid": 4},
        "t":    {"negative": 550, "positive": 108, "neutral": 515, "invalid": 0},
    },
    "imdb_sklearn": {
        "base": {"negative": 5316, "positive": 4684},
        "f":    {"negative": 5157, "positive": 4843},
        "t":    {"negative": 5807, "positive": 4193},
    },
    "sst2": {
        "base": {"negative": 5579, "positive": 3857, "neutral": 561 + 2, "invalid": 3 + 1},  # mixed + neutral 归为 invalid
        "f":    {"negative": 4898, "positive": 4590, "neutral": 506 + 4, "invalid": 6 + 2},  # mixed + neutral 归为 invalid
        "t":    {"negative": 6118, "positive": 2519, "neutral": 1361 + 2, "invalid": 2 + 0},  # mixed + neutral 归为 invalid
    }
}

# Create a folder for saving the plots
if not os.path.exists("plots"):
    os.makedirs("plots")

# Plotting and saving images
for dataset, model_data in normalized_results_with_invalid.items():
    fig, axs = plt.subplots(1, 3, figsize=(12, 6))
    fig.suptitle(f"{dataset} Prediction Distribution", fontsize=14)

    for idx, model in enumerate(['base', 'f', 't']):
        data = model_data.get(model, {})
        labels = list(data.keys())
        sizes = list(data.values())
        axs[idx].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
        axs[idx].axis('equal')
        axs[idx].set_title(f"{model.upper()}")

    # Save the figure to the "plots" folder
    fig.savefig(f"plots/{dataset}_prediction_distribution.png")

plt.close('all')  # Close all the figures to release memory
