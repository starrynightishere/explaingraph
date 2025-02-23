# ExplainGraph

ExplainGraph is a simple interpretable model that predicts outcomes based on feature importance. It provides explanations for predictions and visualizes feature influence using graphs.

## Features
- Computes feature influence dynamically using a gradient-inspired approach.
- Provides human-readable explanations for predictions.
- Visualizes feature importance using directed graphs.
- Includes a synthetic data generator for easy experimentation.

## Installation

Clone the repository and install dependencies:

```sh
git clone https://github.com/starrynightishere/explaingraph.git
cd explaingraph
pip install -r requirements.txt
```

## Usage

### 1. Train and Explain Model

Run the `train.py` script to train the model and generate explanations:

```sh
python train.py
```

### 2. Understanding the Output

The model predicts outcomes and provides explanations in the following format:

```sh
Prediction: 1 with confidence: 0.87
Explanation:
Decision heavily influenced by Feature2. Breakdown: Feature1 (0.12) + Feature2 (0.56) + Feature3 (-0.23) + Feature4 (0.08)
```

Additionally, the script generates a directed graph showing feature influence on predictions.

## File Structure
```
explaingraph/
│── explaingraph.py       # Core ExplainGraph model
│── data_generator.py     # Generates synthetic data
│── train.py              # Trains and evaluates the model
│── requirements.txt      # Dependencies
│── README.md             # Project documentation
```

## Code Overview

### `explaingraph.py`
- Defines the `ExplainGraph` class.
- Computes feature importance using a perturbation-based approach.
- Predicts outcomes with confidence scores.
- Generates explanations for predictions.
- Visualizes feature influence as a graph.

### `data_generator.py`
- Generates synthetic data for model training.
- Creates a dataset with numerical and categorical features.

### `train.py`
- Loads synthetic data.
- Trains the `ExplainGraph` model.
- Generates explanations for sample inputs.
- Visualizes feature influence.

## Dependencies

- `numpy`
- `pandas`
- `networkx`
- `matplotlib`
- `scikit-learn`

Install them using:

```sh
pip install -r requirements.txt
```

## Future Improvements
- Support for real-world datasets.
- Additional visualization enhancements.
- Integration with deep learning models.


## Contributing
Pull requests and suggestions are welcome! Feel free to open an issue if you find a bug or have an idea for improvement.

