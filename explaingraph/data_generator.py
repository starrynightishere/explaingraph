import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def generate_synthetic_data():
    np.random.seed(42)
    X = pd.DataFrame({
        'Feature1': np.random.normal(0, 1, 100),
        'Feature2': np.random.uniform(-1, 1, 100),
        'Feature3': np.random.randint(0, 10, 100),
        'Feature4': np.random.choice([0, 1], 100)
    })
    y = (X['Feature1'] + 2 * X['Feature2'] - 0.5 * X['Feature3'] + 0.8 * X['Feature4'] + np.random.normal(0, 0.5, 100)) > 0
    return X, y.astype(int)

def load_data():
    X, y = generate_synthetic_data()
    return train_test_split(X, y, test_size=0.2, random_state=42)
