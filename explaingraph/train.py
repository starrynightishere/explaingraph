from explaingrad import ExplainGraph
from data_generator import load_data
import pandas as pd

# Load synthetic dataset
X_train, X_test, y_train, y_test = load_data()

# Train ExplainGraph model
model = ExplainGraph()
model.fit(X_train, y_train)

# Test explanation
sample_input = X_test.iloc[:1]
prediction, confidence = model.predict(sample_input)
explanations = model.explain(sample_input)

print("Prediction:", prediction[0], "with confidence:", f"{confidence[0]:.2f}")

print("Explanation:")
for exp in explanations:
    print(exp)

model.visualize_explanation(sample_input)
