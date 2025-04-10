import pandas as pd
from pycaret.classification import *

# Load Iris dataset
df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")

# Setup PyCaret
clf = setup(data=df, target='species', session_id=123, html=False)

# Train and compare models
best_model = compare_models()

# Save the best model
save_model(best_model, 'iris_model')

