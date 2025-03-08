import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
from plotnine import ggplot, aes, geom_line, geom_point  # Importing plotnine for visualization

# Sample dataset
data = {
    "age": [25, 45, 35, 50, 23],
    "income": [5000, 10000, 7500, 12000, 4000],
    "credit_score": [700, 650, 750, 600, 720],
    "loan_amount": [10000, 50000, 30000, 80000, 12000],
    "approved": [1, 0, 1, 0, 1]
}
df = pd.DataFrame(data)

X = df.drop(columns=["approved"])
y = df["approved"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
with open("model/loan_model.pkl", "wb") as file:
    pickle.dump(model, file)

# Visualization example
plot = (ggplot(df, aes(x='income', y='approved')) + 
        geom_point() + 
        geom_line())
print(plot)  # Display the plot
