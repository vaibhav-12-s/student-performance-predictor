import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Sample dataset
data = {
    'Hours': [1, 2, 3, 4, 5, 6, 7, 8],
    'Marks': [10, 20, 30, 40, 50, 60, 70, 80]
}

df = pd.DataFrame(data)

# Features and target
X = df[['Hours']]
y = df['Marks']

# Model training
model = LinearRegression()
model.fit(X, y)

# Input from user
try:
    hours = float(input("Enter study hours: "))
except:
    print("Invalid input. Please enter a number.")
    exit()

# Prediction (fixed warning issue)
input_df = pd.DataFrame([[hours]], columns=['Hours'])
prediction = model.predict(input_df)

print(f"Predicted Marks: {prediction[0]:.2f}")

# Graph visualization
# Graph visualization
plt.scatter(X, y, color='blue', label="Training Data")
plt.plot(X, model.predict(X), color='red', label="Regression Line")

# Plot user input point
plt.scatter(hours, prediction[0], color='green', s=100, label="Your Input")

plt.xlabel("Hours Studied")
plt.ylabel("Marks")
plt.title("Student Performance Prediction")
plt.legend()
plt.show()