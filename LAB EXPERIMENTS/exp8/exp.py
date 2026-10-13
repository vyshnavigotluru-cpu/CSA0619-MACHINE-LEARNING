
from sklearn.linear_model import LinearRegression

# Sample dataset
X = [[1], [2], [3], [4], [5]]
y = [10, 20, 30, 40, 50]

# Create the model
model = LinearRegression()

# Train the model
model.fit(X, y)

# Predict the output
prediction = model.predict([[6]])

print("Predicted value:", prediction[0])
