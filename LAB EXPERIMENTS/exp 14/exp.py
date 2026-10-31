import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Create dataset
data = {
    'Area': [1000, 1200, 1500, 1800, 2000, 2200],
    'Price': [2000000, 2400000, 3000000, 3600000, 4000000, 4400000]
}

df = pd.DataFrame(data)

# Input and output
X = df[['Area']]
y = df['Price']

# Train model
model = LinearRegression()
model.fit(X, y)

# Prediction
prediction = model.predict([[1600]])

print("House Area:", 1600)
print("Predicted House Price:", prediction[0])

# Graph
plt.scatter(X, y)
plt.plot(X, model.predict(X))
plt.xlabel("House Area")
plt.ylabel("House Price")
plt.title("House Price Prediction")
plt.show()
