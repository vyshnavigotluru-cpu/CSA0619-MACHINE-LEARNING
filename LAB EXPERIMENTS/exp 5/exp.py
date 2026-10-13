
from sklearn.neighbors import KNeighborsClassifier

# Dataset
X = [[1, 2], [2, 3], [3, 4], [6, 7], [7, 8], [8, 9]]
Y = [0, 0, 0, 1, 1, 1]

# Create K-NN model
model = KNeighborsClassifier(n_neighbors=3)

# Train the model
model.fit(X, Y)

# Test the model
prediction = model.predict([[2, 4]])

print("Predicted Class:", prediction[0])
