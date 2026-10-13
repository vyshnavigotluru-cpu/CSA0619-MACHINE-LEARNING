
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample dataset
X = [[1], [2], [3], [4], [5], [6], [7], [8]]
y = [0, 0, 0, 0, 1, 1, 1, 1]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Create and train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict the output
y_pred = model.predict(X_test)

print("Actual Output:", y_test)
print("Predicted Output:", y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))
