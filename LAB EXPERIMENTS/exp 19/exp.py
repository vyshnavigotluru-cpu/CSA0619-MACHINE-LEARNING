
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

data = {
    'Income': [25000, 40000, 30000, 60000, 50000, 20000, 70000, 45000],
    'CreditScore': [600, 750, 650, 800, 780, 550, 850, 700],
    'Loan': [0, 1, 0, 1, 1, 0, 1, 1]
}

df = pd.DataFrame(data)

X = df[['Income', 'CreditScore']]
y = df['Loan']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Predicted Results:", y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))
