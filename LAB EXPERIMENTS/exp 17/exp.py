import pandas as pd
from sklearn.ensemble import RandomForestClassifier

data = {
    'RAM': [2, 3, 4, 6, 8, 12],
    'Battery': [2000, 2500, 3000, 3500, 4500, 5000],
    'Storage': [16, 32, 64, 64, 128, 256],
    'Price': [0, 0, 1, 1, 2, 3]
}

df = pd.DataFrame(data)

X = df[['RAM', 'Battery', 'Storage']]
y = df['Price']

model = RandomForestClassifier(random_state=42)
model.fit(X, y)

prediction = model.predict([[8, 4500, 128]])

print("Predicted Price Range:", prediction[0])
