
import pandas as pd
from sklearn.linear_model import LinearRegression

data = {
    'Month': [1, 2, 3, 4, 5, 6, 7, 8],
    'Sales': [100, 120, 130, 150, 160, 180, 190, 210]
}

df = pd.DataFrame(data)

X = df[['Month']]
y = df['Sales']

model = LinearRegression()
model.fit(X, y)

future_month = pd.DataFrame({'Month': [9, 10, 11, 12]})

prediction = model.predict(future_month)

for month, sales in zip(future_month['Month'], prediction):
    print("Month", month, "Predicted Sales:", round(sales, 2))
