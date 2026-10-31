# Car Price Prediction using Linear Regression

year = [2015, 2016, 2017, 2018, 2019, 2020]
price = [300000, 350000, 400000, 450000, 500000, 550000]

n = len(year)

mean_x = sum(year) / n
mean_y = sum(price) / n

num = sum((year[i] - mean_x) * (price[i] - mean_y) for i in range(n))
den = sum((year[i] - mean_x) ** 2 for i in range(n))

m = num / den
c = mean_y - m * mean_x

new_year = 2021
prediction = m * new_year + c

print("Car Year:", new_year)
print("Predicted Car Price:", prediction)
