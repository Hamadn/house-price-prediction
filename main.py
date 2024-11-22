import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("./Real estate.csv")

dataset.info()
dataset.head()
dataset.tail()
dataset.describe()

X = dataset[
    [
        "X2 house age",
        "X3 distance to the nearest MRT station",
        "X4 number of convenience stores",
        "X5 latitude",
        "X6 longitude",
    ]
]
y = dataset["Y house price of unit area"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.7, color="c", edgecolor="k")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("House Price Prediction: Actual vs Prediction Prices")
plt.show()


mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error:", mae)

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error:", mse)
