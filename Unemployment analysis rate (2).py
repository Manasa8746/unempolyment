import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

file_path = r"c:\Users\manas\Downloads\Unemployment in India.csv"   # Replace with your actual file name
data = pd.read_csv(r"c:\Users\manas\Downloads\Unemployment in India.csv", on_bad_lines='skip')

data.columns = data.columns.str.strip()
data.columns = data.columns.str.replace(r'[\s()%]', '', regex=True)

print(data.columns.tolist())

print("Dataset Preview:\n", data.head())

data['Date'] = pd.to_datetime(data['Date'], errors='coerce', dayfirst=True)
print("Columns:", data.columns.tolist())

print("\nMissing Values:\n", data.isnull().sum())

data = data.dropna()

print("\nMissing Values After Cleaning:\n", data.isnull().sum())

print("\nDataset Info:\n")
print(data.info())

print("\nSummary Statistics:\n")
print(data.describe(include='all'))

plt.figure(figsize=(12, 6))
sns.lineplot(data=data, x='Date', y='EstimatedUnemploymentRate', hue='Region')
plt.title("Unemployment Rate Over Time by Region", fontsize=14)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Unemployment Rate (%)", fontsize=12)
plt.legend(title='Region', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.barplot(data=data, x='Area', y='EstimatedUnemploymentRate', estimator='mean', errorbar=None)
plt.title("Average Unemployment Rate by Area", fontsize=13)
plt.xlabel("Area", fontsize=11)
plt.ylabel("Unemployment Rate (%)", fontsize=11)
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(data[['EstimatedUnemploymentRate', 'EstimatedEmployed',
                  'EstimatedLabourParticipationRate']].corr(),
            annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Between Employment Factors", fontsize=13)
plt.show()

X = data[['EstimatedEmployed', 'EstimatedLabourParticipationRate']]
y = data['EstimatedUnemploymentRate']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
print("\nModel Performance:")
print("R2 Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, color='teal')
plt.xlabel("Actual Unemployment Rate (%)")
plt.ylabel("Predicted Unemployment Rate (%)")
plt.title("Actual vs Predicted Unemployment Rate")
plt.show()

sample = pd.DataFrame({
    'EstimatedEmployed': [5000000],
    'EstimatedLabourParticipationRate': [45.0]
})

pred = model.predict(sample)
print("Predicted Unemployment Rate:", pred)
