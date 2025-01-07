import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

# Import dataset
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
dataset = pd.read_csv("C:\\Users\\lesin\\Documents\\Tools & Work\\housing.csv", delimiter=r'\s+', names=column_names)

# Display top 5 rows
print(dataset.head())

# Print dataset shape
print(f"Dataset Shape: {dataset.shape}")

# Describe the dataset
print(dataset.describe())

# Remove 'ZN' and 'CHAS' columns
dataset = dataset.drop(['ZN', 'CHAS'], axis=1)

# Check for null values
print(dataset.isnull().sum())
# Step 3: Visualize the data
# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(dataset.corr(numeric_only=True, method='pearson'), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
# Scatter plot of 'MEDV' (target) vs 'RM' (number of rooms)
sns.scatterplot(data=dataset, x='RM', y='MEDV')
plt.title("Rooms vs Median House Value")
plt.xlabel("Number of Rooms (RM)")
plt.ylabel("Median House Value (MEDV)")
plt.show()
# Histogram of the target variable
dataset['MEDV'].hist(bins=20, edgecolor='black')
plt.title("Distribution of Median House Value")
plt.xlabel("Median House Value (MEDV)")
plt.ylabel("Frequency")
plt.show()
# Step 4: Preprocess the data
# Defne features (X) and target (y)
X =dataset.drop(columns=['MEDV']) # Features
Y = dataset['MEDV'] # Target
#split training and testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Linear Regression
linear = LinearRegression()
linear.fit(X_train, Y_train)
# Step 6: Make predictions
Y_pred_linear = linear.predict(X_test)
# Step 7: Evaluate the model
mse = mean_squared_error(Y_test, Y_pred_linear)
r2 = r2_score(Y_test, Y_pred_linear)
print(f"\nModel Performance:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared Score (R��): {r2:.2f}")
# Step 8: Analyze feature importance
feature_importance = np.abs(model.coef_)
features = X.columns
# Display feature importance
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importance})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
print("\nFeature Importance:")
print(importance_df)
# Plot feature importance
plt.fgure(fgsize=(10, 6))
sns.barplot(data=importance_df, x='Importance', y='Feature', palette='viridis')
plt.title("Feature Importance")
plt.show()
# Step 9: Save processed data to a CSV fle
output_fle = "eduardo_boston.csv"# Import necessary libraries
