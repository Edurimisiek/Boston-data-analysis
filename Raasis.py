import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score

# Import dataset
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
dataset = pd.read_csv("C:\\Users\\lesin\\Documents\\.Tools & Work\\#coding skills\Python\\datasets\\boston_housing\\housing.csv", delimiter=r'\s+', names=column_names)
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
plt.ylabel("Frequency")#hii pia ni noma sana
plt.show()
#Step 4: Preprocess the data
# Defne features (X) and target (y)
X =dataset.drop(columns=['MEDV']) # Features
y = dataset['MEDV'] # Target
#splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Step 5: Train a linear regression model
#LRM using scikit-learn
from sklearn.linear_model import LinearRegression
#create a linear regression model
model = LinearRegression()
#train the model on the training data
model.fit(X_train, y_train)
#Evaluating the model
#predict MEDV for test data
y_pred = model.predict(X_test)
# Standardize the feature data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Step 6: Make predictions
y_pred = model.predict(X_test)
# Step 7: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nModel Performance:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared Score (R²): {r2:.2f}")
# Step 8: Analyze feature importance
feature_importance = np.abs(model.coef_)
features = X.columns
# Display feature importance
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importance})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
print("\nFeature Importance:")
print(importance_df)
# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(data=importance_df, x='Importance', y='Feature')
plt.title("Feature Importance")
plt.show()
#BONUS CHALLENGE
#FEATURES = RM, LSTAT, PTRATIO
new_features = ['RM', 'LSTAT', 'PTRATIO']
X_new = dataset[new_features]
#splitting data
X_train_new,X_test_new,y_train_new,y_test_new = train_test_split(X_new, y, test_size=0.2, random_state=42)
model_new = LinearRegression()
model_new.fit(X_train_new, y_train_new)
#evaluate and compare
y_pred_new = model_new.predict(X_test_new)
mse_new = mean_squared_error(y_test_new, y_pred_new)
r2_new = r2_score(y_test_new, y_pred_new)
print(f"New Model MSE: {mse_new}, New R2: {r2_new}")


# Check and treat outliers

# Plotting boxplots to visualize outliers
fig, ax = plt.subplots(ncols=6, nrows=2, figsize=(15, 5))
ax = ax.flatten()
index = 0
for i in dataset.columns:
    sns.boxplot(y=i, data=dataset, ax=ax[index])
    index += 1
plt.tight_layout(pad=0.4)
plt.show()

# Calculate and print outlier percentages
for i in dataset.columns:
    q1, q3 = np.nanpercentile(dataset[i], [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    outlier_data = dataset[i][(dataset[i] < lower_bound) | (dataset[i] > upper_bound)]
    perc = (outlier_data.count() / dataset[i].count()) * 100
    print(f'Outliers in {i}: {perc:.2f}% ({outlier_data.count()} values)')

# Handle outliers (e.g., removal, imputation) - This part would require further decisions based on your analysis

# Feature Selection

# Independent and dependent variables
X = dataset.iloc[:, :-1]
Y = dataset.iloc[:, -1]

# Backward Elimination (using p-values)
def BackwardElimination(sl, w):
    for i in range(0, len(w.columns)):
        regressor_OLS = sm.OLS(endog=Y, exog=w).fit()
        max_pvalue = max(regressor_OLS.pvalues)
        pvalues = regressor_OLS.pvalues
        if max_pvalue > sl:
            index_max_pvalue = pvalues[pvalues == max_pvalue].index
            w = w.drop(index_max_pvalue, axis=1)
    return w, pvalues, index_max_pvalue

SL = 0.05
ones = np.ones((len(X), 1))
W = X
W.insert(0, 'Constant', ones, True)
W_optimal = W.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]
W_optimal, pvalues, index_max_pvalue = BackwardElimination(SL, W_optimal)
X = W_optimal.drop('Constant', axis=1)


# Print remaining variables after backward elimination
print(f"Remaining variables after backward elimination: {X.columns}")

# Check for multicollinearity
plt.figure(figsize=(8, 8))
sns.heatmap(X.corr(method='pearson').abs(), annot=True, square=True)
plt.show()

# Drop highly correlated features (e.g., 'TAX' and 'NOX')
X.drop('TAX', axis=1, inplace=True)
X.drop('NOX', axis=1, inplace=True)

# Check correlation of remaining features with MEDV
print("Correlation of remaining features with MEDV:")
for i in X.columns:
    corr = X[i], Y
    print(f"{i}: {corr}")

# Drop features with low correlation (e.g., 'DIS', 'RAD')
X.drop(['DIS', 'RAD'], axis=1, inplace=True)

# Print final selected features
print(f"Final selected features: {X.columns}")

# Machine Learning Models

# Polynomial Regression
poly_reg = PolynomialFeatures(degree=3)
X_train_poly = poly_reg.fit_transform(X_train)
X_test_poly = poly_reg.transform(X_test)
poly = LinearRegression()
poly.fit(X_train_poly, y_train)
Y_pred_poly = poly.predict(X_test_poly)

# Support Vector Regression
svr = SVR(kernel='poly', gamma='scale')
svr.fit(X_train, y_train)
Y_pred_svr = svr.predict(X_test)

# Random Forest Regression
rf = RandomForestRegressor(n_estimators=100)
rf.fit(X_train, y_train)
Y_pred_rf = rf.predict(X_test)

# K-Nearest Neighbors Regression
knn = KNeighborsRegressor(n_neighbors=13)
knn.fit(X_train, y_train)
Y_pred_knn = knn.predict(X_test)

# Compare actual and predicted values
models = [
    ('Linear', y_pred),
    ('Polynomial', Y_pred_poly),
    ('Support Vector', Y_pred_svr),
    ('Random Forest', Y_pred_rf),
    ('KNN', Y_pred_knn)
]

for name, y_pred in models:
    df_compare = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    print(f"\n{name} Regression:")
    print(df_compare.head())

# Plot actual vs. predicted values
fig, ax = plt.subplots(ncols=5, nrows=1, figsize=(25, 4))
ax = ax.flatten()
for i, (name, y_pred) in enumerate(models):
    df_compare = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    df_compare.head(10).plot(kind='bar', title=f'{name} Regression', grid=True, ax=ax[i])
plt.show()

# Calculate and compare R-squared scores using cross-validation
model_names = ['Linear', 'Polynomial', 'Support Vector', 'Random Forest', 'KNN']
model_regressors = [model, poly, svr, rf, knn]
scores = []

for i, model in enumerate(model_regressors):
    if model is poly:
        accuracy = cross_val_score(model, X_train_poly, y_train, scoring='r2', cv=5)
    else:
        accuracy = cross_val_score(model, X_train, y_train, scoring='r2', cv=5)
    print(f'Accuracy of {model_names[i]} Regression model: {accuracy.mean():.2f}')
    scores.append(accuracy.mean())

# Plot R-squared scores
pd.DataFrame({'Model Name': model_names, 'Score': scores}).sort_values(by='Score', ascending=True).plot(
    x=0, y=1, kind='bar', figsize=(15, 5), title='Comparison of R2 scores of different models'
)
plt.show()

# Conclusion (based on R-squared scores and analysis)
