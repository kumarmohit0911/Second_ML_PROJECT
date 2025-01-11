# Economic Index Prediction Using Linear Regression

## Overview
This project uses a dataset containing economic indicators such as interest rates, unemployment rates, and index prices. By applying simple and multiple linear regression techniques, we aim to predict the economic index price based on the given features. The project leverages Python libraries for data processing, visualization, and model building.

## Key Features
- Data cleaning and preprocessing.
- Exploratory Data Analysis (EDA) with visualizations using Matplotlib, Seaborn, and Plotly.
- Multiple Linear Regression for predicting index price based on independent variables.
- Calculation of regression coefficients and prediction of economic index prices.

## Installation and Setup

### Prerequisites
Ensure you have the following Python libraries installed:
- pandas
- matplotlib
- numpy
- seaborn
- plotly
- scikit-learn

Install any missing libraries using pip:
```bash
pip install pandas matplotlib numpy seaborn plotly scikit-learn
```

### Dataset
The project requires a CSV file named `economic_index.csv`. Ensure the dataset is in the same directory as the script.

## Code Workflow

### 1. Importing Libraries and Loading the Data
The following libraries are used:
- pandas: For data manipulation and analysis.
- matplotlib and seaborn: For data visualization.
- plotly: For interactive 3D visualization.
- scikit-learn: For data preprocessing and regression modeling.

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
```

### 2. Data Cleaning
Unnecessary columns (`Unnamed: 0`, `year`, `month`) are removed for simplicity.
```python
df_index.drop(columns=["Unnamed: 0", "year", "month"], axis=1, inplace=True)
```

### 3. Exploratory Data Analysis (EDA)
Visualize relationships between features using scatter plots and Seaborn's `pairplot`.
```python
sns.pairplot(df_index)
plt.scatter(df_index['interest_rate'], df_index['unemployment_rate'], color='r')
plt.xlabel("Interest Rate")
plt.ylabel("Unemployment Rate")
```

### 4. Splitting Data into Train and Test Sets
The data is split into training and testing sets with a 75-25 ratio.
```python
from sklearn.model_selection import train_test_split
X = df_index.iloc[:, :-1]
y = df_index.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
```

### 5. Data Scaling
Standardize the data using `StandardScaler` to improve model performance.
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

### 6. Building the Model
A linear regression model is trained on the standardized data.
```python
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train, y_train)
```

### 7. Making Predictions
Use the trained model to predict test set results.
```python
y_pred = regression.predict(X_test)
```

### 8. Model Coefficients and Prediction
The regression equation is:
\[ \hat{y} = b + m_1x_1 + m_2x_2 \]

Where:
- \( m_1 \): Coefficient for `interest_rate`
- \( m_2 \): Coefficient for `unemployment_rate`
- \( b \): Intercept

Example prediction for specific inputs (interest rate = 1.75, unemployment rate = 6.2):
```python
m = regression.coef_
m1 = m[0]
m2 = m[1]
b = regression.intercept_
y_hat = b + (m1 * 1.75) + (m2 * 6.2)
print(y_hat)
```

### 9. Visualization
A 3D scatter plot visualizes the relationship between features and index price:
```python
fig = px.scatter_3d(df_index, x='interest_rate', y='unemployment_rate', z='index_price')
fig.show()
```

## Results and Metrics
Evaluate model performance using:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- R-squared (\( R^2 \))

## Acknowledgments
This project was created to understand the application of linear regression for economic data prediction. Special thanks to the authors of the dataset and the Python open-source community.

## Future Enhancements
- Include additional features to improve prediction accuracy.
- Implement more advanced regression techniques (e.g., polynomial regression).
- Deploy the model using a web framework (e.g., Flask or Django).

