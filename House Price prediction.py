import pandas as pd
import numpy as np
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


Sheet_ID = "1hBTtrIK8BeIhAptyt-ukS0fqJ-0cfBeA_n1hO0MCPXA"
Sheet_GID = "0"

Dataframe = pd.read_csv(f"https://docs.google.com/spreadsheets/d/{Sheet_ID}/export?format=csv&gid={Sheet_GID}")

for column in Dataframe.columns:
    if Dataframe[column].dtype in ['float64', 'int64']:
        Dataframe[column].replace ("Unknown", np.nan, inplace = True)
    else: 
        Dataframe[column].fillna("Unknown", inplace = True)


Dataframe['date'] = pd.to_datetime(Dataframe['date'], errors = "coerce")

print("Data Types: ")
print(Dataframe.dtypes)

numeric_df = Dataframe.select_dtypes(include=[int, float])
print("Summary Statistics (Numeric Columns Only): ")
print(numeric_df.describe())
print("\nCorrelation Matrix (Numeric Columns Only): ")
print(numeric_df.corr())

print(Dataframe.head())

features = ["area", "code", "houses_sold",	"no_of_crimes",	"borough_flag"]
x = Dataframe[features] # Features (excluding targets)
y = Dataframe["average_price"] #Target variable
x = x.dropna()
y = y[x.index]

for feature in features :
    x[feature] = x[feature].astype(str).apply(lambda v: v.lower().replace(' ', '_'))

Categorical_features = ["area", "code", "borough_flag"]

preprocessor = ColumnTransformer(transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), Categorical_features), ("num", StandardScaler(), ["houses_sold", "no_of_crimes"])], remainder="passthrough")

#Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)


#Define a function to train and evaluate models
def evaluate_model(model, x_train, x_test, y_train, y_test, model_name):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{model_name} - Mean Squared Error: {mse:.2f}, R-squared: {r2:.2f}")


#Linear regression pipeline
linear_model = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", LinearRegression())])
evaluate_model(linear_model, x_train, x_test, y_train, y_test, "Linear Regression")


#Ridge regression pipeline
ridge_model = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", Ridge(alpha=1.0))])
evaluate_model(ridge_model, x_train, x_test, y_train, y_test, "Ridge Regression")


#Lasso regression pipeline
lasso_model = Pipeline(steps = [("preprocessor", preprocessor), ("regressor", Lasso(alpha=0.65, max_iter=20000))])
evaluate_model(lasso_model, x_train, x_test, y_train, y_test, "Lasso Regression")