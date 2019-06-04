# House-Prices-Prediction
Predicting house prices with Gradient Boosting Regression.

## Code Requirements

* Numpy
* Pandas
* Seaborn,matplotlib
* Scipy
* Sklearn

## Description

This is a supervised learning prediction problem on Kaggle Competitions.The dataset contains 2 folders -train.csv , test.csv
To download the data files visit :-  https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data.


## Breakdown of the code:

1. Loading the dataset: Load the data and import the libraries.
2. Data Preprocessing:
     * Reading the data stored in 2 folders(Train,Test).
     * Removing redundant columns.
     * Data imputation for missing data (nulls or NA)
     * Removing outliers
     * Removing skewness in target variable
     * One hot encoding for categorical features.
3. Using GridSearchCV to find best hyperparameters for Gradient Boosting Regressor.
4. Individual ordered testing to get best hyperparameters by varying one parameter and keeping rest constant.
5. Implementing GradientBoostingRegressor with hyperparameters obtained above.
6. Prediction: Run predictions and calculate cross_validation_score and accuracies for target variable(house prices)
7. Submission:Writing the predicted house prices and ids to csv file and submitting at Kaggle.

## Accuracy scores

KAGGLE SCORE (RMSE) = 0.13
