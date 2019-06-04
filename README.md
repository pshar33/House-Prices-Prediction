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
     * Reading the images stored in 2 folders(Train,Test).
     * Removing redundant columns.
     * Data imputation for missing data (nulls or NA)
     * Removing outliers
     * Removing skewness in target variable
     * One hot encoding for categorical features.
3. Data Augmentation: Augment the train,validation and test data using ImageDataGenerator
4. Creating and Training the Model: Create a cnn model in KERAS.
5. Evaluation: Display the plots from the training history.
6. Prediction: Run predictions with model.predict
7. Conclusion: Comparing original labels with predicted labels and calculating recall score.

## Accuracy and loss plots
