import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from scipy import stats
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score


#read the train and test csv file
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
train_parent=train_data
test_parent=test_data

#drop the id column from train and test data
train_data = train_data.drop('Id', axis=1)
test_data = test_data.drop('Id', axis=1)

#plot to see the columns with most NA values
na_val = train_data.isna().sum().reset_index(name="total_na").sort_values(by="total_na", ascending=False)
fig,ax = plt.subplots(1,figsize=(8,8))
sns.barplot(ax=ax,x="total_na", y="index", data=na_val)
plt.show()

#use 50% threshold to get the redundant columns
threshold = 0.5 * len(train_data)
df=pd.DataFrame(len(train_data) - train_data.count(),columns=['count'])
# df.index[df['count'] > threshold]
# df.index[df['count']]


#drop the 5 redundant columns
train_data = train_data.drop(['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature', 'Utilities'], axis=1)
test_data = test_data.drop(['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature','Utilities'], axis=1)

#segregate training data into numeric,categorical,float
features = train_data.columns.values
float_data = []
int_data = []
categorical_data = []
year_data = []

for feature in features:
    dtype = train_data[feature].dtypes
    if feature == 'SalePrice':
        continue
    elif 'Yr' in feature or 'Year' in feature:
        year_data.append(feature)
    elif dtype == 'int64':
        int_data.append(feature)
    elif dtype == 'float64':
        float_data.append(feature)
    else:
        categorical_data.append(feature)


print("int data is ",int_data)
print("float data is ",float_data)
print("categorical data is ",categorical_data)

#get the skewness of the target variable.If its skewed, make it normal for better linear regression.
# print ("Skew is:", train_data.SalePrice.skew())
plt.hist(train_data.SalePrice, color='blue')
plt.show()

#since its skewed,log is taken
train_data.SalePrice = np.log(train_data.SalePrice)
# train_data.SalePrice=np.exp(train_data.SalePrice)
# print ("Skew is:", train_data.SalePrice.skew())
plt.hist(train_data.SalePrice, color='blue')
plt.show()

#remove outliers
def remove_outlier(df, data):
    for feature in (data):
        col = pd.DataFrame(df[feature].dropna())
        col_in = col[(np.abs(stats.zscore(col)) < 3).all(axis=1)]
        col_mean = int(col_in.mean())
        col_out = col[(np.abs(stats.zscore(col)) >= 3).all(axis=1)]
        df.loc[list(col_out.index.values), feature] = col_mean

remove_outlier(train_data, int_data)
remove_outlier(train_data, float_data)
remove_outlier(test_data, int_data)
remove_outlier(test_data, float_data)

print("after removing outliers, train data is ",train_data)

def fill_na(df):
    null_columns = df.columns[df.isnull().any()]
    null_rows = df[df.isnull().any(axis=1)][null_columns]

    garage_yr_med = df['GarageYrBlt'].dropna().median()
    df['GarageYrBlt'].fillna(garage_yr_med, inplace=True)

    masvnr_mean = df['MasVnrArea'].dropna().mean()
    # masvnr_mode = train['MasVnrArea'].dropna().mode()
    df['MasVnrArea'].fillna(masvnr_mean, inplace=True)

    lot_mean = df['LotFrontage'].dropna().mean()
    df['LotFrontage'] = pd.DataFrame(df['LotFrontage']).fillna(lot_mean)

    elec_mode = df['Electrical'].dropna().mode()
    df['Electrical'] = pd.DataFrame(df['Electrical']).fillna(elec_mode[0])


fill_na(train_data)
fill_na(test_data)

#remove na values from numeric columns in both train and test data
for col in int_data:
    train_data[col] = train_data[col].fillna(0)
    test_data[col] = test_data[col].fillna('0')

# # remove na values from numeric columns in both train and test data
    for col in float_data:
        train_data[col] = train_data[col].fillna(0)
        test_data[col] = test_data[col].fillna('0')

# remove na values from categorical columns in both train and test data
    for col in categorical_data:
        train_data[col] = train_data[col].fillna('None')
        test_data[col] = test_data[col].fillna('None')


#check for any null value in train and test data
# train_data[train_data.isnull().any(axis=1)]
# test_data[test_data.isnull().any(axis=1)]
# print("after removing nulls, train data is ",train_data)



#one hot encoding
train=train_data
test=test_data

#Assigning a flag to training and testing dataset for segregation after OHE .
ntrain = train.shape[0]
ntest = test.shape[0]

#Combining training and testing dataset
sales=train_data.SalePrice
combined=pd.concat([train,test])
combined.drop(['SalePrice'], axis=1, inplace=True)

from scipy import stats
from scipy.stats import norm, skew

numeric_feats = combined.dtypes[combined.dtypes != "object"].index

# Check the skew of all numerical features
skewed_feats = combined[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
print(skewness.head(10))

skewness = skewness[abs(skewness) > 0.75]
print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

from scipy.special import boxcox1p, inv_boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    #all_data[feat] += 1
    combined[feat] = boxcox1p(combined[feat], lam)






ohe_data_frame=pd.get_dummies(combined,
                           columns=['MSZoning', 'Street', 'LotShape', 'LandContour',
       'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
       'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
       'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
       'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
       'Functional', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
       'PavedDrive', 'SaleType', 'SaleCondition'],)


#Splitting the combined dataset after doing OHE .
train_df=ohe_data_frame[:ntrain]
print("tdf",train_df)

test_df=ohe_data_frame[ntrain:]
print("testdf",test_df)

# train_df.drop(['flag'],axis=1,inplace=True)             #Drop the Flag(train) coloumn from training dataset
# test_df.drop(['SalePrice'],axis=1,inplace=True)     #Drop the Flag(train),Label(SalePrice) coloumn from test dataset

#putting back one hot encoded data into original train and test
train_data=train_df
test_data=test_df

#defining xtrain,ytrain,xtest
X_train = train_data
# Taking the labels (price)
Y_train = sales
X_test = test_data



num_estimators = [500,1000,3000]
learn_rates = [0.01, 0.02, 0.05, 0.1]
max_depths = [1, 2, 3, 4]
min_samples_leaf = [5,10,15]
min_samples_split = [2,5,10]

param_grid = {'n_estimators': num_estimators,
              'learning_rate': learn_rates,
              'max_depth': max_depths,
              'min_samples_leaf': min_samples_leaf,
              'min_samples_split': min_samples_split}

grid_search = GridSearchCV(GradientBoostingRegressor(loss='huber'),
                           param_grid, cv=3, return_train_score=True)
grid_search.fit(X_train, Y_train)

print(grid_search.best_params_, grid_search.best_score_)

params = {'n_estimators': 3000, 'max_depth': 2, 'min_samples_leaf':5, 'min_samples_split':2,
          'learning_rate': 0.02, 'loss': 'huber','max_features':'sqrt'}
gbr_model = GradientBoostingRegressor(**params)
gbr_model.fit(X_train, Y_train)

print(gbr_model.score(X_train, Y_train))

y_grad_predict = gbr_model.predict(X_test)
print(np.exp(y_grad_predict))




##individual ordered testing
param_test1 = {'n_estimators':range(200,3000,100)}
gsearch1 = GridSearchCV(GradientBoostingRegressor(learning_rate=0.1, min_samples_split=2,min_samples_leaf=5,max_depth=2,max_features='sqrt'),
param_grid = param_test1, cv=5,scoring='neg_mean_absolute_error', return_train_score=True)
gsearch1.fit(X_train, Y_train)

print(gsearch1.best_params_)
print(gsearch1.best_score_)

param_test2 = {'max_depth':range(2,10,2), 'min_samples_split':range(2,4,1)}
gsearch2 = GridSearchCV(GradientBoostingRegressor(learning_rate=0.1,min_samples_leaf=5, n_estimators=1200, max_features='sqrt'),
param_grid = param_test2, scoring='neg_mean_absolute_error', cv=5, return_train_score=True)
gsearch2.fit(X_train, Y_train)

print(gsearch2.best_params_)
print(gsearch2.best_score_)


param_test3 = {'min_samples_leaf':range(2,7,1)}
gsearch3 = GridSearchCV(GradientBoostingRegressor(learning_rate=0.1, n_estimators=1200,max_depth=2,max_features='sqrt', min_samples_split=2),
param_grid = param_test3, scoring='neg_mean_absolute_error', cv=5, return_train_score=True)
gsearch3.fit(X_train, Y_train)
print(gsearch3.best_params_, gsearch3.best_score_)

param_test4 = {'max_features':range(7,20,2)}
gsearch4 = GridSearchCV(GradientBoostingRegressor(learning_rate=0.1, n_estimators=1200,max_depth=2, min_samples_split=2, min_samples_leaf=5),
param_grid = param_test4, scoring='neg_mean_absolute_error', cv=5)
gsearch4.fit(X_train, Y_train)
print(gsearch4.best_params_, gsearch4.best_score_)


params = {'n_estimators': 1200, 'max_depth': 2, 'min_samples_leaf':5, 'min_samples_split':2,
          'learning_rate': 0.1, 'loss': 'huber','max_features':17}
gbr_model1 = GradientBoostingRegressor(**params)
gbr_model1.fit(X_train, Y_train)
print("Score for learning rate=0.1, n_estimators=1200 is:- ",gbr_model1.score(X_train, Y_train))  # score returns the coefficient of determination R^2 of the prediction.
y_predict = gbr_model1.predict(X_test)
y_predict=np.exp(y_predict)
print("prediction for learning rate=0.1, n_estimators=1200 is:- ",y_predict)


#half the learning rate, twice the n_estimators
params = {'n_estimators': 2400, 'max_depth': 2, 'min_samples_leaf':5, 'min_samples_split':2,
          'learning_rate': 0.05, 'loss': 'huber','max_features':17}
gbr_model2 = GradientBoostingRegressor(**params)
gbr_model2.fit(X_train, Y_train)
print(gbr_model2.score(X_train, Y_train))
y_predict2 = gbr_model2.predict(X_test)
print("prediction for learning rate 0.05, n_estimators 2400 ",np.exp(y_predict2))


# #one fifth the learning rate, five times the n_estimators
params = {'n_estimators': 6000, 'max_depth': 1, 'min_samples_leaf':15, 'min_samples_split':10,
          'learning_rate': 0.01, 'loss': 'huber','max_features':17}
gbr_model3 = GradientBoostingRegressor(**params)
gbr_model3.fit(X_train, Y_train)
print("Score for learning rate=0.07, n_estimators=3000 is:- ",gbr_model3.score(X_train, Y_train))   # score returns the coefficient of determination R^2 of the prediction.
y_predict3 = gbr_model3.predict(X_test)
# X_test=inv_boxcox1p(X_test,lam)
# y_predict3=inv_boxcox1p(y_predict3,lam)
y_predict3=np.exp(y_predict3)
print("prediction for learning rate=0.07, n_estimators=3000 is:- ",y_predict3)

#Cross validation score for training data
scores = cross_val_score(gbr_model3, X_train, Y_train, cv=10)
print("the cross val train score is",scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


#Cross validation score for testing data
scores_test = cross_val_score(gbr_model3, X_test, y_predict3, cv=10)
print("the cross val test score is",scores_test)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores_test.mean(), scores_test.std() * 2))

my_submission = pd.DataFrame({'Id': test_parent.Id, 'SalePrice': y_predict3})
print(my_submission)

my_submission.to_csv('submission.csv', encoding='utf-8', index=False)
