
import pandas as pd
import numpy as np
from time import time
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR, LinearSVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
import scipy
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import graphviz
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
from scipy import stats

import warnings
warnings.filterwarnings('ignore')


train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')


train_df.head()

train_df.describe()

train_df.info()

train_df['y'].value_counts()

def data_summary(df):
    print(df.shape)
    print(df.info())   
    print(df.head(5))

""" calling functions to print summary statistics of training data """
data_summary(train_df)
data_summary(test_df)


# **Removing the variables which have only one value as zero**
train_df.drop(['X11', 'X93', 'X107', 'X233', 'X235', 'X268', 'X289', 'X290', 'X293', 'X297', 'X330', 'X339', 'X347'],axis=1,inplace=True)
test_df.drop(['X11', 'X93', 'X107', 'X233', 'X235', 'X268', 'X289', 'X290', 'X293', 'X297', 'X330', 'X339', 'X347'],axis=1,inplace=True)


# **Checking for any missing values**
def check_missing_values(df):
    if df.isnull().any().any():
        print("There are missing values in the dataframe")
    else:
        print("There are no missing values in the dataframe")
        
""" calling functions to check missing values on training and test datasets """
check_missing_values(train_df)
check_missing_values(test_df)


# **Outlier Detection and plotting the graphs**
plt.figure(figsize=(8,6))
plt.scatter(range(train_df.shape[0]), np.sort(train_df.y.values))
plt.xlabel('Number of observations', fontsize=12)
plt.ylabel('Values of target variables', fontsize=12)
plt.title('Distribution of values of target variable')
plt.show()

#Histogram for training data
plt.figure(figsize=(8,6))
sns.distplot(train_df.y.values, bins=50, kde=False)
plt.xlabel('y value', fontsize=12)
plt.title('Histogram plot for the distribution of target variable')
plt.show()

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
df_num = train_df.select_dtypes(include=numerics)       #  numeric dataframe
objects = ['O']
df_cat = train_df.select_dtypes(include=objects)
print(df_num.shape,df_cat.shape)
print(df_cat.columns,'\n','--------------------------------------------------------------------------------','\n',df_num.columns)


# **Removing the outliers**
#setting the threshold as 150 to remove the outliers
temp=train_df.y.values
df_cat['y']=temp
print(df_cat.head())

print((train_df.loc[train_df.y>150,'y'].values))
train_df=train_df[train_df.y<150]
print("Removing outliers based on above information and setting 150 as a threshold value . . . . . . . . . . . . . . . . . . . . ")
print(train_df.shape)

#Taking Log Transformation
plt.figure(figsize=(9,7))
res = stats.probplot(np.log1p(train_df["y"]), plot=plt)
plt.title('Log transformation Probabiliy plot')


# **Plotting the categorical variables**

#Categorical variable: "X1"
var_name = "X1"
col_order = np.sort(train_df[var_name].unique()).tolist()
plt.figure(figsize=(12,6))
sns.stripplot(x=var_name, y='y', data=train_df, order=col_order)
plt.xlabel(var_name, fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title("Distribution of y variable with "+var_name, fontsize=15)
plt.show()

#Categorical variable: "X2"
var_name = "X2"
col_order = np.sort(train_df[var_name].unique()).tolist()
plt.figure(figsize=(12,6))
sns.boxplot(x=var_name, y='y', data=train_df, order=col_order)
plt.xlabel(var_name, fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title("Distribution of y variable with categorical variable: "+var_name, fontsize=15)
plt.show()


def initial_datatype_conversion(df):
    cols = ['X0','X1','X2','X3','X4','X5','X6','X8']
    for col in cols:
        df[col] = df[col].astype('category')
    return df

""" datatype conversion """
ret_train_df = initial_datatype_conversion(train_df)
ret_test_df = initial_datatype_conversion(test_df)

""" combining categorical attributes from training and test datasets """
train_df_cat = ret_train_df.loc[:,['X0','X1','X2','X3','X4','X5','X6','X8']]
test_df_cat = ret_test_df.loc[:,['X0','X1','X2','X3','X4','X5','X6','X8']]
train_df_cat = train_df_cat.add_prefix('train_')
test_df_cat = test_df_cat.add_prefix('test_')
combined = train_df_cat.append(test_df_cat, ignore_index=True)


# **Label Encoding for categorical variables by using get_dummies**

le = LabelEncoder()
cols = ['X0', 'X1', 'X2','X3','X4','X5','X6','X8']

ret_train_df = pd.get_dummies(ret_train_df, columns=['X0', 'X1', 'X2','X3','X4','X5','X6','X8'], prefix=['X0', 'X1', 'X2','X3','X4','X5','X6','X8'])
ret_test_df = pd.get_dummies(ret_test_df, columns=['X0', 'X1', 'X2','X3','X4','X5','X6','X8'], prefix=['X0', 'X1', 'X2','X3','X4','X5','X6','X8'])

ret_train_df.head()


cols = ret_test_df.filter(like='_bb').columns
train_X = ret_train_df.drop(['ID','y'], axis=1)
train_Y = ret_train_df['y']
train_Y = train_Y.values
test_X = ret_test_df.drop(['ID'],axis=1)
test_X = test_X.drop(cols, axis=1)

matching_cols = train_X.columns.intersection(test_X.columns)
matching_cols_list = matching_cols.tolist()

test_X = test_X[matching_cols_list]
train_X = train_X[matching_cols_list]


# **Splitting the dataset**

X_train, X_test, y_train, y_test = train_test_split(train_X, train_Y, test_size=0.33, random_state=48)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# **Standardisation of training and test data**

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# **Model Evaluation**

#Support Vector Machine
svr_clf = SVR(kernel="poly",degree=6,coef0=1,C=10)
svr_clf.fit(X_train, y_train)
pred_Y = svr_clf.predict(X_test)
r2_score_svc = round(r2_score(y_test, pred_Y),3)
accuracy = round(svr_clf.score(X_train, y_train) * 100, 2)
print(accuracy)
returnval = {'model':'SVR', 'r2_score':r2_score_svc}
print(returnval)


#Random Forest Regressor
rfr_clf = RandomForestRegressor(n_estimators = 50,max_depth=30)
rfr_clf.fit(X_train, y_train)
pred_Y = rfr_clf.predict(X_test)
r2_score_rfc = round(r2_score(y_test, pred_Y),3)
accuracy = round(rfr_clf.score(X_train, y_train) * 100, 2)
print(accuracy)
returnval = {'model':'RandomForestRegressor','r2_score':r2_score_rfc}
print(returnval)

#K-Nearest Neighbors Regressor
knn = KNeighborsRegressor(n_neighbors=3, metric = 'minkowski', p = 2)
knn.fit(X_train, y_train)
pred_Y = knn.predict(X_test)
r2_score_knn = round(r2_score(y_test, pred_Y),3)
accuracy = round(knn.score(X_train, y_train) *100,2)
print(accuracy)
returnval = {'model':'KNeighborsRegressor','r2_score':r2_score_knn}
print(returnval)

