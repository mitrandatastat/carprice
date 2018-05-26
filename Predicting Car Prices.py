
# coding: utf-8

# ### Introduction

# In[1]:


import math as m
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from operator import itemgetter
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
get_ipython().run_line_magic('matplotlib', 'inline')

# Decide about the database column headers by getting detail data insight at: https://archive.ics.uci.edu/ml/datasets/automobile
col_name = ['symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style', 
            'drive-wheels', 'engine-location', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type', 
            'num-of-cylinders', 'engine-size', 'fuel-system', 'bore', 'stroke', 'compression-rate', 'horsepower', 'peak-rpm', 
            'city-mpg', 'highway-mpg', 'price']


# ### Getting Information About the Dataset

# In[2]:


cars = pd.read_csv(r"./databank/import-85.csv", names=col_name)
cars.head(5)


# In[3]:


print(cars.info())


# ### Select Columns with Continuous Data Values

# In[4]:


# Using data info available at: http://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.names
col_vals = ['normalized-losses', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 'bore', 'stroke', 
            'compression-rate', 'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price']
cars_num = cars[col_vals]
cars_num.head(5)


# ### Prepare Data for Missing Value Fill-up by Replacing Special Characters

# In[5]:


# As seen above, the dataframe carries "?". Replace it with null value
cars_num = cars_num.replace("?", np.NaN)
cars_num.head(5)


# ### Convert All Values to Same Data Type

# In[6]:


# Convert all columns of the dataframe to float data type
cars_num = cars_num.astype('float64')
# Check for all columns data type
cars_num.info()


# ### Select Target Column

# In[7]:


target_col = cars_num['price']


# ### Remove Null Values from Target Columns

# In[8]:


cars_num = cars_num.dropna(subset=['price'], axis=0)
cars_num['price'].isnull().sum()


# ### Check for Null Values in Other Columns

# In[9]:


cars_num.isnull().sum()


# ### Fill out Missing Column Values

# In[10]:


# Fill out the missing values with column mean
cars_num = cars_num.fillna(cars_num.mean())
# Check for no null values
cars_num.isnull().sum().sum()


# ### Feature Scaling

# In[11]:


temp_target = cars_num['price']
cars_num = (cars_num - cars_num.mean())/(cars_num.max() - cars_num.min())
cars_num['price'] = temp_target
cars_num.head(5)


# ### Develop Univariate Model

# In[12]:


# Set up a function that develop model and return RMSE

def knn_train_test(train_col, test_col, data):    

# Randomize the data 
    np.random.seed(1)
    shuflled_idx = np.random.permutation(data.index)
    data_rand = data.reindex(shuflled_idx)

# Divide the data set 50/50 between the training and test set 
    break_pt = int(round(len(data)/2, 0))
    train_df = data_rand.iloc[:break_pt]
    test_df = data_rand.iloc[break_pt+1:]
    
# Defining the Training and Test Set
    train_features = train_df[[train_col]]
    train_target = train_df[test_col]
    test_features = test_df[[train_col]]
    test_target = test_df[test_col]

# Develop KNeighbor Regression
    knn = KNeighborsRegressor()     
    knn.fit(train_features, train_target)
    predictions = knn.predict(test_features)

# Calculate the model error matrix
    mse_val = mean_squared_error(test_target, predictions)
    rmse = m.sqrt(mse_val)
    return rmse

trn_col = cars_num.drop(labels='price', axis=1).columns
tst_col = 'price'
rmse_dict = {}

for col in trn_col:
    rmse_dict[col] = knn_train_test(col, 'price', cars_num)
rmse_dict


# ### Univariate Model with Multiple Clusters

# In[13]:


# Defining the number of split values as variable
cluster_num = [1, 3, 5, 7, 9]

# Set up function that develop model that can handle multiple split values and generate RMSE
def knn_train_test_splits(train_col, test_col, data): 
    
# Randomize the data 
    np.random.seed(1)
    shuflled_idx = np.random.permutation(data.index)
    data_rand = data.reindex(shuflled_idx)

# Divide the data set 50/50 between the training and test set 
    break_pt = int(round(len(data)/2, 0))
    train_df = data.iloc[:break_pt]
    test_df = data.iloc[break_pt+1:]

# Defining the Training and Test Set
    train_features = train_df[[train_col]]
    train_target = train_df[test_col]
    test_features = test_df[[train_col]]
    test_target = test_df[test_col]

# Develop KNeighbor Regression
    rmse_val = {}
    for i in cluster_num:
        knn = KNeighborsRegressor(n_neighbors = i) 
        knn.fit(train_features, train_target)
        predictions = knn.predict(test_features)

# Calculate the model error matrix
        mse_val = mean_squared_error(test_target, predictions)
        rmse_val[i] = m.sqrt(mse_val)
    return rmse_val

# Establish RMSE dictionary with varying clusters for every feature of the dataset
RMSE_MC = {}
trn_col = cars_num.drop(labels='price', axis=1).columns
tst_col = 'price'
for i in trn_col:
    RMSE_MC[i] = knn_train_test_splits(i, tst_col, cars_num)

RMSE_MC


# ### Plot the Univariate Multicluster Model Results

# In[14]:


# Developing the plot series
for k, v in RMSE_MC.items():
    X = list(v.keys())
    y = list(v.values())
    
    plt.plot(X, y)
    plt.xlabel("Cluster Number")
    plt.ylabel("RMSE")
    plt.title("K-Nearest Neighbors Univariate Model Results")
plt.show()


# ### Develop Multivariate Model (Default Clusters)

# In[15]:


# Develop Multivariate Model with Single Default Neighbor Value as 5 from scikit-learn

def knn_train_test_mv(train_col, test_col, data):    

# Randomize the data 
    np.random.seed(1)
    shuflled_idx = np.random.permutation(data.index)
    data_rand = data.reindex(shuflled_idx)

# Divide the data set 50/50 between the training and test set 
    break_pt = int(round(len(data)/2, 0))
    train_df = data_rand.iloc[:break_pt]
    test_df = data_rand.iloc[break_pt+1:]
    
# Defining the Training and Test Set
    train_features = train_df[train_col]
    train_target = train_df[test_col]
    test_features = test_df[train_col]
    test_target = test_df[test_col]

# Develop KNeighbor Regression
    knn = KNeighborsRegressor()     
    knn.fit(train_features, train_target)
    predictions = knn.predict(test_features)

# Calculate the model error matrix
    mse_val = mean_squared_error(test_target, predictions)
    rmse = m.sqrt(mse_val)
    return rmse

# Trying different best features and verify how RMSE varies 
# To identify the preference of top 5 best features, let's extract 
# the RMSE value in ascending order from the previous Univariate Model

best_features = {}
for k, v in RMSE_MC.items():
    best_features[k] = sum(v.values())/len(v.values())

top5_nn = sorted(best_features.items(), key=itemgetter(1))
print(top5_nn)

rmse_mv = {}

two_best_features = ['horsepower', 'highway-mpg']
three_best_features = ['horsepower', 'highway-mpg', 'city-mpg']
four_best_features = ['horsepower', 'highway-mpg', 'city-mpg','curb-weight']
five_best_features = ['horsepower', 'highway-mpg', 'city-mpg','curb-weight', 'width']
six_best_features = ['horsepower', 'highway-mpg', 'city-mpg','curb-weight', 'width', 'compression-rate']

rmse_mv_keys = ['two_best_features', 'three_best_features', 'four_best_features', 'five_best_features', 'six_best_features']
rmse_mv_vals = [two_best_features, three_best_features, four_best_features, five_best_features, six_best_features]

for i in range(len(rmse_mv_keys)):
    rmse_mv[rmse_mv_keys[i]] = knn_train_test_mv(rmse_mv_vals[i], 'price', cars_num)
    
rmse_mv


# ### Multivariate Model With Multiple Clusters

# In[16]:


# Develop Multivariate Model with Multiple Neighbor Values Using the Top 6 Best Features from above

def knn_train_test_mvc(train_col, test_col, data):    

# Randomize the data 
    np.random.seed(1)
    shuflled_idx = np.random.permutation(data.index)
    data_rand = data.reindex(shuflled_idx)

# Divide the data set 50/50 between the training and test set 
    break_pt = int(round(len(data)/2, 0))
    train_df = data_rand.iloc[:break_pt]
    test_df = data_rand.iloc[break_pt+1:]    

# Defining the Training and Test Set
    train_features = train_df[train_col]
    train_target = train_df[test_col]
    test_features = test_df[train_col]
    test_target = test_df[test_col]

# Defining the cluster values from 1 - 25
    k_val = [x for x in range(25)]
    rmse_val_mvc = {}
    
# Develop KNeighbor Regression
    for k in k_val:
        knn = KNeighborsRegressor(n_neighbors = k + 1)     
        knn.fit(train_features, train_target)
        predictions = knn.predict(test_features)

# Calculate the model error matrix
        mse_val_mvc = mean_squared_error(test_target, predictions)
        rmse_val_mvc[k + 1] = m.sqrt(mse_val_mvc)
    return rmse_val_mvc

rmse_mvc = {}

two_best_features = ['horsepower', 'highway-mpg']
three_best_features = ['horsepower', 'highway-mpg', 'city-mpg']
four_best_features = ['horsepower', 'highway-mpg', 'city-mpg','curb-weight']
five_best_features = ['horsepower', 'highway-mpg', 'city-mpg','curb-weight', 'width']
six_best_features = ['horsepower', 'highway-mpg', 'city-mpg','curb-weight', 'width', 'compression-rate']

rmse_mvc_keys = ['two_best_features', 'three_best_features', 'four_best_features', 'five_best_features', 'six_best_features']
rmse_mvc_vals = [two_best_features, three_best_features, four_best_features, five_best_features, six_best_features]

for i in range(len(rmse_mvc_keys)):
    rmse_mvc[rmse_mvc_keys[i]] = knn_train_test_mvc(rmse_mvc_vals[i], 'price', cars_num)
    
rmse_mvc


# ### Plot the Multivariate Multicluster Model Result

# In[17]:


# Developing the plot series
for k, v in rmse_mvc.items():
    X = list(v.keys())
    y = list(v.values())
    
    plt.plot(X, y)
    plt.xlabel("Cluster Number")
    plt.ylabel("RMSE")
    plt.title("K-Nearest Neighbors Multivariate Model Results")
plt.show()

Let's improve model performance by using KFold Cross Validation Method.
# ### Develop Univariate Model Using KFold Cross Validation

# In[18]:


# Develop Univariate KFold Model with default Split Value and default number of neighbors from scikit-learn

def kfold_train_test(train_col, test_col, data):    
    
# Divide the data set 50/50 between the training and test set & randomize the data
    kfd = KFold(n_splits = 3, shuffle=True)

# Develop KNeighbor Regression
    knn = KNeighborsRegressor(n_neighbors=5)
    for train_index, test_index, in kfd.split(data):
        train_kf = data.iloc[train_index]
        test_kf = data.iloc[test_index]
            
        train_features = train_kf[[train_col]]
        train_target = train_kf[test_col]
        test_features = test_kf[[train_col]]
        test_target = test_kf[test_col]
            
        knn.fit(train_features, train_target)
        predictions = knn.predict(test_features)

# Calculate the model error matrix
    
        mse_val_kfd = mean_squared_error(test_target, predictions)
        rmse_val_kfd = m.sqrt(mse_val_kfd)
    return rmse_val_kfd

trn_col = cars_num.drop(labels='price', axis=1).columns
tst_col = 'price'
rmse_val_kfd = {}

for col in trn_col:
    rmse_val_kfd[col] = kfold_train_test(col, 'price', cars_num)
rmse_val_kfd


# ### Univariate Model Using KFold Cross Validation with Multiple Clusters

# In[19]:


# Set up a KFold cross validation that develop model with multiple clusters, 5 data folds and return RMSE
clusters = [1, 3, 5, 7, 9]

def kfold_train_test_mc(train_col, test_col, data):    
    
# Divide the data set between the training and test set & randomize the data
    kfd = KFold(n_splits = 5, shuffle=True)
    kfold_rmse = []
    rmse_kfd = {}

# Develop KNeighbor Regression.
    for i in clusters:
        knn = KNeighborsRegressor(n_neighbors=i)
        for train_index, test_index, in kfd.split(data):
            train_kf = data.iloc[train_index]
            test_kf = data.iloc[test_index]
            
            train_features = train_kf[[train_col]]
            train_target = train_kf[test_col]
            test_features = test_kf[[train_col]]
            test_target = test_kf[test_col]
            
            knn.fit(train_features, train_target)
            predictions = knn.predict(test_features)

# Calculate the model error matrix  
            mse_val_kfd = mean_squared_error(test_target, predictions)
            kfold_rmse.append(m.sqrt(mse_val_kfd))
        rmse_kfd[i] = np.mean(kfold_rmse)
    return rmse_kfd

trn_col = cars_num.drop(labels='price', axis=1).columns
tst_col = 'price'
rmse_kfd_mc = {}

for col in trn_col:
    rmse_kfd_mc[col] = kfold_train_test_mc(col, 'price', cars_num)
rmse_kfd_mc


# ### Plot Univariate Multicluster Kfold Cross Validation Result

# In[20]:


# Developing the Kfold plot series
for k, v in rmse_kfd_mc.items():
    X = list(v.keys())
    y = list(v.values())
    
    plt.plot(X, y)
    plt.xlabel("Cluster Number")
    plt.ylabel("RMSE")
    plt.title("Kfold Univariate Model Results")
plt.show()


# ### Identify the Top 5 Best Features from KFold Cross Validation

# In[21]:


best_kfd_features = {}
for k, v in rmse_kfd_mc.items():
    best_kfd_features[k] = sum(v.values())/len(v.values())

# Print features in ascending order of error values
top5_kfd = sorted(best_kfd_features.items(), key=itemgetter(1))
print(top5_kfd)


# ### Multivariate KFold Cross Validation With Multiple Clusters

# In[22]:


# Set up a KFold cross validation that develop model with multiple clusters, 10 data folds and return RMSE
clusters_mvmc = [x for x in range(1, 26)]

def kfold_train_test_mvmc(train_col, test_col, data):    
    
# Divide the data set between the training and test set & randomize the data
    kfd = KFold(n_splits = 10, shuffle=True)
    kfold_rmse_mvmc = []
    rmse_kfd_mvmc = {}

# Develop KNeighbor Regression 
    for k in clusters_mvmc:
        knn = KNeighborsRegression(n_neighbors= k)
        for train_index, test_index, in kfd.split(data):
            train_kf = data.iloc[train_index]
            test_kf = data.iloc[test_index]
            
            train_features = train_kf[[train_col]]
            train_target = train_kf[test_col]
            test_features = test_kf[[train_col]]
            test_target = test_kf[test_col]
            
            knn.fit(train_features, train_target)
            predictions = knn.predict(test_features)

# Calculate the model error matrix  
            mse_val_mvmc = mean_squared_error(test_target, predictions)
            kfold_rmse_mvmc.append(m.sqrt(mse_val_mvmc))
        rmse_kfd_mvmc[i] = np.mean(kfold_rmse_mvmc)
    return rmse_kfd_mvmc

rmse_mvmc = {}

two_best_features = ['horsepower', 'curb-weight']
three_best_features = ['horsepower', 'curb-weight', 'width']
four_best_features = ['horsepower', 'curb-weight', 'width','city-mpg']
five_best_features = ['horsepower', 'curb-weight', 'width','city-mpg', 'highway-mpg']
six_best_features = ['horsepower', 'curb-weight', 'width','city-mpg', 'highway-mpg', 'wheel-base']

rmse_mvmc_keys = ['two_best_features', 'three_best_features', 'four_best_features', 'five_best_features', 'six_best_features']
rmse_mvmc_vals = [two_best_features, three_best_features, four_best_features, five_best_features, six_best_features]

for i in range(len(rmse_mvmc_keys)):
    rmse_mvmc[rmse_mvmc_keys[i]] = knn_train_test_mvc(rmse_mvmc_vals[i], 'price', cars_num)

rmse_mvmc


# ### Plot Multivariate Multicluster KFold Cross Validation Result

# In[23]:


# Developing the Kfold plot series
for k, v in rmse_mvmc.items():
    X = list(v.keys())
    y = list(v.values())
    
    plt.plot(X, y)
    plt.xlabel("Cluster Number")
    plt.ylabel("RMSE")
    plt.title("Kfold Multivariate Model Results")
plt.show()


# ### Compare Results of Two Different Model Approaches

# #### Univariate Model Errors

# In[24]:


UVE = pd.DataFrame.from_dict(rmse_dict, orient='index')
UVE.columns = ["Nearest Neighbors"]
UVE["KFold"] = pd.DataFrame.from_dict(rmse_val_kfd, orient='index')
uve_delta = ((UVE['KFold'] - UVE['Nearest Neighbors'])*100/UVE['Nearest Neighbors']).mean()
print("Overall change in the error matrix = " + str(round(uve_delta, 2)) + "%\n")
UVE


# #### Multivariate Model Errors for Multiclusters

# In[25]:


multiclusters_nn = {}
mutliclusters_kfd = {}

for k, v in RMSE_MC.items():
    multiclusters_nn[k] = sum(v.values())/len(v.values())

for k, v in rmse_kfd_mc.items():
    mutliclusters_kfd[k] = sum(v.values())/len(v.values())

MCE = pd.DataFrame.from_dict(multiclusters_nn, orient='index')
MCE.columns = ["Nearest Neighbors"]
MCE["KFold"] = pd.DataFrame.from_dict(mutliclusters_kfd, orient='index')
mce_delta = ((MCE['KFold'] - MCE['Nearest Neighbors'])*100/MCE['Nearest Neighbors']).mean()
print("Overall change in the error matrix = " + str(round(mce_delta, 2)) + "%\n")
MCE


# #### Multivariate Model Errors for Top 5 Best Features

# In[26]:


# Getting the average value of each of the top 5 best features
multivariate_nn = {}
mutlivariate_kfd = {}

for k, v in rmse_mvc.items():
    multivariate_nn[k] = sum(v.values())/len(v.values())

for k, v in rmse_mvmc.items():
    mutlivariate_kfd[k] = sum(v.values())/len(v.values())

MVE = pd.DataFrame.from_dict(multivariate_nn, orient='index')
MVE.columns = ["Nearest Neighbors"]
MVE["KFold"] = pd.DataFrame.from_dict(mutlivariate_kfd, orient='index')
mve_delta = ((MVE['KFold'] - MVE['Nearest Neighbors'])*100/MVE['Nearest Neighbors']).mean()
print("Overall change in the error matrix = " + str(round(mve_delta, 2)) + "%\n")
MVE


# ### Conclusion
We can summarize our findings from the project outcome as: 

1.	Reducing the data noise by eliminating the data gaps, 
2.	effective data cleaning with proper understanding of the data, 
3.	filling out the missing data properly feature mean values, 
4.	careful selection of various features of high significance value, 
5.	reasonably increasing the number data points/clusters, and 
6.	use of proper model validation technique

can help to improve the predictive model’s accuracy for care sale price. 