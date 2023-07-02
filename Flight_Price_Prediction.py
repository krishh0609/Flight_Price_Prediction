#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Install libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
get_ipython().run_line_magic('matplotlib', 'inline')

sns.set()


# In[2]:


# TRAIN DATA


# In[3]:


# Read Train Data
df_train = pd.read_excel('Data_Train.xlsx')


# In[4]:


#df_train['Arrival_Min']=df_train['Arrival_Min'].astype(int)


# In[5]:


df_train.head()


# In[6]:


df_train.shape


# In[7]:


df_train.info()


# In[8]:


df_train['Date_of_Journey'].isnull().sum()


# In[9]:


# Split Time Series data
df_train['Journey_Date'] = pd.to_datetime(df_train['Date_of_Journey'],format="%d/%m/%Y").dt.day
df_train['Journey_Month'] = pd.to_datetime(df_train['Date_of_Journey'],format="%d/%m/%Y").dt.month
#df_train['Journey_Year'] = pd.to_datetime(df_train['Date_of_Journey'],format="%d/%m/%Y").dt.year

# Drop Data_of_Journey Column
df_train.drop(['Date_of_Journey'],axis=1,inplace=True)


# In[10]:


df_train.head(3)


# In[11]:


# Split Time Series data
df_train['Dept_hour'] = pd.to_datetime(df_train['Dep_Time'],format="%H:%M").dt.hour
df_train['Dept_min'] = pd.to_datetime(df_train['Dep_Time'],format="%H:%M").dt.minute

# Drop Dept_Time
df_train.drop('Dep_Time',axis=1,inplace=True)


# In[12]:


df_train.head(1)


# In[13]:


#df_final['Arrival_hour'] = pd.to_datetime(df_final['Arrival_Time'],format="%H:%M").dt.hour
#df_final['Arrival_minute'] = pd.to_datetime(df_final['Arrival_Time'],format="%H:%M").dt.minute
#df_final.drop('Arrival_Time',axis=1,inplace=True)12162


# Split Time Series data
df_train['Arrival_hour'] = df_train['Arrival_Time'].str.split(':').str[0]
df_train['Arrival_min'] = df_train['Arrival_Time'].str.split(':').str[1]
df_train['Arrival_Min'] = df_train['Arrival_min'].str.split(' ').str[0]

# Convert series data into integer
df_train['Arrival_hour']=df_train['Arrival_hour'].astype(int)
df_train['Arrival_Min']=df_train['Arrival_Min'].astype(int)


# In[14]:


df_train.head(5)


# In[15]:


# Drop Arrival_min,Arrival_Time
df_train.drop(['Arrival_min','Arrival_Time'],axis=1,inplace=True)


# In[16]:


df_train.head(2)


# In[17]:


# Checking Null Values
df_train['Total_Stops'].isnull().sum()


# In[18]:


# Find mode() of Total_Stops
df_train['Total_Stops'].mode()


# In[19]:


df_train['Total_Stops'].unique()


# In[20]:


#df_train['Total_Stops'] = df_train['Total_Stops'].replace({'non-stop':0, '2 stops':2, '1 stop':1, '3 stops':3, 'nan':1, '4 stops':4},inplace=True)
#df_train['Total_Stops'] = df_train['Total_Stops'].map({'non-stop':int(0), '2 stops':int(2), '1 stop':int(1), '3 stops':int(3), 'nan':int(1), '4 stops':int(4)})
df_train['Total_Stops'] = df_train['Total_Stops'].map({'non-stop':'0', '2 stops':'2', '1 stop':'1', '3 stops':'3', 'nan':'1', '4 stops':'4'})


# In[21]:


df_train['Total_Stops'].unique()


# In[22]:


# Find NaN value in which row.
df_train[df_train['Total_Stops'].isnull()]


# In[23]:


# Drop NaN Row
# Reason bcoz it creates redundancy in preprocessing
df_train=df_train.drop([9039],axis=0)


# In[24]:


df_train.info()


# In[25]:


# Check whether Total_Stops is null or not
df_train[df_train['Total_Stops'].isnull()]


# In[26]:


df_train.head(3)


# In[27]:


df_train.info()


# In[28]:


df_train.head(3)


# In[29]:


duration = list(df_train['Duration'])
for i in range(len(duration)):
    if len(duration[i].split()) != 2:
        if "h" in duration[i]:
            duration[i] = duration[i].strip() + " 0m"
        else:
            duration[i] = "0h " + duration[i]
            
duration_hours = []   
duration_mins = []  

for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep="h")[0]))
    duration_mins.append(int(duration[i].split(sep="m")[0].split()[-1]))
            


# In[30]:


df_train['Duration_hour'] = duration_hours
df_train['Duration_min'] = duration_mins


# In[31]:


df_train.head(3)


# In[32]:


df_train.drop('Duration',axis=1,inplace=True)
df_train.head(3)


# In[33]:


# Categorical Feature Engineering
# OneHotEncoding ---> get_dummies


# In[34]:


df_train['Airline'].value_counts()


# In[35]:


# Draw Plot
sns.catplot(y="Price",x="Airline",data=df_train.sort_values('Price',ascending=False),kind="boxen",height=6,aspect=3)
plt.show()


# In[36]:


# Get_dummies (OneHotEncoding)

Airline = df_train[['Airline']]
Airline = pd.get_dummies(Airline,drop_first=True)
Airline.head()


# In[37]:


df_train['Source'].value_counts()


# In[38]:


sns.catplot(y="Price",x="Source",data=df_train.sort_values('Price',ascending=False),kind="boxen",height=6,aspect=3)
plt.show()


# In[39]:


Source = df_train[['Source']]
Source = pd.get_dummies(Source,drop_first=True)
Source.head()


# In[40]:


df_train['Destination'].value_counts()


# In[41]:


sns.catplot(y="Price",x="Destination",data=df_train.sort_values('Price',ascending=False),kind="boxen",height=6,aspect=3)
plt.show()


# In[42]:


Destination = df_train[['Destination']]
Destination = pd.get_dummies(Destination,drop_first=True)
Destination.head()


# In[43]:


df_train['Route']


# In[44]:


df_train['Additional_Info']


# In[45]:


df_train.drop(['Route','Additional_Info'],axis=1,inplace=True)


# In[46]:


df_train.head()


# In[47]:


df_train_final = pd.concat([df_train,Airline,Source,Destination],axis=1)


# In[48]:


df_train_final.head()


# In[49]:


df_train_final.drop(['Airline','Source','Destination'],axis=1,inplace=True)


# In[50]:


df_train_final.head(10)


# In[51]:


df_train_final.shape


# In[52]:


df_train_final.info()


# In[53]:


# TEST DATA


# In[54]:


# Read Test data
df_test = pd.read_excel('Test_Set.xlsx')


# In[55]:


df_test.head()


# In[56]:


df_test.info()


# In[57]:


# Split Time Series data
df_test['Journey_Date'] = pd.to_datetime(df_test['Date_of_Journey'],format="%d/%m/%Y").dt.day
df_test['Journey_Month'] = pd.to_datetime(df_test['Date_of_Journey'],format="%d/%m/%Y").dt.month
#df_test['Journey_Year'] = pd.to_datetime(df_test['Date_of_Journey'],format="%d/%m/%Y").dt.year

# Drop Date_of_Journey
df_test.drop(['Date_of_Journey'],axis=1,inplace=True)


# In[58]:


df_test.head()


# In[59]:


# Split Time Series data
df_test['Arrival_hour'] = df_test['Arrival_Time'].str.split(':').str[0]
df_test['Arrival_min'] = df_test['Arrival_Time'].str.split(':').str[1]
df_test['Arrival_Min'] = df_test['Arrival_min'].str.split(' ').str[0]

# Convert series data into integer
df_test['Arrival_hour'] = df_test['Arrival_hour'].astype(int)
df_test['Arrival_Min'] = df_test['Arrival_Min'].astype(int)


# In[60]:


df_test.head()


# In[61]:


# Split Time Series data
df_test['Dept_hour'] = pd.to_datetime(df_test['Dep_Time'],format="%H:%M").dt.hour
df_test['Dept_min'] = pd.to_datetime(df_test['Dep_Time'],format="%H:%M").dt.minute

# Drop Dept_Time
df_test.drop('Dep_Time',axis=1,inplace=True)


# In[62]:


# Drop Arrival_min,Arrival_Time
df_test.drop(['Arrival_min','Arrival_Time'],axis=1,inplace=True)


# In[63]:


df_test.head(2)


# In[64]:


df_test['Total_Stops'].isnull().sum()


# In[65]:


df_test['Total_Stops'].mode()


# In[66]:


df_test['Total_Stops'].unique()


# In[67]:


df_test['Total_Stops'] = df_test['Total_Stops'].map({'non-stop':0, '2 stops':2, '1 stop':1, '3 stops':3, 'nan':1, '4 stops':4})


# In[68]:


df_test.head()


# In[69]:


duration = list(df_test['Duration'])
for i in range(len(duration)):
    if len(duration[i].split()) != 2:
        if "h" in duration[i]:
            duration[i] = duration[i].strip() + " 0m"
        else:
            duration[i] = "0h " + duration[i]
            
duration_hours = []   
duration_mins = []  

for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep="h")[0]))
    duration_mins.append(int(duration[i].split(sep="m")[0].split()[-1]))
            


# In[70]:


df_test['Duration_hour'] = duration_hours
df_test['Duration_min'] = duration_mins


# In[71]:


df_test.head()


# In[72]:


df_test.drop('Duration',axis=1,inplace=True)
df_test.head(3)


# In[73]:


df_test['Airline'].value_counts()


# In[74]:


Airline = df_test[['Airline']]
Airline = pd.get_dummies(Airline,drop_first=True)
Airline.head()


# In[75]:


df_test['Source'].value_counts()


# In[76]:


Source = df_test[['Source']]
Source = pd.get_dummies(Source,drop_first=True)
Source.head()


# In[77]:


df_test['Destination'].value_counts()


# In[78]:


Destination = df_test[['Destination']]
Destination = pd.get_dummies(Destination,drop_first=True)
Destination.head()


# In[79]:


df_test['Route']


# In[80]:


df_test['Additional_Info']


# In[81]:


df_test.drop(['Route','Additional_Info'],axis=1,inplace=True)


# In[82]:


df_test.head()


# In[83]:


df_test_final = pd.concat([df_test,Airline,Source,Destination],axis=1)


# In[84]:


df_test_final.head()


# In[85]:


df_test_final.drop(['Airline','Source','Destination'],axis=1,inplace=True)


# In[86]:


df_test_final.head(10)


# In[87]:


df_test_final.shape


# In[88]:


df_test_final.info()


# In[89]:


# Feature Selection
# Finding out the best feature which will contribute and have good relation with target varible.
# Following are some of the feature selection methods.

# 1.Heatmap
# 2.feature_importance
# 3.selectKBest


# In[90]:


df_train_final.shape


# In[91]:


df_train_final.columns


# In[92]:


X = df_train_final.loc[:,['Total_Stops', 'Journey_Date', 'Journey_Month',
       'Dept_hour', 'Dept_min', 'Arrival_hour', 'Arrival_Min', 'Duration_hour',
       'Duration_min', 'Airline_Air India', 'Airline_GoAir', 'Airline_IndiGo',
       'Airline_Jet Airways', 'Airline_Jet Airways Business',
       'Airline_Multiple carriers',
       'Airline_Multiple carriers Premium economy', 'Airline_SpiceJet',
       'Airline_Trujet', 'Airline_Vistara', 'Airline_Vistara Premium economy',
       'Source_Chennai', 'Source_Delhi', 'Source_Kolkata', 'Source_Mumbai',
       'Destination_Cochin', 'Destination_Delhi', 'Destination_Hyderabad',
       'Destination_Kolkata', 'Destination_New Delhi']]
X.head()


# In[93]:


y = df_train_final.iloc[:,1]
y.head()


# In[94]:


# Find correlation between Independent and dependent attributes

plt.figure(figsize = (20,20))
sns.heatmap(df_train.corr(),annot=True,cmap="YlGnBu")
plt.show()


# In[95]:


# Important features using ExtraTreesRegressor

from sklearn.ensemble import ExtraTreesRegressor
selection = ExtraTreesRegressor()
selection.fit(X,y)


# In[96]:


print(selection.feature_importances_)


# In[97]:


# Plot Important features in graphical form

plt.figure(figsize=(12,10))
feat_importances = pd.Series(selection.feature_importances_,index=X.columns)
feat_importances.nlargest(20).plot(kind="barh")
plt.show()


# In[98]:


# Fitting model using Random Forest
   
# 1.Split dataset into train and test set in order to prediction w.r.t X_test
# 2.If needed do scaling of data
#    a.Scaling is not done in Random forest   
# 3.Import model
# 4.Fit the data
# 5.Predict w.r.t X_test
# 6.In regression check RSME Score
# 7.Plot graph


# In[99]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=42)


# In[100]:


from sklearn.ensemble import RandomForestRegressor
reg_rf = RandomForestRegressor()
reg_rf.fit(X_train,y_train)


# In[101]:


y_pred = reg_rf.predict(X_test)


# In[102]:


reg_rf.score(X_train,y_train)


# In[103]:


reg_rf.score(X_test,y_test)


# In[104]:


sns.distplot(y_test-y_pred)
plt.show()


# In[105]:


plt.scatter(y_test,y_pred,alpha = 0.5)
plt.xlabel("y_test")
plt.ylabel("y_pred")
plt.show()


# In[106]:


from sklearn import metrics


# In[107]:


print('MAE:',metrics.mean_absolute_error(y_test,y_pred))
print('MSE:',metrics.mean_squared_error(y_test,y_pred))
print('MAE:',np.sqrt(metrics.mean_squared_error(y_test,y_pred)))


# In[108]:


metrics.r2_score(y_test,y_pred)


# In[109]:


# Hyperparameter Tuning
# 1.Choose following method for hyperparameter tuning
#    a.RandamizedSearchCV --> Fast
#    b.GridSearchCV
# 2.Assign hyperparameters in form of dictionery
# 3.Fit the model
# 4.Check best parameter and best score


# In[110]:


from sklearn.model_selection import RandomizedSearchCV


# In[111]:


# RandomizedSearchCV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=100,stop=1200,num=12)]

# Number of features to consider at every split
max_features = ['auto','sqrt']

# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5,30,num=6)]

# Minimum number of samples required to split a node
min_samples_split = [2,5,10,15,100]

# Minimum number of samples required at each leaf node
min_samples_leaf = [1,2,5,10]


# In[112]:


# Create the random grid

random_grid = {
    'n_estimators':n_estimators,
    'max_features':max_features,
    'max_depth':max_depth,
    'min_samples_split':min_samples_split,
    'min_samples_leaf' :min_samples_leaf
}


# In[113]:


rf_random = RandomizedSearchCV(estimator = reg_rf,param_distributions = random_grid  ,scoring ='neg_mean_squared_error',n_iter = 10, 
                               cv = 5 ,verbose=2,random_state=42,n_jobs=1)


# In[114]:


rf_random.fit(X_train,y_train)


# In[115]:


rf_random.best_params_


# In[116]:


prediction = rf_random.predict(X_test)


# In[117]:


plt.figure(figsize=(8,8))
sns.distplot(y_test-prediction)
plt.show()


# In[118]:


plt.figure(figsize=(8,8))
plt.scatter(y_test,prediction,alpha=0.5)
plt.xlabel("y_test")
plt.ylabel("y_pred")
plt.show()


# In[119]:


print('MAE:',metrics.mean_absolute_error(y_test,prediction))
print('MSE:',metrics.mean_squared_error(y_test,prediction))
print('MAE:',np.sqrt(metrics.mean_squared_error(y_test,prediction)))


# In[ ]:


# Save the model to reuse it again


# In[120]:


import pickle


# In[129]:


# Open a file where you opt to store the data
file = open('flight_rf.pkl','wb')

# Dump information to that file
pickle.dump(rf_random,file)


# In[130]:


model = open('flight_rf.pkl','rb')
forest = pickle.load(model)


# In[131]:


y_prediction = forest.predict(X_test)


# In[132]:


metrics.r2_score(y_test,y_prediction)

