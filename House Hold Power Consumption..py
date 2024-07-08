#!/usr/bin/env python
# coding: utf-8

# # 1.Problem statement

# This dataset contains Minute by minute powerconsumption data for a Single household in Sceaux (7km of Paris, France) between December 2006 and November 2010 (47 months) 

# # 2. Data collection

# The dataset was collected from the UCI ML repository.Link for the dataset- https://archive.ics.uci.edu/ml/datasets/Individual+household+electric+power+consumption

# # 3. Data Ingestion

# Importing the necessary libraries

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# In[4]:


df=pd.read_csv('household_power_consumption.txt',sep=';')


# In[5]:


df.head()


# In[6]:


df.shape


# # 4. Data Cleaning

# In[7]:


df.info()


# In[8]:


df.isnull().sum()


# It can be seen that there are 25979 null values in sub_metering_3

# ##### Since the dataset is too large so we'll select a sample of the dataset having 100000 values and use it for further study

# In[9]:


df_sample=df.sample(100000)


# In[10]:


df_sample.head()


# In[11]:


df_sample.reset_index(inplace=True,drop=True)


# In[12]:


df_sample.head()


# In[13]:


df_sample.shape


# In[14]:


df_sample.isnull().sum()


# In[15]:


df_sample.info()


# In[16]:


df_sample['Global_active_power'] = pd.to_numeric(df_sample['Global_active_power'], errors='coerce')
df_sample['Global_reactive_power'] = pd.to_numeric(df_sample['Global_reactive_power'], errors='coerce')
df_sample['Voltage'] = pd.to_numeric(df_sample['Voltage'], errors='coerce')
df_sample['Global_intensity'] = pd.to_numeric(df_sample['Global_intensity'], errors='coerce')
df_sample['Sub_metering_1'] = pd.to_numeric(df_sample['Sub_metering_1'], errors='coerce')
df_sample['Sub_metering_2'] = pd.to_numeric(df_sample['Sub_metering_2'], errors='coerce')


# In[17]:


df_sample['Global_active_power'].fillna(df_sample['Global_active_power'].mean(),inplace=True)
df_sample['Global_reactive_power'].fillna(df_sample['Global_reactive_power'].mean(),inplace=True)
df_sample['Voltage'].fillna(df_sample['Voltage'].mean(),inplace=True)
df_sample['Global_intensity'].fillna(df_sample['Global_intensity'].mean(),inplace=True)
df_sample['Sub_metering_1'].fillna(df_sample['Sub_metering_1'].mean(),inplace=True)
df_sample['Sub_metering_2'].fillna(df_sample['Sub_metering_2'].mean(),inplace=True)
df_sample['Sub_metering_3'].fillna(df_sample['Sub_metering_3'].mean(),inplace=True)
     


# In[18]:


df_sample['Date']=pd.to_datetime(df_sample['Date'])
df_sample['Day']=df_sample['Date'].dt.day
df_sample['Month']=df_sample['Date'].dt.month
df_sample['Year']=df_sample['Date'].dt.year


# In[24]:


df_sample['Hour']=pd.to_datetime(df_sample['Time'],format='%H:%M:%S').dt.hour
df_sample['Minutes']=pd.to_datetime(df_sample['Time'],format='%H:%M:%S').dt.minute


# In[25]:


df_sample.head()


# In[26]:


df_sample.isnull().sum()


# In[27]:


df.duplicated().sum()


# In[28]:


df_sample.info()


# ##### Creating column for total metering

# In[260]:


df_sample['Total_metering']=df_sample['Sub_metering_1']+df_sample['Sub_metering_2']+df_sample['Sub_metering_3']


# In[261]:


df_sample.head()


# ###### Dropping the unwanted columns from the dataset

# In[30]:


df_sample.drop(columns=['Date','Time','Sub_metering_1', 'Sub_metering_2',
       'Sub_metering_3'],inplace=True)


# In[31]:


df_sample.head()


# # 5. EDA

# ##### Features information 

# In[32]:


df_sample.columns


# There are total 9 features in the dataset which are:

# * date: Date in format dd/mm/yyyy
# * time: time in format hh: mm: ss
# * global_active_power: household globalminute-averaged active power (in kilowatt)
# * global_reactive_power: household globalminute-averaged reactive power (in kilowatt)
# * voltage: minute-averaged voltage (in volt)
# * global_intensity: household globalminute-averaged current intensity (in ampere) 
# * sub_metering_1: energy sub-metering No.1 (in watt-hour of active energy). It corresponds to the kitchen, containing mainly a dish washer, an oven and amicrowave (hot plates are not electric but gas powered). 
# * sub_metering_2: energy sub-metering No.2 (in watt-hour of active energy). It corresponds to the laundry room,containing a washing-machine, a tumble-drier, a refrigerator and a light 
# * sub_metering_3: energy sub-metering No.3 (in watt-hour of active energy). It corresponds to an electric water-heater andan air-conditioner

# ##### Statistical description of the dataset

# In[265]:


df_sample.describe().T


# In[266]:


plt.figure(figsize=(15,17))
plt.suptitle("Univariate Analysis",fontsize=20,fontweight='bold',y=1)
for i in range(0,len(df_sample.columns)):
    plt.subplot(5,3,i+1)
    sns.kdeplot(x=df_sample[df_sample.columns[i]],shade=True,color='r')
    plt.xlabel(df_sample.columns[i])
    plt.tight_layout()


# Observations:
# * Global_active_power- Power is distributed between 0 to 8. Most of the power distributed between 0 to 2.
# * Global_reactive_power-Reactive power is distributed between 0 to 0.8. Most of the power distributed between 0 to 2.
# * Voltage= Vlotage is distributed between 230 to 250, most of the voltage distributed between 0 to 10.
# * Global_intensity- Intensity is distributed between 0 to 20. Most of the intensity distrubuted between 0 to 10.
# * Total_metering= metering is distributed between 0 to 60 most of the distribution is between 0 to 25.

# In[267]:


#Realtion of each feature with Total_metering
plt.figure(figsize=(15,17))
plt.suptitle("Total metering vs each feature",fontsize=20,fontweight='bold',y=1)
for i in range(0,len(df_sample.columns)):
    plt.subplot(5,3,i+1)
    sns.scatterplot(x=df_sample['Total_metering'],y=df_sample[df_sample.columns[i]])
    plt.ylabel(df_sample.columns[i])
    plt.xlabel('Total_metering')
    plt.tight_layout()


# In[268]:


#Year wise Total_metering
plt.figure(figsize=(15,5))
plt.suptitle("Year wise Total metering",fontsize=20,y=1)
df_sample.groupby(df_sample.Year)['Total_metering'].sum().plot(kind='bar',xlabel='Year',ylabel='Reading in watt-hour')


# In[269]:


#Year wse total power consumption
plt.figure(figsize=(15,8))
plt.suptitle('Yearly-total watt-hour Power consumption',fontsize=20)
df_sample.groupby(df_sample.Year)['Global_active_power'].sum().plot(kind='bar',xlabel='Year',ylabel='Readings in watt-hour')
plt.show()


# In[270]:


#Year wse total power consumption
plt.figure(figsize=(15,8))
plt.suptitle('Monthly Voltage',fontsize=20,y=1)
df_sample.groupby(df_sample.Month)['Voltage'].sum().plot(kind='bar',xlabel='Month',ylabel='Readings in watt-hour')
plt.show()


# In[271]:


plt.figure(figsize=(9,9))
year_labels= [2006,2007,2008,2009,2010]
plt.pie(x=df_sample.groupby(df_sample['Year'])['Total_metering'].sum(),autopct='%1.2f%%',labels=year_labels)
plt.title('Yearly percentage wise distribution of Total_metering',fontsize=20)
plt.show()


# In[272]:


plt.figure(figsize=(9,9))
year_labels= [2006,2007,2008,2009,2010]
plt.pie(x=df_sample.groupby(df_sample['Year'])['Global_reactive_power'].sum(),autopct='%1.2f%%',labels=year_labels)
plt.title('Percentage wise distribution of Global_reactive_power',fontsize=20)
plt.show()


# In[273]:


plt.figure(figsize=(9,9))
year_labels= [2006,2007,2008,2009,2010]
plt.pie(x=df_sample.groupby(df_sample['Year'])['Global_active_power'].sum(),autopct='%1.2f%%',labels=year_labels)
plt.title('Yearly percentage wise distribution of Global_active_power',fontsize=20)
plt.show()


# In[274]:


plt.figure(figsize=(9,9))
year_labels= [2006,2007,2008,2009,2010]
plt.pie(x=df_sample.groupby(df_sample['Year'])['Global_intensity'].sum(),autopct='%1.2f%%',labels=year_labels)
plt.title('Yearly percentage wise distribution of Global_intensity',fontsize=20)
plt.show()


# In[275]:


plt.figure(figsize=(9,9))
year_labels= [2006,2007,2008,2009,2010]
plt.pie(x=df_sample.groupby(df_sample['Year'])['Global_active_power'].sum(),autopct='%1.2f%%',labels=year_labels)
plt.title('Yearly percentage wise distribution of Global_active_power',fontsize=20)
plt.show()


# In[276]:


plt.figure(figsize=(9,9))
year_labels= [2006,2007,2008,2009,2010]
plt.pie(x=df_sample.groupby(df_sample['Year'])['Voltage'].sum(),autopct='%1.2f%%',labels=year_labels)
plt.title('Yearly percentage wise distribution of Voltage',fontsize=20)
plt.show()


# In[277]:


plt.figure(figsize=(15,7))
sns.lineplot(x='Hour',y='Total_metering',data=df_sample,color='blue')
list = np.arange(0,26,2)
plt.xticks(list)
plt.show()


# Here it can be seen that in Hourly vs Total_metering the total metering increase after 6 and it is less during 0 to 5 hrs.

# In[278]:


plt.figure(figsize=(15,7))
sns.lineplot(x='Month',y='Total_metering',data=df_sample,color='blue')
list = np.arange(0,12,1)
plt.xticks(list)
plt.show()


# In[279]:


plt.figure(figsize=(15,15))
sns.pairplot(df_sample)
plt.show()


# 𝐂𝐡𝐞𝐜𝐤𝐢𝐧𝐠 𝐜𝐨𝐫𝐫𝐞𝐥𝐚𝐭𝐢𝐨𝐧

# In[280]:


plt.figure(figsize=(15,8))
sns.heatmap(data=df_sample.corr(),annot=True)
plt.show()


# 𝐂𝐡𝐞𝐜𝐤𝐢𝐧𝐠 𝐟𝐨𝐫 𝐨𝐮𝐭𝐥𝐢𝐞𝐫𝐬

# In[281]:


plt.figure(figsize=(15,12))
plt.suptitle("Outliers Analysis",fontsize=20,y=1)
for i in range(0,len(df_sample.columns)):
    plt.subplot(5,3,i+1)
    sns.boxplot(df_sample[df_sample.columns[i]])
    plt.tight_layout()


# 𝐒𝐚𝐯𝐢𝐧𝐠 𝐭𝐡𝐢𝐬 𝐜𝐥𝐞𝐚𝐧𝐞𝐝 𝐝𝐚𝐭𝐚

# In[282]:


df_sample.to_csv('cleaned_power_cosumption_data_1.csv')


# In[283]:


df_new=pd.DataFrame()


# In[284]:


df_new=df_sample


# In[285]:


df_new.head()


# In[286]:


df_new.shape


# # 6. Data Preprocessing

# Spiltting the input and output feature

# In[287]:


X=df_new.drop(['Day','Month','Year','Hour','Minutes','Total_metering'],axis=1)
Y=df_new['Total_metering']


# In[288]:


X.head()


# In[289]:


Y.head()


# ###### Splitting the dataset into train test data

# In[290]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.33)


# In[33]:


X_train.shape,X_test.shape,Y_train.shape,Y_test.shape


# In[292]:


X_train.head()


# In[293]:


X_test.head()


# In[294]:


Y_train.head()


# In[295]:


Y_test.head()


# ###### Feature Scaling

# In[296]:


from sklearn.preprocessing import MinMaxScaler
Scaler=MinMaxScaler(feature_range=(0,1))
X_train_scaled=Scaler.fit_transform(X_train)
X_test_scaled=Scaler.transform(X_test)


# In[297]:


X_train_scaled


# In[298]:


X_test_scaled


# ###### Variance Inflation Factor

# In[299]:


from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[300]:


vif= pd.DataFrame()
vif['vif']=[variance_inflation_factor(X_train_scaled,i) for i in range(X_train_scaled.shape[1])]
vif['Features']=X_train.columns
vif


# ### Model Building

# ### 1. Linear Regression

# In[301]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score


# In[302]:


Lreg=LinearRegression()
Lreg=Lreg.fit(X_train_scaled,Y_train)


# In[303]:


#prediction
Lreg_pred=Lreg.predict(X_test_scaled)


# In[304]:


#Performance metrics
MAE=mean_absolute_error(Y_test,Lreg_pred)
MSE=mean_squared_error(Y_test,Lreg_pred)
RMSE=np.sqrt(MSE)
r2=r2_score(Y_test,Lreg_pred)
training_accuracy=Lreg.score(X_train_scaled,Y_train)
testing_accuracy=Lreg.score(X_test_scaled,Y_test)


# In[305]:


print("𝐑𝐞𝐬𝐮𝐥𝐭𝐬 𝐟𝐨𝐫 𝐋𝐢𝐧𝐞𝐚𝐫 𝐑𝐞𝐠𝐫𝐞𝐬𝐬𝐢𝐨𝐧 :- ")
print("The linear regression coefficients are ",Lreg.coef_)
print("Training Accuracy : {:.5f}".format(training_accuracy))
print("Testing Accuracy : {:.5f}".format(testing_accuracy))
print("MAE value: {:.4f}".format(MAE))
print("MSE value: {:.4f}".format(MSE))
print("RMSE value: {:.4f}".format(RMSE))
print("R2 score value:",r2)


# ### 2. Support Vector Regressor

# In[306]:


from sklearn.svm import SVR


# In[307]:


svr=SVR(kernel='linear')
svr.fit(X_train_scaled,Y_train)


# In[308]:


#Predicting the test data
svr_pred=svr.predict(X_test_scaled)


# In[309]:


#Performance metrics
MAE=mean_absolute_error(Y_test,svr_pred)
MSE=mean_squared_error(Y_test,svr_pred)
RMSE=np.sqrt(MSE)
r2=r2_score(Y_test,svr_pred)
training_accuracy=svr.score(X_train_scaled,Y_train)
testing_accuracy=svr.score(X_test_scaled,Y_test)


# In[310]:


print("𝐑𝐞𝐬𝐮𝐥𝐭𝐬 𝐟𝐨𝐫 𝐒𝐮𝐩𝐩𝐨𝐫𝐭 𝐕𝐞𝐜𝐭𝐨𝐫 𝐑𝐞𝐠𝐫𝐞𝐬𝐬𝐨𝐫 :- ")
print("Training Accuracy : {:.5f}".format(accuracy))
print("Testing Accuracy : {:.5f}".format(accuracy))
print("MAE value: {:.4f}".format(MAE))
print("MSE value: {:.4f}".format(MSE))
print("RMSE value: {:.4f}".format(RMSE))
print("R2 score value:",r2)


# ### 3. Decision Tree Regressor

# In[311]:


from sklearn.tree import DecisionTreeRegressor


# In[312]:


dtr=DecisionTreeRegressor()
dtr.fit(X_train_scaled,Y_train)


# In[314]:


#predicting the test data using the decision tree regressor
dtr_pred=dtr.predict(X_test_scaled)


# In[316]:


#performance metrics
MAE=mean_absolute_error(Y_test,dtr_pred)
MSE=mean_squared_error(Y_test,dtr_pred)
RMSE=np.sqrt(MSE)
r2=r2_score(Y_test,dtr_pred)
training_accuracy=dtr.score(X_train_scaled,Y_train)
testing_accuracy=dtr.score(X_test_scaled,Y_test)


# In[317]:


print("𝐑𝐞𝐬𝐮𝐥𝐭𝐬 𝐟𝐨𝐫 𝐃𝐞𝐜𝐢𝐬𝐢𝐨𝐧 𝐓𝐫𝐞𝐞 𝐑𝐞𝐠𝐫𝐞𝐬𝐬𝐨𝐫 :- ")
print("Training Accuracy :{:.5f}".format(training_accuracy))
print("Testing Accuracy : {:.5f}".format(testing_accuracy))
print("MAE value: {:.4f}".format(MAE))
print("MSE value: {:.4f}".format(MSE))
print("RMSE value: {:.4f}".format(RMSE))
print("R2 score value:",r2)


# Here it is observed that the Training accuracy is 99.98% and testing accuracy is 55.7%. This is the condition of overfitting so I'm using Random forest to prevent overfitting.

# ### 4. Random Forest Regressor

# In[323]:


from sklearn.ensemble import RandomForestRegressor


# In[325]:


rfr=RandomForestRegressor()
rfr.fit(X_train_scaled,Y_train)


# In[327]:


#predicting the test data
rfr_pred=rfr.predict(X_test_scaled)


# In[328]:


#performance metrics
MAE= mean_absolute_error(Y_test,rfr_pred)
MSE=mean_squared_error(Y_test,rfr_pred)
RMSE=np.sqrt(MSE)
r2=r2_score(Y_test,rfr_pred)
training_accuracy=rfr.score(X_train_scaled,Y_train)
testing_accuracy=rfr.score(X_test_scaled,Y_test)


# In[329]:


print("𝐑𝐞𝐬𝐮𝐥𝐭𝐬 𝐟𝐨𝐫 𝐑𝐚𝐧𝐝𝐨𝐦 𝐅𝐨𝐫𝐞𝐬𝐭 𝐑𝐞𝐠𝐫𝐞𝐬𝐬𝐨𝐫 :- ")
print("Training Accuracy :{:.5f}".format(training_accuracy))
print("Testing Accuracy : {:.5f}".format(testing_accuracy))
print("MAE value: {:.4f}".format(MAE))
print("MSE value: {:.4f}".format(MSE))
print("RMSE value: {:.4f}".format(RMSE))
print("R2 score value:",r2)


# In[ ]:




