#!/usr/bin/env python
# coding: utf-8

# # WorldWideProducts - Project 5 - Data Science
# 
# ### Project Description
# #### The project is to evaluate the trend of a demand of the product and forecast its demand trend 
# #### for the next 1 year.
# 
# ### Dataset source
# #### https://www.kaggle.com/felixzhao/productdemandforecasting

# #### import required libraries

# In[1]:


from pandas import DataFrame, read_csv
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


# #### Load required product data

# In[2]:


productData=matchesRaw = pd.read_csv('/World Wide Products - Project 5/data/processed/Historical Product Demand.csv')


# In[3]:


productData.head(10)


# ### Data - Data processing Feature Engineering
# 
# #### 1. Bucket days to months and separate out years.
# #### 2. Drop Date column
# #### 3. One hot encoding for product_code,Product_Category,Warehouse to uniquely identify by integers.
# #### 4. OrderDemand seems to be a string and has negative values mentioned by value in (), 
# ####    converting all the values to integers and stringreplace to replace() with a '-'
# #### 5. Finally dropping unnecessary columns.

# In[4]:


productData['dateformat'] = pd.to_datetime(productData['Date'])


# In[5]:


productData['year'], productData['month'] = productData['dateformat'].dt.year, productData['dateformat'].dt.month


# In[6]:


from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
productData['ProdID']=encoder.fit_transform(productData['Product_Code'].astype('str'))
productData['CategoryID']=encoder.fit_transform(productData['Product_Category'].astype('str'))
productData['WarehouseID']=encoder.fit_transform(productData['Warehouse'].astype('str'))


# In[7]:


processedProductData=productData.drop(['Date','Product_Category','Product_Code','Warehouse'],axis=1)


# In[8]:


processedProductData['Order_Demand'] = processedProductData['Order_Demand'].str.strip()
processedProductData['Order_Demand']= processedProductData['Order_Demand'].str.replace("(","-")
processedProductData['Order_Demand']= processedProductData['Order_Demand'].str.replace(")","")
processedProductData['Order_Demand'] = processedProductData['Order_Demand'].astype(int)


# In[9]:


processedProductData.head(5)


# ### Data-Exploration 
# 
# #### Trying to identify the Product with Highest demand and perform forecast on that product.
# #### 1.GroupBy ProductId and Perform sum operation on Order_Demand, followed by a sort_values

# In[10]:


GroupedByOrderDemand=processedProductData.groupby(['ProdID'])['Order_Demand'].sum().reset_index()


# In[11]:


TOP5ProductsInDemand = GroupedByOrderDemand.sort_values('Order_Demand', ascending=False).head(5)
TOP5ProductsInDemand


# In[12]:


plt.figure(figsize=(10,10))
import seaborn as sns
ax = sns.barplot(x='ProdID',y='Order_Demand',data=TOP5ProductsInDemand)
ax.set(xlabel='ProdID', ylabel='Order_Demand')
plt.title("TOP 5 products in demand (Demand in multiples of 100 million)")
plt.show()


# #### Identified the top on demand product.. exploring other details for the top on demand  product...-1348

# In[13]:


ExploringDataForHighestDemandProduct = processedProductData.loc[(processedProductData['ProdID']==1348)]
ExploringDataForHighestDemandProduct


# #### Analysis of Trend of the demand by Year

# In[14]:


TOPProdDemandByYear=ExploringDataForHighestDemandProduct.groupby(['year'])['Order_Demand'].sum().reset_index()


# In[15]:


TOPProdDemandByYear


# In[16]:


plt.figure(figsize=(10,10))
import seaborn as sns
ax = sns.barplot(x='year',y='Order_Demand',data=TOPProdDemandByYear)
ax.set(xlabel='Year', ylabel='Order_Demand')
plt.title("Demand analysis for the product 1348 by Year, that has highest demand (Demand in multiples of 100 million)")
plt.show()


# #### Analysis of Trend of the demand by Month

# In[17]:


TOPProdDemandByMonth=ExploringDataForHighestDemandProduct.groupby(['month'])['Order_Demand'].sum().reset_index()
TOPProdDemandByMonth


# In[18]:


plt.figure(figsize=(10,10))
import seaborn as sns
ax = sns.barplot(x='month',y='Order_Demand',data=TOPProdDemandByMonth)
ax.set(xlabel='Month', ylabel='Order_Demand')
plt.title("Demand analysis for the product 1348 by Month, that has highest demand (Demand in multiples of 100 million)")
plt.show()


# #### Diving into forecasting analysis for the identified product that has the highest demand - ID-1348

# ### More Data processing...
# #### 1. Removing other unnecessary columns 
# #### 2. FBprophet needs Columns to be ds and y, so renaming them.

# In[19]:


FutureDemandAnalysisForHighestDemandProduct=ExploringDataForHighestDemandProduct.drop(['year','month','ProdID','WarehouseID','CategoryID'],axis=1)


# ### FB Prophet forecasting analysis

# In[20]:


from fbprophet import Prophet
obj = Prophet(daily_seasonality=True)
FutureDemandAnalysisForHighestDemandProduct=FutureDemandAnalysisForHighestDemandProduct.reset_index()
FutureDemandAnalysisForHighestDemandProduct = FutureDemandAnalysisForHighestDemandProduct.rename(columns = {'dateformat': 'ds', 'Order_Demand': 'y' })
obj.fit(FutureDemandAnalysisForHighestDemandProduct)


# ### Next 1 year(365 days) forecast

# In[21]:


future = obj.make_future_dataframe(periods=365)


# In[22]:


import datetime
forecast = obj.predict(future)


# ### Filtering the forecast for next 1 year after 2017 and Plot all related components.

# In[23]:


mask = (forecast['ds'] > '02-01-2017')
forecast=forecast.loc[mask]
figure1 = obj.plot(forecast)


# In[24]:


figure2 = obj.plot_components(forecast)


# ### Observations on Visualizations and Reports

# #### 1. We can see from the forecast that there are high chances for Product 1358, to remain the product with highest demand for next one year too.
# 
# #### 2. Based on the demand graph consistently has risen from 2017 to 2018 with some possibility in the deviations, 
# #### that can go far better than what its previous demand is, on the best case.
# #### On the worst case, it shows a deviation that can fall just below the demand of what it is before.
# 
# #### 3. The weekly forecast suggests that Sunday would be the prime day for the product, and Saturday being sloppy on the demand perspective.
# 
# #### 4. The yearly forecast suggests that January end to February mid is the prime period for the product's business, 
# #### and that February end can see a dip in the same.
