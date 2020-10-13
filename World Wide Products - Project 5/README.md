# WorldWideProducts - Project 5 - Forecasting

#Project Description
The project tries to evaluate the trend of the demand of a product 
and forecast its demand trend for the next 1 year.

# Input data and Output Information

This project takes raw data related to products and its demand, 
loads the data into data frames and then performs some forecasting 
on its demand for the next 1 year.

# Source of Raw/Processed Data

Raw data has been taken from : https://www.kaggle.com/felixzhao/productdemandforecasting

# Data Processing and Feature Engineering

1. Bucket days to months and separate out years.
2. Drop Date column
3. One hot encoding for product_code,Product_Category,Warehouse to uniquely identify by integers.
4. OrderDemand is a string and has negative values mentioned by value in (), 
   converting all the values to integers and stringreplace to replace() with a '-'
5. Finally dropping unnecessary columns.

# Data Exploration and reports
1. Trying to identify the Product with Highest demand and perform forecast on that product.
   GroupBy ProductId and Perform sum operation on Order_Demand, followed by a sort_values

From Step 1 we identify the Top most On demand product - in our case 1358.
2. Analysis of Trend of the demand of Product 1358 by Year
3. Analysis of Trend of the demand of Product 1358 by Month

Steps 2 and 3 use GroupBy on Year and Month respectively on data that is specific to product 1358

Data exploration also generates few reports available in the Reports folder.


# Forecast Analysis.

I used Forecasting algorithm, FBProphet for period (365 days) 
I filtered the forecast next 1 year after 2017 and plotted all related components 
to visualize the forecast of the trend of the product 1358's demand.

# Plotting and exploration results

Forecast related reports and plottings are saved under reports.

# Conclusions and further improvements:
1. We can see from the forecast that there are high chances for Product 1358, to remain the product with highest demand for next one year too.

2. Based on the demand graph consistently has risen from 2017 to 2018 with some possibility in the deviations, 
   that can go far better than what its previous demand is, on the best case.
   On the worst case, it shows a deviation that can fall just below the demand of what it is before.

3. The weekly forecast suggests that Sunday would be the prime day for the product, and Saturday being sloppy on the demand perspective.

4. The yearly forecast suggests that January end to February mid is the prime period for the product's business, 
    and that February end can see a dip in the same.