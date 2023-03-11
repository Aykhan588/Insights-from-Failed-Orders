#!/usr/bin/env python
# coding: utf-8

# Gett, previously known as GetTaxi, is an Israeli-developed technology platform solely focused on corporate Ground Transportation Management (GTM). They have an application where clients can order taxis, and drivers can accept their rides (offers). At the moment, when the client clicks the Order button in the application, the matching system searches for the most relevant drivers and offers them the order. In this task, I would like to investigate some matching metrics for orders that did not completed successfully, i.e., the customer didn't end up getting a car.

# # Assignment
# Please complete the following tasks.
# 
# 1.Build up distribution of orders according to reasons for failure: cancellations before and after driver assignment, and reasons for order rejection. Analyse the resulting plot. Which category has the highest number of orders?
# 
# 2.Plot the distribution of failed orders by hours. Is there a trend that certain hours have an abnormally high proportion of one category or another? What hours are the biggest fails? How can this be explained?
# 
# 3.Plot the average time to cancellation with and without driver, by the hour. If there are any outliers in the data, it would be better to remove them. Can we draw any conclusions from this plot?
# 
# 4.Plot the distribution of average ETA by hours. How can this plot be explained?

# # Data Description
# 
# We have two data sets: data_orders and data_offers, both being stored in a CSV format. The data_orders data set contains the following columns:
# 
# order_datetime - time of the order
# 
# origin_longitude - longitude of the order
# 
# origin_latitude - latitude of the order
# 
# m_order_eta - time before order arrival
# 
# order_gk - order number
# 
# order_status_key - status, an enumeration consisting of the following mapping:
# 4 - cancelled by client,
# 9 - cancelled by system, i.e., a reject
# 
# is_driver_assigned_key - whether a driver has been assigned
# 
# cancellation_time_in_seconds - how many seconds passed before cancellation
# 
# 
# 
# The data_offers data set is a simple map with 2 columns:
# 
# order_gk - order number, associated with the same column from the orders data set
# offer_id - ID of an offer

# # Read and Explore the Datasets

# In[1]:


import pandas as pd


# In[2]:


orders = pd.read_csv(filepath_or_buffer="data_orders.csv")


# In[3]:


# rows, columns
orders.shape


# In[4]:


# the random_state argument ensures that we get the same sample
# each time we call the method with the same arguments
orders.sample(n=10, random_state=42)


# In[5]:


offers = pd.read_csv(filepath_or_buffer="data_offers.csv")


# In[6]:


offers.shape


# In[7]:


offers.sample(n=10, random_state=42)


# Next, we would like to merge the two DataFrames into one, for easier manipulation. Pandas has the merge() method for doing exactly that. It is similar to joining tables in SQL. We specify how we want the merge to be carried out (inner) and on which column should it occur (order_gk). Then, we print a sample of the merged DataFrame

# In[8]:


df = orders.merge(right=offers, how="inner", on="order_gk")
#order_gk is the common column which is used to join them


# In[9]:


df.sample(n=10, random_state=42)


# We may be able to do something to improve the data quality a bit. For example, the values in order_status_key and is_driver_assigned_key are not informative of the contents but are rather just keys that point to some internal description. We could replace their values with more informative information, like replacing the 1s in is_driver_assigned_key with the string Yes and the 0s with the string No. The same can be done for the order_status_key column. Moreover, the names of the columns sound a bit technical, but we can modify them.

# In[10]:


import numpy as np

df["is_driver_assigned"] = np.where(df["is_driver_assigned_key"] == 1, "Yes", "No")
df["order_status"] = np.where(df["order_status_key"] == 4, "Client Cancelled", "System Reject")

df.drop(columns=["is_driver_assigned_key", "order_status_key"], inplace=True)


# In[11]:


df = df.rename(columns={
    "order_datetime": "order_time"
})


# In[12]:


df.sample(n=10, random_state=42)


# At this point we are ready to start answering the questions, so let's go!
# 
# 

# # Question 1
# Build up a distribution of orders according to reasons for failure: cancellations before and after driver assignment (YES OR NO), and reasons for order rejection (CLIENT CANCELLED OR SYSTEM REJECT). Analyse the resulting plot. Which category has the highest number of orders?
# 
# One, straightforward solution to solve this question is to use the groupby method to group the DataFrame by the is_driver_assigned and order_status columns, and then count the rows in each group, i.e., each combination of values for the grouping columns.
# 
# Since both are binary variables (have only two possible values), there are four possible combinations. The cell below prints the output of the proposed approach.

# In[13]:


df.groupby(by=["is_driver_assigned", "order_status"])["order_gk"].count()


# We observe a high number of orders cancelled before a driver is assigned, implying that maybe customers have waited too long and have decided on an alternative for their transportation needs. We have 13435 orders cancelled by the client, and 9469 rejected by the system. There are 8360 client cancellations after a driver has been assigned, and only four which were rejected by the system, for some reason

# Another interesting approach to this question is to use pivot tables (for more information, see the Pandas method and this Wikipedia article).

# In[14]:


df_q1 = df.pivot_table(columns=["is_driver_assigned", "order_status"], values="order_gk", aggfunc="count")
_ = df_q1.plot(kind="bar", subplots=False, figsize=(7, 7), legend=True, rot=0)


# the structure of the pivot table is very similar to the group-by dataframe

# In[15]:


df_q1


# # Question 2
# Plot the distribution of failed orders by hours. Is there a trend that certain hours have an abnormally high proportion of one category or another? What hours are the biggest fails? How can this be explained?
# 
# This question builds up upon the previous one by delving deeper into the analysis of failed orders. Rather than just plotting the distribution of fails by category (reason for cancellation, and the driver assignment), we want to know when these fails occur, and if there is some specific period in the day when one category prevails over others.

# In[16]:


# extract hour from the time column
df["order_hour"] = df["order_time"].str.split(":").apply(lambda split: split[0])


# In[18]:


# print a small sample to make sure that the transformation is correct
df.sample(n=5, random_state=42)
#new column is created


# In[19]:


_ = df.groupby(by="order_hour")["order_gk"].count().plot(figsize=(10, 7),
                                                         legend=True,
                                                         xticks=range(0, 24),
                                                         title="Count of Failed Orders by Hour of Day")


# As a first step towards a DataFrame that contains such aggregated information we group-by by the order hour, the driver-assigned flag, and the order status, and then count the number of order_gk.

# In[20]:


grouped_q2 = df.groupby(by=["order_hour", "is_driver_assigned", "order_status"])["order_gk"].count()
grouped_q2


# In[21]:


_ = grouped_q2.reset_index().pivot(index="order_hour",
                                   columns=["is_driver_assigned", "order_status"],
                                   values="order_gk").plot(xticks=range(0, 24),
                                                           figsize=(13, 7),
                                                           title="Count of Failed Orders Per Hour and Category")


# The four system rejects with the assigned driver occurred around midnight (see the output of the grouped DataFrame). The rest of the categories follow similar distribution, with the client cancellations with the assigned driver having a substantially lower count during the night hours.

# # Question 3
# Plot the average time to cancellation with and without driver, by hour. Can we draw any conclusions from this plot?
# 
# To solve this question we are going to take the same approach as the previous question.
# 
# First, we will aggregate the DataFrame by the order hour and the driver-assignment flag. Then, we will aggregate the cancellations_time_in_seconds column with the function mean. This will give us the required, aggregated information.

# In[22]:


grouped_q3 = df.groupby(by=["order_hour", "is_driver_assigned"])["cancellations_time_in_seconds"].mean()
grouped_q3


# In[23]:


_ = grouped_q3.reset_index().pivot(index="order_hour",
                                   columns="is_driver_assigned",
                                   values="cancellations_time_in_seconds").plot(xticks=range(0, 24),
                                                                                figsize=(13, 7),
                                                                                title="Average Time to Cancellation Per Hour and Driver Assignment")


# The average time to cancellation is higher on orders with an assigned driver than without, for each hour without exception. The peak occurs at 3 A.M. At this time there are a lot of client cancellations, so a logical explanation would be that clients have waited too long for the driver.

# # Question 4
# Plot the distribution of average ETA by hours. How can this plot be explained?
# 
# The solution to this question is quite straightforward. We simply group-by the DataFrame on the order hour, and aggregate the m_order_eta column with a mean function. Then, we plot the resulting DataFrame. The result is in the cell immediately below.

# In[24]:


_ = df.groupby(by="order_hour")["m_order_eta"].mean().plot(figsize=(14, 7),
                                                           xticks=range(0, 24),
                                                           title="Average ETA per hour")


# The line very closely matches the count of failed orders per hour, indicating that the number of failed orders increases as the average waiting time of the client increases.

# In[ ]:




