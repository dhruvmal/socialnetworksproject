#!/usr/bin/env python
# coding: utf-8

# In[7]:


# import relevant packages
from datascience import *
import numpy as np
import pandas as pd


import statsmodels.formula.api as smf
import matplotlib as mp
from dateutil.relativedelta import relativedelta
get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt


# # Consumer Defensive Sector

# In[8]:


data_cons_def = pd.read_csv('ConsumerDefensive.csv')


# ### Consumer Defensive Scatter Plot

# In[9]:


#rangeplot = range(-16,12)
x = np.random.randint(-16,12, size = (100, 100))
preds = 2.3990 + (0.2455*x)


# In[10]:


plt.scatter(data_cons_def['Net_Twitter_Sentiment'], data_cons_def['Surprise_Percent'])
plt.xlabel("Net Twitter Sentiment")
plt.ylabel("Surprise %")
plt.title("Net Twitter Sentiment Vs. Surprise % for Consumer Defensive Sector")
plt.plot(x,preds, color = 'red')


# In[11]:


data_cons_def.corr()


# ### Consumer Defensive Correlation Coefficient

# Consumer Defense Sector Correlation Coefficient: 0.115165
# 
# This means that there's a very weak positive correlation between Net Twitter Sentiment and Surprise % in the Consumer Defense Sector

# ### Consumer Defensive Ordinary Least Squares Analysis

# In[12]:


# Generate regression data
results_cons_def = smf.ols('Surprise_Percent ~ Net_Twitter_Sentiment', data=data_cons_def).fit()

# Show summary of regression data
results_cons_def.summary()


# The coefficient is 0.2455, meaning that as the Net Twitter Sentiment increases by 1, the Surpise % increases by 0.2455% in the Consumer Defensive Sector.
# 
# The P value is greater than 0.05, meaning that we reject the alternative hypothesis, that Net Twitter Sentiment has an effect on the Surprise % in the agricultural sector. We accept the Null hypothesis, that Net Twitter Sentiment has an effect on Surprise % in the Consumer Defensive Sector.
# 
# The R^2 value is 0.013, meaning that only 1.3% of the variance in the Surprise % can be explained by the Net Twitter Sentiment in the Consumer Defensive Sector.

# # Consumer Cyclical Sector

# In[13]:


data_cons_cyc = pd.read_csv('ConsumerCyclical.csv')


# ### Consumer Cyclical Scatter Plot

# In[14]:


x = np.random.randint(-70,30, size = (100, 100))
preds = 5.0685 + (0.1093*x)


# In[15]:


plt.scatter(data_cons_cyc['Net_Twitter_Sentiment'], data_cons_cyc['Surprise_Percent'])
plt.xlabel("Net Twitter Sentiment")
plt.ylabel("Surprise %")
plt.title("Net Twitter Sentiment Vs. Surprise % for Consumer Cyclical Sector")
plt.plot(x,preds, color = 'red')


# ### Consumer Cyclical Sector Correlation Coefficient

# In[16]:


data_cons_cyc.corr()


# Consumer Cyclical Sector Correlation Coefficient: 0.25110
# 
# This means theres weak positive correlation between nnet twitter sentiment and surprise % in the Consumer Cyclical sector

# ### Consumer Cyclical Ordinary Least Squares Analysis

# In[17]:


results_cons_cyc = smf.ols('Surprise_Percent ~ Net_Twitter_Sentiment', data=data_cons_cyc).fit()

# Show summary of regression data
results_cons_cyc.summary()


# The coefficient is 0.1093, meaning that as the Net Twitter Sentiment increases by 1, the Surpise % increases by 0.1093% in the Consumer Cyclical Sector.
# 
# The P value is greater than 0.05, meaning that we reject the alternative hypothesis, that Net Twitter Sentiment has an effect on the Surprise % in the Consumer Cyclical Sector. We accpet the Null hypothesis, that Net Twitter Sentiment has an effect on Surprise % in the Consumer Cyclical Sector.
# 
# The R^2 value is 0.063, meaning that 6.3% of the variance in the Surprise % can be explained by the Net Twitter Sentiment in the Consumer Cyclical Sector.

# # Industrial Sector
# 

# In[18]:


data_industrial = pd.read_csv('Industrial.csv')


# ### Industrial Scatter Plot

# In[19]:


x = np.random.randint(-20,30, size = (100, 100))
preds = 0.0427 + (0.5561*x)


# In[20]:


plt.scatter(data_industrial['Net_Twitter_Sentiment'], data_industrial['Surprise_Percent'])
plt.xlabel("Net Twitter Sentiment")
plt.ylabel("Surprise %")
plt.title("Net Twitter Sentiment Vs. Surprise % for Industrial Sector")
plt.plot(x,preds, color = 'red')


# ### Industrial Sector Correlation Coefficient

# In[21]:


data_industrial.corr()


# Industrial Sector Correlationn Coefficient: 0.424807
# 
# This means that there is a medium positive relationship between Net twitter sentiment and Surprise % in the Industrial Sector

# ### Industrial Ordinary Least Squares Analysis

# In[22]:


results_industrial = smf.ols('Surprise_Percent ~ Net_Twitter_Sentiment', data=data_industrial).fit()

# Show summary of regression data
results_industrial.summary()


# The coefficient is 0.5561, meaning that as the Net Twitter Sentiment increases by 1, the Surpise % increases by 0.5561% in the Industrial Sector.
# 
# The P value is less than 0.05, meaning that we accept the alternative hypothesis, that Net Twitter Sentiment has an effect on the Surprise % in the Industrial Sector. We reject the Null hypothesis, that Net Twitter Sentiment has an effect on Surprise % in the Industrial Sector.
# 
# The R^2 value is 0.180, meaning that 1.8% of the variance in the Surprise % can be explained by the Net Twitter Sentiment in the Industrial Sector.

# # Commodities Sector

# In[23]:


data_commodities = pd.read_csv('Commodities.csv')


# ### Commodities Scatter Plot

# In[24]:


x = np.random.randint(-45,15, size = (100, 100))
preds = -2.0001 + (-0.0572*x)


# In[25]:


plt.scatter(data_commodities['Net_Twitter_Sentiment'], data_commodities['Surprise_Percent'])
plt.xlabel("Net Twitter Sentiment")
plt.ylabel("Surprise %")
plt.title("Net Twitter Sentiment Vs. Surprise % for Commodities Sector")
plt.plot(x,preds, color = 'red')


# ### Commodities Correlation Coefficient

# In[26]:


data_commodities.corr()


# Commodities Correlation Coefficient: -0.046301
# 
# This means that there a very weak negative correlation between Net twitter sentiment and surprise % in the Commodities sector

# ### Commodities Ordinary Least Squares Analysis

# In[27]:


results_commodities = smf.ols('Surprise_Percent ~ Net_Twitter_Sentiment', data=data_commodities).fit()

# Show summary of regression data
results_commodities.summary()


# The coefficient is -0.0572, meaning that as the Net Twitter Sentiment increases by 1, the Surpise % decreases by 0.0572% in the Commodities Sector.
# 
# The P value is greater than 0.05, meaning that we reject the alternative hypothesis, that Net Twitter Sentiment has an effect on the Surprise % in the Commodities Sector. We accept the Null hypothesis, that Net Twitter Sentiment has an effect on Surprise % in the Commodities Sector.
# 
# The R^2 value is 0.002, meaning that 0.2% of the variance in the Surprise % can be explained by the Net Twitter Sentiment in the Commodities Sector.

# # Healthcare Sector

# In[28]:


data_healthcare = pd.read_csv('Healthcare.csv')


# ### Healthcare Scatterplot

# In[29]:


x = np.random.randint(-30,25, size = (100, 100))
preds = 1.3512 + (-0.1756*x)


# In[30]:


plt.scatter(data_healthcare['Net_Twitter_Sentiment'], data_healthcare['Surprise_Percent'])
plt.xlabel("Net Twitter Sentiment")
plt.ylabel("Surprise %")
plt.title("Net Twitter Sentiment Vs. Surprise % for Healthcare Sector")
plt.plot(x,preds, color = 'red')


# ### Healthcare Correlation Coefficient

# In[31]:


data_healthcare.corr()


# Healthcare Correlation Coefficient: -0.126491
# 
# This means that there's a weak negative correlation between Net twitter sentiment and Surprise % in the Healthcare sector

# ### Healthcare Ordinary Least Squares Analysis

# In[32]:


results_healthcare = smf.ols('Surprise_Percent ~ Net_Twitter_Sentiment', data=data_healthcare).fit()

# Show summary of regression data
results_healthcare.summary()


# The coefficient is -0.1756, meaning that as the Net Twitter Sentiment increases by 1, the Surpise % decreases by 0.1756 in the Healthcare Sector.
# 
# The P value is greater than 0.05, meaning that we reject the alternative hypothesis, that Net Twitter Sentiment has an effect on the Surprise % in the Healthcare Sector. We accept the Null hypothesis, that Net Twitter Sentiment has an effect on Surprise % in the Healthcare Sector.
# 
# The R^2 value is 0.016, meaning that 1.6% of the variance in the Surprise % can be explained by the Net Twitter Sentiment in the Healthcare Sector.
