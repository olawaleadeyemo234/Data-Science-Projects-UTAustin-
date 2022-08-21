#!/usr/bin/env python
# coding: utf-8

# In the 21st century, cars are an important mode of transportation that provides us the opportunity for personal control and autonomy. In day-to-day life, people use cars for commuting to work, shopping, visiting family and friends, etc. Research shows that more than 76% of people prevent themselves from traveling somewhere if they don't have a car. Most people tend to buy different types of cars based on their day-to-day necessities and preferences. So, it is essential for automobile companies to analyze the preference of their customers before launching a car model into the market. Austo, a UK-based automobile company aspires to grow its business into the US market after successfully establishing its footprints in the European market.
# 
# In order to be familiar with the types of cars preferred by the customers and factors influencing the car purchase behavior in the US market, Austo has contracted a consulting firm. Based on various market surveys, the consulting firm has created a dataset of 3 major types of cars that are extensively used across the US market. They have collected various details of the car owners which can be analyzed to understand the automobile market of the US.
# 
# Objective
# 
# Austo’s management team wants to understand the demand of the buyers and trends in the US market. They want to build customer profiles based on the analysis to identify new purchase opportunities so that they can manipulate the business strategy and production to meet certain demand levels. Further, the analysis will be a good way for management to understand the dynamics of a new market. Suppose you are a Data Scientist working at the consulting firm that has been contracted by Austo. You are given the task to create buyer’s profiles for different types of cars with the available data as well as a set of recommendations for Austo. Perform the data analysis to generate useful insights that will help the automobile company to grow its business.
# 
# Data Description
# 
# austo_automobile.csv: The dataset contains buyer’s data corresponding to different types of products(cars).
# 
# Data Dictionary
# 
# Age: Age of the customer
# Gender: Gender of the customer
# Profession: Indicates whether the customer is a salaried or business person
# Marital_status: Marital status of the customer
# Education: Refers to the highest level of education completed by the customer
# No_of_dependents: Number of dependents(partner/children/spouse) of the customer
# Personal_loan: Indicates whether the customer availed a personal loan or not
# House_loan: Indicates whether the customer availed house loan or not
# Partner_working: Indicates whether the customer's partner is working or not
# Salary: Annual Salary of the customer
# Partner_salary: Annual Salary of the customer's partner
# Total_salary: Annual household income (Salary + Partner_salary) of the customer's family
# Price: Price of the car
# Make: Car type (Hatchback/Sedan/SUV)
# 

# In[4]:


import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set (color_codes=True)
get_ipython().run_line_magic('matplotlib', 'inline')


import warnings
warnings.filterwarnings("ignore")


# In[5]:


data = pd.read_csv ("Downloads/download-3.csv")


# In[6]:


df = data.copy()


# In[7]:


df.head()


# In[8]:


df.head (50)

Observations

1.Age range for the data set varies. 
2.Profession in the data set is only salaried and business.
3.Some have partners that are unemployed.

# In[ ]:


data.shape


# Comment
# 
# The data has 1581 rows and 14 columns.

# In[10]:


data.dtypes. value_counts()


# In[11]:


data.info()


# In[ ]:


Observations

All column have 1,581 each.
The data contians intger and object data-type.


# In[12]:


df['Gender'] = df.Gender.astype('category')
df['Profession'] = df.Profession.astype('category')
df['Marital_status'] = df.Marital_status.astype('category')
df['Education'] = df.Education.astype ('category')
df['Personal_loan'] = df.Personal_loan.astype('category')
df['House_loan'] = df.House_loan.astype('category')
df['Partner_working'] = df.Partner_working.astype('category')
df['Make'] = df.Make.astype('category')

# Converting categorical column to categorical type 


# In[13]:


df.dtypes


# In[14]:


df.dtypes


# In[15]:


df.describe().T


# In[16]:


df.describe(exclude='number').T


# In[17]:


cat_col=['Gender', 'Profession', 'Marital_status', 'Education', 'Personal_loan', 'House_loan', 'Partner_working', 'Make']

for column in cat_col:
    print(df[column].value_counts())
    print('-'*50)

Observations

1.Marrried, Educated Male that does not own a house bought most of the cars.
2.Majority of the car bought was hatcback
3.The median Age, No of dependents, salary, partner-salary, total salary, and price is 29, 2, 59,000, 25,000,     78,000, 31,000 respectively.
4.Max age is 60 and minimum 22
5.Max total salary is 158, 000 and minimum 30,000
6.The average price for a car is between 31,000 - 35,000.
# In[18]:


df.isna().sum()


# No missing value in the data set

# # Let's check the count of each unique category in each of the categorical variables.

# In[19]:


def histogram_boxplot(data, feature, figsize=(12, 7), kde=False, bins=None):
    """
    Boxplot and histogram combined

    data: dataframe
    feature: dataframe column
    figsize: size of figure (default (12,7))
    kde: whether to show the density curve (default False)
    bins: number of bins for histogram (default None)
    """
    f2, (ax_box2, ax_hist2) = plt.subplots(
        nrows=2,  # Number of rows of the subplot grid= 2
        sharex=True,  # x-axis will be shared among all subplots
        gridspec_kw={"height_ratios": (0.25, 0.75)},
        figsize=figsize,
    )  # creating the 2 subplots
    sns.boxplot(
        data=data, x=feature, ax=ax_box2, showmeans=True, color="violet"
    )  # boxplot will be created and a star will indicate the mean value of the column
    sns.histplot(
        data=data, x=feature, kde=kde, ax=ax_hist2, bins=bins, palette="winter"
    ) if bins else sns.histplot(
        data=data, x=feature, kde=kde, ax=ax_hist2
    )  # For histogram
    ax_hist2.axvline(
        data[feature].mean(), color="green", linestyle="--"
    )  # Add mean to the histogram
    ax_hist2.axvline(
        data[feature].median(), color="black", linestyle="-"
    )  # Add median to the histogram


# In[20]:


def histogram_boxplot(data, feature, figsize=(12, 7), kde=False, bins=None):
   
    f2, (ax_box2, ax_hist2) = plt.subplots(
        nrows=2, 
        sharex=True,  
        gridspec_kw={"height_ratios": (0.25, 0.75)},
        figsize=figsize,
    ) 
    sns.boxplot(
        data=data, x=feature, ax=ax_box2, showmeans=True, color="violet"
    )  
    sns.histplot(
        data=data, x=feature, kde=kde, ax=ax_hist2, bins=bins, palette="winter"
    ) if bins else sns.histplot(
        data=data, x=feature, kde=kde, ax=ax_hist2
    )  
    ax_hist2.axvline(
        data[feature].mean(), color="green", linestyle="--"
    )  
    ax_hist2.axvline(
        data[feature].median(), color="black", linestyle="-"
    )  


# In[21]:


# Let's construct a function that shows the summary and density distribution of a numerical attribute:
def summary(x):
    x_min = data[x].min()
    x_max = data[x].max()
    Q1 = data[x].quantile(0.25)
    Q2 = data[x].quantile(0.50)
    Q3 = data[x].quantile(0.75)
    print(f'5 Point Summary of {x.capitalize()} Attribute:\n'
          f'{x.capitalize()}(min) : {x_min}\n'
          f'Q1                    : {Q1}\n'
          f'Q2(Median)            : {Q2}\n'
          f'Q3                    : {Q3}\n'
          f'{x.capitalize()}(max) : {x_max}')

    fig = plt.figure(figsize=(16, 10))
    plt.subplots_adjust(hspace = 0.6)
    sns.set_palette('pastel')
    
    plt.subplot(221)
    ax1 = sns.distplot(data[x], color = 'r')
    plt.title(f'{x.capitalize()} Density Distribution')
    
    plt.subplot(222)
    ax2 = sns.violinplot(x = data[x], palette = 'Accent', split = True)
    plt.title(f'{x.capitalize()} Violinplot')
    
    plt.subplot(223)
    ax2 = sns.boxplot(x=data[x], palette = 'cool', width=0.7, linewidth=0.6)
    plt.title(f'{x.capitalize()} Boxplot')
    
    plt.subplot(224)
    ax3 = sns.kdeplot(data[x], cumulative=True)
    plt.title(f'{x.capitalize()} Cumulative Density Distribution')
    
    plt.show()


# In[22]:


def box_plot(x = 'bmi'):
    def add_values(bp, ax):
        """ This actually adds the numbers to the various points of the boxplots"""
        for element in ['whiskers', 'medians', 'caps']:
            for line in bp[element]:
                # Get the position of the element. y is the label you want
                (x_l, y),(x_r, _) = line.get_xydata()
                # Make sure datapoints exist 
                # (I've been working with intervals, should not be problem for this case)
                if not np.isnan(y): 
                    x_line_center = x_l + (x_r - x_l)/2
                    y_line_center = y  # Since it's a line and it's horisontal
                    # overlay the value:  on the line, from center to right
                    ax.text(x_line_center, y_line_center, # Position
                            '%.2f' % y, # Value (3f = 3 decimal float)
                            verticalalignment='center', # Centered vertically with line 
                            fontsize=12, backgroundcolor="white")

    fig, axes = plt.subplots(1, figsize=(4, 8))

    red_diamond = dict(markerfacecolor='r', marker='D')

    bp_dict = data.boxplot(column = x, 
                             grid=True, 
                             figsize=(4, 8), 
                             ax=axes, 
                             vert = True, 
                             notch=False, 
                             widths = 0.7, 
                             showmeans = True, 
                             whis = 1.5,
                             flierprops = red_diamond,
                             boxprops= dict(linewidth=3.0, color='black'),
                             whiskerprops=dict(linewidth=3.0, color='black'),
                             return_type = 'dict')

    add_values(bp_dict, axes)

    plt.title(f'{x.capitalize()} Boxplot', fontsize=16)
    plt.ylabel(f'{x.capitalize()}', fontsize=14)
    plt.show()
    
    skew = data[x].skew()
    Q1 = data[x].quantile(0.25)
    Q3 = data[x].quantile(0.75)
    IQR = Q3 - Q1
    total_outlier_num = ((data[x] < (Q1 - 1.5 * IQR)) | (data[x] > (Q3 + 1.5 * IQR))).sum()
    print(f'Mean {x.capitalize()} = {data[x].mean()}')
    print(f'Median {x.capitalize()} = {data[x].median()}')
    print(f'Skewness of {x}: {skew}.')
    print(f'Total number of outliers in {x} distribution: {total_outlier_num}.')   


# In[ ]:





# In[ ]:





# In[23]:


histogram_boxplot(df, 'Age')


# In[24]:


summary ('Age')


# In[85]:


df['Age']. mode()


# In[22]:


# Which age is paying the highest charges for car?
data[df['Age'] == df['Age'].max()]


# # Observation
# 
# 1.Maximum age is 60, minimum is 22
# 
# 2.The age distribution is skewened to the right
# 
# 3.Majority of the age is middle aged
# 
# 4.Median age is equal 29 but the mean is ~32
# 
# 5.There are outliers in this variable.
# 
# 6.Age 60 is buying the expensive car

# In[23]:


histogram_boxplot(df, 'No_of_Dependents')


# In[24]:


summary ('No_of_Dependents')


# In[25]:


histogram_boxplot(df, 'Salary')


# In[26]:


summary ('Salary')


# In[27]:


histogram_boxplot(df, 'Partner_salary')


# In[28]:


summary ('Partner_salary')


# In[29]:


histogram_boxplot(df, 'Total_salary')


# In[30]:


summary ('Total_salary')


# # Observations
# 
# 1.Salary and Total-salary does not have any outliers.
# 
# 2.The distribution for Salary and Total-salary is close to normal, suggesting possible correlation between the two variables.
# 
# 3.There is no outlier for all of the salary variables except total salary.
# 
# 4.Partner-salary is skewed to the right, Also there is possible correlation between Partner-salary and dependents because of the visual proportion.

# In[31]:


histogram_boxplot(df, 'Price')


# In[32]:


summary ('Price')


# # Observation 
# 
# 1.The highest price for car is 80,000 and the minimum 18,000.
# 
# 2.The skewness follows the proportion of the age and partner-salary indicating correlation.

# # Let's explore the categorical variables now¶
# 
# 

# In[33]:


# function to create labeled barplots


def labeled_barplot(data, feature, perc=False, n=None):
    """
    Barplot with percentage at the top

    data: dataframe
    feature: dataframe column
    perc: whether to display percentages instead of count (default is False)
    n: displays the top n category levels (default is None, i.e., display all levels)
    """

    total = len(data[feature])  # length of the column
    count = data[feature].nunique()
    if n is None:
        plt.figure(figsize=(count + 1, 5))
    else:
        plt.figure(figsize=(n + 1, 5))

    plt.xticks(rotation=90, fontsize=15)
    ax = sns.countplot(
        data=data,
        x=feature,
        palette="Paired",
        order=data[feature].value_counts().index[:n].sort_values(),
    )

    for p in ax.patches:
        if perc == True:
            label = "{:.1f}%".format(
                100 * p.get_height() / total
            )  # percentage of each class of the category
        else:
            label = p.get_height()  # count of each level of the category

        x = p.get_x() + p.get_width() / 2  # width of the plot
        y = p.get_height()  # height of the plot

        ax.annotate(
            label,
            (x, y),
            ha="center",
            va="center",
            size=12,
            xytext=(0, 5),
            textcoords="offset points",
        )  # annotate the percentage

    plt.show()  # show the plot


# In[34]:


def labeled_barplot(data, feature, perc=False, n=None):
   

    total = len(data[feature]) 
    count = data[feature].nunique()
    if n is None:
        plt.figure(figsize=(count + 1, 5))
    else:
        plt.figure(figsize=(n + 1, 5))

    plt.xticks(rotation=90, fontsize=15)
    ax = sns.countplot(
        data=data,
        x=feature,
        palette="Paired",
        order=data[feature].value_counts().index[:n].sort_values(),
    )

    for p in ax.patches:
        if perc == True:
            label = "{:.1f}%".format(
                100 * p.get_height() / total
            )  
        else:
            label = p.get_height()  

        x = p.get_x() + p.get_width() / 2  
        y = p.get_height()  

        ax.annotate(
            label,
            (x, y),
            ha="center",
            va="center",
            size=12,
            xytext=(0, 5),
            textcoords="offset points",
        ) 

    plt.show() 
    plt.figure(figsize=(10,7))


# In[35]:


# Create a function that returns a Pie chart for categorical variable:
def pie_chart(x = 'smoker'):
    """
    Function creates a Pie chart for categorical variables.
    """
    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(aspect="equal"))

    s = data.groupby(x).size()

    mydata_values = s.values.tolist()
    mydata_index = s.index.tolist()

    def func(pct, allvals):
        absolute = int(pct/100.*np.sum(allvals))
        return "{:.1f}%\n({:d})".format(pct, absolute)


    wedges, texts, autotexts = ax.pie(mydata_values, autopct=lambda pct: func(pct, mydata_values),
                                      textprops=dict(color="w"))

    ax.legend(wedges, mydata_index,
              title="Index",
              loc="center left",
              bbox_to_anchor=(1, 0, 0.5, 1))

    plt.setp(autotexts, size=12, weight="bold")

    ax.set_title(f'{x.capitalize()} Piechart')

    plt.show()
    
   


# In[36]:


pie_chart('Gender')


# In[37]:


labeled_barplot(df, 'Gender', perc=True)


# # Observations
# 
# 1.Male is 79.2% of the sample population
# 
# 2.Female is 20.8% of the sample population

# In[39]:


pie_chart('Profession')


# In[40]:


labeled_barplot(df, 'Profession', perc=True)


# # Observations
# 
# 1.Salaried is 56.7% of the sample population
# 
# 2.Female is 43.3% of the sample population

# In[41]:


pie_chart('Marital_status')


# In[42]:


labeled_barplot(df, 'Marital_status', perc=True)


# # Observations
# 
# 1.Married has the highest popuolation with 91.3%.
# 
# 2.Single population is  8.7% 

# In[43]:


pie_chart('Education')


# In[44]:


labeled_barplot(df, 'Education', perc=True)


# In[45]:


pie_chart('Make')


# In[46]:


labeled_barplot(df, 'Make', perc=True)


# # Observations
# 
# 1.Hatchback has the highest count with 884 (55.9%) of the sample population
# 
# 2.Sedan is 460(29.1%)
# 
# 3.SUV is 237(15.0%)

# In[47]:


sns.countplot(x = 'Make', data = data)


# In[106]:



data.plot(kind="scatter", x="Gender", y="Price", 
    s=data["Salary"]*25, label="Salary", figsize=(14,10),
    c='Age', cmap=plt.get_cmap("jet"), colorbar=True,
    sharex=False)
plt.legend()


# 
# Married, Educated, Male with income as Salaried bought most of the hatchback

# In[48]:


# Dispersion in type of car
sns.catplot(x='Price',
             col='Make', 
             data=df,
             col_wrap=4,
             kind="violin")
plt.show()


# In[49]:


# Dispersion around age
sns.catplot(x='Age',
             col='Make', 
             data=df,
             col_wrap=4,
             kind="violin")
plt.show()


# In[50]:


# Dispersion Total-salary
sns.catplot(x='Total_salary',
             col='Make', 
             data=df,
             col_wrap=4,
             kind="violin")
plt.show()


# In[51]:


# Dispersion around Make
sns.catplot(x='Salary',
             col='Make', 
             data=df,
             col_wrap=4,
             kind="violin")
plt.show()


# In[57]:


# Check for correlation among numerical variables
num_var = ['Age', 'No_of_Dependents', 'Salary', 'Partner_salary', 'Total_salary', 'Price']

corr = df[num_var].corr()

# plot the heatmap

plt.figure(figsize=(15, 7))
sns.heatmap(corr, annot=True, vmin=-1, vmax=1, fmt=".2f", cmap="Spectral", xticklabels=corr.columns, yticklabels=corr.columns)
plt.show()


# In[107]:


sns.heatmap(df.corr(), annot=True)  # plot the correlation coefficients as a heatmap


# # Observations
# 
# 1.As expected,Age shows high correlation with Price
# 
# 2.No fo dependents of course would be negatively correlated with Price because number of dependents is low in single households.
# 
# 3.It is important to note that correlation does not imply causation.
# 
# 4.There does not seem to be a strong relationship between Total-salary and Price.

# In[58]:


sns.pairplot(data=df[num_var], diag_kind="kde")
plt.show()


# In[59]:


# Now we can plot all columns of our dataset in a pairplot!
sns.pairplot(data, hue  = 'Make')


# A partcularly interesting relationship between Age, Income and Make of car can be seen in this graph

# In[60]:


df.corr()   # displays the correlation between every possible pair of attributes as a dataframe


# In[72]:


plt.figure(figsize=(9, 9))
sns.histplot(data = df, x = 'Age', hue = 'Make')
plt.show()


# In[76]:


plt.figure(figsize=(9, 9))
sns.histplot(data = df, x = 'Salary', hue = 'Make')
plt.show()


# In[109]:


plt.figure(figsize=(9, 9))
sns.histplot(data = df, x = 'Total_salary', hue = 'Make')
plt.show()


# In[61]:


sns.scatterplot(df['Age'], df['Price'])  # Plots the scatter plot using two variables


# In[62]:


sns.scatterplot(df['Age'], df['Salary'])  # Plots the scatter plot using two variables


# In[79]:


sns.scatterplot(df['Age'], df['Total_salary'])  # Plots the scatter plot using two variables


# In[63]:


sns.heatmap(df.corr(), annot=True)  # plot the correlation coefficients as a heatmap


# this is not applicable because the salary is not in proportion format like ddate, time and year

# In[64]:


plt.figure(figsize=(15,7))           
sns.boxplot(df['Make'],df['Total_salary'])
plt.ylabel('Total_salary')
plt.xlabel('Make')
plt.show()


# In[65]:


plt.figure(figsize=(15,7))           
sns.boxplot(df['Make'],df['Salary'])
plt.ylabel('Salary')
plt.xlabel('Make')
plt.show()


# In[66]:


plt.figure(figsize=(15,7))           
sns.boxplot(df['Make'],df['Salary'])
plt.ylabel('Salary')
plt.xlabel('Make')
plt.show()


# In[67]:


plt.figure(figsize=(15,7))           
sns.boxplot(df['Make'],df['Price'])
plt.ylabel('Price')
plt.xlabel('Make')
plt.show()


# In[83]:


pd.crosstab(df['Make'],df['Gender']).plot(kind="bar", figsize=(8,10),
                 stacked=True)
plt.legend()
plt.show()


# In[84]:


pd.crosstab(df['Make'],df['Marital_status']).plot(kind="bar", figsize=(8,10),
                 stacked=True)
plt.legend()
plt.show()


# Comment
# 
# 
# 1.There is a clear difference the prices of the each Make.
# 
# 2.SUV is the most expensive car
# 
# 3.Hatcback is the least expensive

# In[68]:


data.groupby(['No_of_Dependents']).agg('count')['Total_salary']


# In the dataset, approximately 84% (1332 / 1581) of the Total salary have 2 and more dependents .

# In[69]:


data.groupby(['Make']).agg('count')['Total_salary']


# # Observation
# 1.Suv is the least car bought by total_salary, 
# 
# 2.This indicate household with two salary prefer Hatchback and Sedan
# 
# 3.indicating more than one car in household, Since both husband and wife are working.

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




