#!/usr/bin/env python
# coding: utf-8

# # 1. Loading libraries

# In[1]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

# Removes the limit from the number of displayed columns and rows.
# This is so I can see the entire dataframe when I print it
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.max_rows', 200)


# # 2. Loading and exploring the data
# 
# In this section the goals are to load the data into python and then to check its basic properties. This will include the dimension, column types and names, and missingness counts.

# In[2]:


df = pd.read_csv("FIFA2019.csv", index_col=0)
print(f'There are {df.shape[0]} rows and {df.shape[1]} columns.')  # f-string

# I'm now going to look at 10 random rows
# I'm setting the random seed via np.random.seed so that
# I get the same random results every time
np.random.seed(1)
df.sample(n=10)


# Looking at these 10 random rows, we can see that it'll be safe to drop a few columns right away. The `Photo`, `Flag`, and `Club Logo` column all contain URLs to images which we won't be using, so these can be safely dropped. Similarly, since we aren't trying to match this dataset to anything else, we can drop the included `ID` and just work with a default integer ID if needed. I'm also dropping `Real Face` since that seems to be an indicator of whether or not the player's real face was used for the character model and that's not something we care about for this analysis as it's unrelated to the player's statistics. Similarly `Jersey Number` won't be used.
# 
# We can also see that right now there are some columns that are represented as strings but that we really will want to be numeric. This includes columns like `Release Clause` which needs to be turned from an amount of Euros into just a number, and columns like the position ratings (e.g. `LS` and `ST`) which look something like `66+2` (when they're not missing).
# 
# This preview also shows that some columns potentially have a lot of missingness so we'll want to make sure to look into that later. `Loaned From` in particular has lots of missing values.

# In[3]:


df.drop(['ID', 'Photo', 'Flag', 'Club Logo', 'Real Face', 'Jersey Number'],axis=1,inplace=True)


# In[4]:


df.info()  # down to 82 columns after the initial 88


# In[5]:


# looking at which columns have the most missing values
df.isnull().sum().sort_values(ascending=False)


# It looks like there are blocks of similar features with the exact same number of missing values. I hypothesize that this is due to a player having either all or none of those features filled in.

# # 3. Processing columns
# 
# I want to get summary statistics and start getting a sense of the distributions of these variables and how they relate to each other, but first I need to turn many of these into numeric columns.

# ### Columns containing Euro amounts
# 
# There are some columns that represent money amounts. The values all begin with `€` and may have a `K` or `M` to represent thousands or millions. First I want to detect which columns fit this pattern, and then I'll turn these into numbers. 

# In[6]:


# this loop prints the names of the columns where there is
# at least one entry beginning the character '€'
money_cols = []
for colname in df.columns[df.dtypes == 'object']:  # only need to consider string columns
    if df[colname].str.startswith('€').any():  # using `.str` so I can use an element-wise string method
        money_cols.append(colname)
print(money_cols)


# I'm building this list automatically rather than manually entering the columns. In general this is a good practice because if I just did a quick manual skim, I might miss some columns. It'd be a good idea though to also look at the data and confirm that I haven't missed any columns that I can see have euros in them.

# In[7]:


def income_to_num(income_val):
    """This function takes in a string representing a salary in Euros
    and converts it to a number. For example, '€220K' becomes 220000.
    If the input is already numeric, which probably means it's NaN,
    this function just returns np.nan."""
    if isinstance(income_val, str):  # checks if `income_val` is a string
        multiplier = 1  # handles K vs M salaries
        if income_val.endswith('K'):
            multiplier = 1000
        elif income_val.endswith('M'):
            multiplier = 1000000
        return float(income_val.replace('€', '').replace('K', '').replace('M', '')) * multiplier
    else:  # this happens when the current income is np.nan
        return np.nan

for colname in money_cols:
    df[colname] = df[colname].apply(income_to_num)
    
df[money_cols].head()  # good to go!


# What are some ways that this processing could be made more robust? (hint: what would happen if there's leading or trailing whitespace?)
# 

# ### Position ratings
# 
# For these, we have columns that are strings containing a mix of values like `"64+2"` and `np.nan`. We will process these by just taking the number before the `+` symbol for the non-missing values, while the `np.nan`s will stay as such.

# In[8]:


def position_to_num(pos_val):
    """For each value, take the number before the '+'
    unless it is not a string value. This will only happen
    for NaNs so in that case we just return NaN.
    """
    if isinstance(pos_val, str):
        return float(pos_val.split('+')[0])
    else:
        return np.nan

position_cols = [
    'LS', 'ST', 'RS', 'LW', 'LF', 'CF', 'RF', 'RW',
    'LAM', 'CAM', 'RAM', 'LM', 'LCM', 'CM', 'RCM',
    'RM', 'LWB', 'LDM', 'CDM', 'RDM', 'RWB', 'LB',
    'LCB', 'CB', 'RCB', 'RB'
]

for colname in position_cols:
    df[colname] = df[colname].apply(position_to_num)


# ### height and weight
# 
# These also need to be turned into numerics.

# In[9]:


def height_to_num(height):
    """Converts a height of the form 5'11 (i.e. feet then inches) to inches
    and makes it an integer. Non-string heights are treated as missing."""
    if isinstance(height, str):
        splt = height.split("'")
        return float(splt[0]) * 12 + float(splt[1])
    else:
        return np.nan
    
    
def weight_to_num(weight):
    """In the weight column I'm replacing the terminal 'lbs' with
    the empty string and converting to a float. Non-strings are 
    np.nans and are kept as np.nans."""
    if isinstance(weight, str):
        return float(weight.replace('lbs', ''))
    else:
        return np.nan
    
    
    
# I could just do this by copy-pasting one line and editing which column and
# which function I'm using for each one in turn. With only two columns
# that's not so bad, but it gets cumbersome quickly

# df['Height'] = df['Height'].apply(height_to_num)
# df['Weight'] = df['Weight'].apply(weight_to_num)

# A more general way is to collect the columns and column-processing functions
# into a data structure and loop over that. That avoids bugs like forgetting to
# change the second 'Height' into 'Weight' when I copy-paste the first line.
# Here, the keys of the dictionary are the column names and the values are 
# the function that I'll use to replace that column's values. I now don't
# have to worry about mixing up column names and processing functions.
col_transforms = {
    'Height': height_to_num,
    'Weight': weight_to_num
}

# k is the key, so the column name here
# v is the value, which a function in this case and is
#     either `height_to_num` or `weight_to_num`
for k,v in col_transforms.items():
    df[k] = df[k].map(v)


# ### converting `Joined` to datetime and adding in year of joining

# In[10]:


df['Joined'] = pd.to_datetime(df['Joined'])
df['Joined year'] = df['Joined'].dt.year  # adding in a feature that's just the year
print(min(df['Joined']), max(df['Joined']))
df['Joined'].head()


# In[11]:


# investigating the players with this earliest Joined date
df[df['Joined'] == min(df['Joined'])]


# I googled this player and it seems to be [Óscar Pérez Rojas](https://en.wikipedia.org/wiki/%C3%93scar_P%C3%A9rez_Rojas). I was surprised by the time between his `Joined` date and `Contract Valid Until` but apparently he had a very long career and these values are plausible.

# ### splitting `Work Rate` into two columns

# In[12]:


df["Work Rate"].head()


# In[13]:


workrt = df["Work Rate"].str.split("/ ", n = 1, expand = True) 
workrt.head()


# In[14]:


df.drop(['Work Rate'], axis=1, inplace=True)
df["Workrate_attack"]= workrt[0]   
df["Workrate_defense"]= workrt[1]

del workrt  # don't need to do this but can keep things tidy


# In[15]:


df.head(2)


# # 4. Feature Engineering
# 
# We can always do more, but for now we'll reduce the dimensionality by a lot by replacing many of the columns with summaries of them instead. If these summaries capture most of the meaningful information in those columns then this is a good way to make a dataset more manageable.

# In[16]:


# replacing the position columns with attack, midfield, and defense averages
positiontype_to_cols = {
    'Attack': ['LS', 'ST', 'RS', 'LF', 'CF', 'RF'],
    'Midfield': ['LW', 'RW', 'LAM', 'CAM', 'RAM', 'LM', 'LCM', 'CM', 'RCM', 'RM', 'CDM', 'RDM', 'LDM'],
    'Defense': ['LWB', 'RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB']
}

for pos_type, colvec in positiontype_to_cols.items():
    df[pos_type + '_rate'] = round(df[colvec].mean(axis=1))

# we've summarized these and are content with these aggregates so we drop these columns
df.drop(position_cols, axis=1, inplace=True)

print(df.shape)  # down to 61 columns


# Combining a few more statistics to reduce the dimension

# In[17]:


summaryname_to_cols = {
    'Passing': ['Crossing', 'ShortPassing', 'LongPassing'],
    'Shooting': ['Finishing', 'Volleys', 'FKAccuracy', 'ShotPower',
                 'LongShots', 'Penalties', 'HeadingAccuracy'],
    'Defending': ['Marking', 'StandingTackle', 'SlidingTackle', 'Interceptions'],
    'Speed': ['SprintSpeed', 'Agility', 'Acceleration', 'Reactions', 'Stamina'],
    'Control': ['BallControl', 'Curve', 'Dribbling'],
    'GoalKeeping': ['GKDiving', 'GKHandling', 'GKPositioning', 'GKKicking', 'GKReflexes'],
    'Mental': ['Composure', 'Vision', 'Aggression', 'Positioning'],
    'Power': ['Strength', 'Balance', 'Jumping'],
    'Avg_rating': ['Overall', 'Potential']
    
}

for summarycol, colvec in summaryname_to_cols.items():
    df[summarycol] = round(df[colvec].mean(axis=1))
    
# now I don't have a vector that contains all of these old columns in one so I need to make it
# this is a "nested list comprehension" and is a common way to flatten a list of lists,
# which is what summaryname_to_cols.values() effectively is, so that the result is a 
# single list. The syntax says to loop colvec through each element of summaryname_to_cols.values(),
# and then within each iteration of that loop we loop colname through colvec and we simply
# keep colvec without doing anything else to it.
cols_to_drop = [colname for colvec in summaryname_to_cols.values() for colname in colvec]
df.drop(cols_to_drop, axis=1, inplace=True)
print(df.shape)  # big reduction in size. Getting much more manageable
df.head(2)


# In[18]:


df.info()


# Now everything that should be numeric indeed is and we can start looking at summary statistics.

# # 5. Basic summary statistics and consequences
# 
# This is EDA not data preprocessing but we need to know what's going on in order to know how to prepare the data for whatever comes next.

# In[19]:


df.describe().T  # quick summary of numeric features


# In[20]:


# looking at value counts for non-numeric features

num_to_display = 10  # defining this up here so it's easy to change later if I want
for colname in df.dtypes[df.dtypes == 'object'].index:
    val_counts = df[colname].value_counts(dropna=False)  # i want to see NA counts
    print(val_counts[:num_to_display])
    if len(val_counts) > num_to_display:
        print(f'Only displaying first {num_to_display} of {len(val_counts)} values.')
    print('\n\n') # just for more space between 


# In the `Body Type` column there are a few famous players that just have their own name as their body type, in addition to one player with a value of `PLAYER_BODY_TYPE_25`. 
# 
# There are also values in `Contract Valid Until` that don't look valid.
# 
# This is an example of how EDA, like looking at these distributions, shows more preprocessing to do.

# #### Fixing `Body Type`

# In[21]:


df[df['Body Type'] == 'PLAYER_BODY_TYPE_25']


# M. Salah is also quite famous so that fits this pattern of a few famous players not having one of the three listed body types.
# 
# For our later analyses it won't be helpful to have this categorical variable with some values that only appear once, so I'll do my best to assign those players to one of the thrree listed body types.

# In[22]:


df.loc[df['Body Type'] == 'PLAYER_BODY_TYPE_25', 'Body Type'] = 'Lean'
df.loc[df['Body Type'] == 'Neymar', 'Body Type'] = 'Lean'
df.loc[df['Body Type'] == 'Shaqiri', 'Body Type'] = 'Stocky'
df.loc[df['Body Type'] == 'Messi', 'Body Type'] = 'Lean'
df.loc[df['Body Type'] == 'C. Ronaldo', 'Body Type'] = 'Stocky'
df.loc[df['Body Type'] == 'Courtois', 'Body Type'] = 'Lean'
df.loc[df['Body Type'] == 'Akinfenwa', 'Body Type'] = 'Stocky'
# why do these work with NaNs?

df['Body Type'].value_counts(dropna=False)


# #### Fixing `Contract Valid Until`

# In[23]:


df['Contract Valid Until'].value_counts(dropna=False)


# In[24]:


contract_dates = pd.to_datetime(df['Contract Valid Until']).dt.year
print(contract_dates.value_counts(dropna=False))
df['Contract Valid Until'] = contract_dates


# ### Distributions

# In[25]:


# df['is_GK'] = df['Position'] == 'GK'  # for hue
# cols_to_exclude = ['International Reputation', 'Weak Foot', 'Skill Moves']
# sns.pairplot(df[[colname for colname in df.columns if colname not in cols_to_exclude]], hue = 'is_GK')
# df.drop(['is_GK'], axis=1, inplace=True)


# There are some really tightly correlated ones in there. I'll drop some of those.

# In[26]:


df.drop(['Release Clause'], axis=1, inplace=True)


# ### Log transformation
# 
# Some features are very skewed and will likely behave better on the log scale.
# 
# I'll transform both `Wage` and `Value`.

# In[27]:


cols_to_log = ['Wage', 'Value']
for colname in cols_to_log:
    plt.hist(df[colname], bins=50)
    plt.title(colname)
    plt.show()
    print(np.sum(df[colname] <= 0))


# Unfortunately there are some non-positive values (in this case, exact zeros) so we can't directly take the log of these numbers.
# 
# We have a couple options. One is to just add some small positive value to every element of these columns so that the log is defined on every value in the column we're transforming. In this case, since these are like counts, we could just add $1$ since $1$ euro is tiny compared to the amounts described by these variables, so we're not changing the data in a meaningful way.
# 
# But sometimes the context is such that it is not acceptable to change the data by adding something. In these cases we'll want to consider other transformations. One option is to use a power like `sqrt` which is like a weaker log transform but it can handle zeros. Another option is to use `np.arcsinh` which is like the log for large values but handles negative and zero values as well.

# In[28]:


plt.hist(np.log(df['Wage'] + 1), 50)
plt.title('log(Wage + 1)')
plt.show()
plt.hist(np.arcsinh(df['Wage']), 50)
plt.title('arcsinh(Wage)')
plt.show()
plt.hist(np.sqrt(df['Wage']), 50)
plt.title('sqrt(Wage)')
plt.show()


# All three have helped but the sqrt is not quite strong enough and the result is still too skewed in my opinion, so I prefer the log or arcsinh. The log and arcsinh look similar so the difference there is more be about interpretation. It will likely be easier to explain the log of a number to someone than the arcsinh of a number since that's a less known transformation, so if this is for a client or something similar I'd likely choose log(Wage + 1), but if this is just for my own internal use I'd pick arcsinh since I prefer to use transformations that naturally are defined on all the values I have rather than needing to modify the data first.

# In[29]:


for colname in cols_to_log:
    df[colname + '_log'] = np.log(df[colname] + 1)
df.drop(cols_to_log, axis=1, inplace=True)


# The log transformation decreases the scale of the distributions, even with the huge range of Wage. It seems the outliers caused the log-transformed distributions to still be a bit skewed, but it is closer to normal than the original distribution.

# ### Binning
# 
# Generally it's better to keep continuous features as such, but sometimes binning is necessary so we'll do an example of that here with `Height`.

# In[30]:


# Height is in inches
binned_ht = pd.cut(df['Height'], [-np.inf, 5*12, 5*12+6, 6*12, np.inf])
binned_ht


# In[31]:


binned_ht.value_counts(dropna=False)


# In[32]:


# can add custom labels
df['height_bin'] = pd.cut(
    df['Height'], [-np.inf, 5*12, 5*12+6, 6*12, np.inf], 
    labels = ["Under 5'", "5' to 5'6", "5'6 to 6'", "Over 6'"]
)
df.drop(['Height'], axis=1, inplace=True)
df['height_bin'].value_counts(dropna=False)


# ### Changing units
# 
# We'll change the units of `Weight` to be in kg instead of lbs

# In[33]:


print(df['Weight'].head(2))
df['Weight'] = df['Weight'].apply(lambda wt: round(wt * 0.4535, 2))
df['Weight'].head(2)


# ### Making categoricals into categorical types

# In[34]:


cat_vars = ['Preferred Foot', 'Body Type', 'Position',
            'Workrate_attack', 'Workrate_defense']
# the other categorical variables have lots of levels
# and I wouldn't dummy encode them as such

for colname in cat_vars:
    df[colname] = df[colname].astype('category')
    
df.info()


# ### Text processing
# 
# I'll do a few text processing things here as examples.

# In[35]:


# how many players are in clubs that start with 'FC'?
df['Club'].str.startswith('FC').sum()


# In[36]:


# how many letters and words in the unique club names?

# doing i == i as a quick check for NaNs
# using .title() in case of capitalization issues
club_data = pd.DataFrame(
    data = [(i, len(i), len(i.split())) if i == i else (i, 0, 0)
            for i in df['Club'].str.strip().str.title().unique()],
    columns = ['Club', 'Number of Letters', 'Number of Words']
)
club_data.head()


# In[37]:


club_data['Number of Letters'].value_counts().plot.bar()
plt.title('Distribution of Number of Letters')
plt.show()

club_data['Number of Words'].value_counts().plot.bar()
plt.title('Distribution of Number of Words')
plt.show()

print('These are the clubs with the most words in the name:')
club_data.loc[club_data['Number of Words'] == club_data['Number of Words'].max(), 'Club']


# ### Standardizing continuous features
# 
# For some features maybe it makes more sense to have the values be how many standard deviations from the mean that player is. This would be done via a z-transformation.
# 
# In other cases, we might want to make sure that all of the values are between 0 and 1. This can be done with minmax scaling.

# In[38]:


from sklearn.preprocessing import StandardScaler, MinMaxScaler


# In[39]:


std_scaler = StandardScaler()

df['Weight'].hist(bins=20)
plt.title('Weight before z transformation')
plt.show()
# fit_transform requires a DataFrame, not a Series, hence
# the double brackets to keep df[['Weight']] as a 1 column 
# DataFrame rather than a Series, like if I did df['Weight']


df['Weight_z_std'] = std_scaler.fit_transform(df[['Weight']])
df['Weight_z_std'].hist(bins=20)
plt.title('Weight after z transformation')
plt.show()
# exact same shape since it's a linear transformation.
df.drop(['Weight'], axis=1, inplace=True)


# In[40]:


# replacing with scaled 
df['Attack_rate'].hist(bins=20)
plt.title('Attack_rate before minmax scaling')
plt.show()

df[['Attack_rate', 'Midfield_rate', 'Defense_rate']] = MinMaxScaler().fit_transform(
    df[['Attack_rate', 'Midfield_rate', 'Defense_rate']]
)

df['Attack_rate'].hist(bins=20)
plt.title('Attack_rate after minmax scaling')
plt.show()

# if the minimum and maximum are treated as fixed, this is also a linear transformation
# so the shape is the same


# Min-Max scaler doesn’t reduce the skewness of a distribution. It simply shifts the distribution to a smaller scale [0–1]. For this reason, it seems Min-Max scaler isn’t the best choice for a distribution with outliers or severe skewness.

# The standard scaler assumes features are normally distributed and will scale them to have a mean 0 and standard deviation of 1. Unlike Min-Max scaler the Standard scaler doesn’t have a predetermined range to scale to.

# # 6. Missing values
# 
# There are lots of ways to handle missing values. 
# 
# I'm going to start by investigating the patterns in the missingness.

# In[41]:


df.isnull().sum() # lots of columns don't have missingness


# In[42]:


# counting the number of missing values per row
df.isnull().sum(axis=1).value_counts()


# The `Loaned From` column has a lot of missing values. These could be players who have never been loaned but there could also be players where the value is unknown. Since I'm not particularly interested in this column and I can't confidently separate out those two concepts, I'll drop it.

# In[43]:


df.drop(['Loaned From'], axis=1, inplace=True)


# In[44]:


# most rows don't have missing values now
num_missing = df.isnull().sum(axis=1)
num_missing.value_counts()


# I'll check the rows that have exactly 2 and exactly 3 missing values to see what's going on.

# In[45]:


# these are missing `Joined` and `Joined year`
df[num_missing == 2].sample(n=5)


# In[46]:


# these are missing `Attack_rate`, `Midfield_rate`, and `Defense_rate`
df[num_missing == 3].sample(n=5)


# Overall it looks like the missingness has a structure to it. I'll now do this programatically to investigate this.

# In[47]:


for n in num_missing.value_counts().sort_index().index:
    if n > 0:
        print(f'For the rows with exactly {n} missing values, NAs are found in:')
        n_miss_per_col = df[num_missing == n].isnull().sum()
        print(n_miss_per_col[n_miss_per_col > 0])
        print('\n\n')
        


# This confirms that certain columns tend to be missing together or all nonmissing together. How exactly we handle this will depend on what we're doing. For visualization we may just drop the missing values, but for modeling we will likely want to either impute them or use a method that can handle missing predictor values.
# 
# I'll demonstrate a few common techniques before moving on.

# #### filling in with a standard value

# In[48]:


# nans are floats so they become strings here
# we also need this to be strings because we're adding a category that's not present
df['height_bin'] = df['height_bin'].astype(str).replace('nan', 'is_missing').astype('category')


# In[49]:


# now using `fillna` with a numeric column
print(df['Passing'].isnull().sum())
df['Passing'].fillna(df['Passing'].mean(), inplace=True)  # mean imputation
df['Passing'].isnull().sum()


# Later we'll see how we can use a model to fill in missing values based on similarities in non-missing values.

# ### Encoding categorical columns
# 
# Many machine learning algorithms can support categorical values without further manipulation but there are many more algorithms that do not. 
# 
# Sometimes we want to one-hot encode where every level gets an indicator, but other times we want to drop one level to account for the fact that in a model we will have an intercept.
# 
# This is worth doing at the very end because for many earlier analysis tasks (e.g. visualization and value counts) it will be more convenient to keep categorical variables as a single column. Additionally, we'll want to make sure missing values are resolved by this point. 

# In[50]:


pd.get_dummies(df['height_bin'], drop_first=True)


# In[51]:


# can do one hot encoding with get_dummies
pd.get_dummies(df['height_bin'], drop_first=False).iloc[:10, :]


# In[52]:


# or we can use sklearn
from sklearn.preprocessing import OneHotEncoder

OneHotEncoder(sparse=False).fit_transform(df[['height_bin']])[:10,:]


# # Outliers

# An outlier is a data point that is distant from other similar points. 
# 
# Outliers in the data can distort predictions and affect the accuracy so it's important to flag them for review. This is especially the case with regression models.  
# 
# The challenge with outlier detection is determining if a point is truly a problem or simply a large value. If a point is genuine then it is very important to keep it in the data as otherwise we're removing the most interesting pdata points. Regardless, it is essential to understand their impact on our predictive models and statistics, and the extent to which a small number of outlying points are dominating the fit of the model (for example, the mean is much more sensitive to outliers than the median). It is left to the best judgement of the investigator to decide whether treating outliers is necessary and how to go about it. Knowledge of domain and impact of the business problem tend to drive this decision.

# ### Outlier detection using Z-Score
# 
# The z-transformation used earlier can flag a point as being far away from the mean. If the data are normally distributed then we expect the vast majority of points to be within 3 standard deviations of the mean, which corresponds to a z score with an absolute value of at most 3. 
# 
# If the data are not normal, however, the situation is more complicated.

# In[53]:


def z_transform(x):
    return (x - np.mean(x)) / np.std(x)

np.random.seed(1)
x1 = np.random.normal(size=1000)
x2 = np.random.lognormal(size=1000)


plt.hist(z_transform(x1))
plt.title('z-transformed normal data')
plt.show()


plt.hist(z_transform(x2))
plt.title('z-transformed lognormal data')
plt.show()


# All of the points in each plot are drawn from the exact same distribution, so it's not fair to call any of the points outliers in the sense of there being bad data. But depending on the distribution in question, we may have almost all of the z-scores between -3 and 3 or instead there could be extremely large values. 

# ### Outlier detection using IQR
# 
# Another way to flag points as outliers is to compute the IQR, which is the interval going from the 1st quartile to the 3rd quartile of the data in question, and then flag a point for investigation if it is outside 1.5 * IQR. 

# In[54]:


def frac_outside_1pt5_IQR(x):
    length = 1.5 * np.diff(np.quantile(x, [.25, .75]))
    return np.mean(np.abs(x - np.median(x)) > length)

print(frac_outside_1pt5_IQR(x1))
print(frac_outside_1pt5_IQR(x2))


# With the normal data this only flags 5% of the points as suspicious, but with the lognormal data over 13% of the sample is flagged. This again shows how these statistics depend on the underlying distributions and can't be used without the context.

# ### Applying these to the FIFA data
# 
# As an example, we'll look at `Power`.

# In[55]:


plt.hist(df['Power'], 20)
plt.title('Histogram of Power')
plt.show()

sns.boxplot(df['Power'])
plt.title('Boxplot of Power')
plt.show()


# `Power` has a lot of values that are flagged as suspicious by the boxplot, but in the histogram we can see that the distribution is skewed so these points aren't inconsistent with the overall distribution of the data. Nevertheless, having a heavy tail means we might want to consider statistics less sensitive to large values, so e.g. the median may be a better measure of central tendancy. 

# #### handling outliers
# 
# If we decide that we do actually have some problematic outliers, we have a couple options.
# 
# - if the point seems truly nonsensical it may be best to treat it as missing
# 
# - alternatively, we could drop that observation or we could use statistics that are robust to outliers
# 
# It's often a good idea to examine the sensitivity to outliers by running an analysis with and without them.

# In[58]:


quartiles = np.quantile(df['Power'][df['Power'].notnull()], [.25, .75])
power_4iqr = 4 * (quartiles[1] - quartiles[0])
print(f'Q1 = {quartiles[0]}, Q3 = {quartiles[1]}, 4*IQR = {power_4iqr}')
outlier_powers = df.loc[np.abs(df['Power'] - df['Power'].median()) > power_4iqr, 'Power']
outlier_powers


# In[59]:


# making the situation more extreme
df['Power'].hist(bins=20)
plt.title('Power before exaggerating the outliers')
plt.show()
print(df['Power'].mean())
df.loc[outlier_powers.index, 'Power'] = [-200000.0, -1200000.0]
df['Power'].hist(bins=20)
plt.title('Power after exaggerating outliers')
plt.show()


# In[ ]:


# if we wanted to make these NA we could just do this
# [not run]
df.loc[np.abs(df['Power'] - df['Power'].median()) > power_4iqr, 'Power'] = np.nan


# In[ ]:


# dropping these rows
# [not run]
df.drop(outlier_powers.index, axis=0, inplace=True)


# In[60]:


power = df['Power'][df['Power'].notnull()]

print(power.mean())  # the mean is being pulled
print(power.median())


# In[61]:


from scipy.stats import tmean

print(tmean(power, limits=np.quantile(power, [.1, .9])))
print(tmean(power, limits=[0,100]))


# In[ ]:




