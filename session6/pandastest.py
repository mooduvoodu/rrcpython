
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#something

# Creating a Series by passing a list of values, letting pandas create a default integer index

s = pd.Series([1,3,5,np.nan,6,5])



# Creating a DataFrame by passing a numpy array, with a datetime index and labeled columns.

dates = pd.date_range('20130101',periods=6)
print(dates)



df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD'))
print(df)

#Creating a DataFrame by passing a dict of objects that can be converted to series-like.

df2 = pd.DataFrame({'A': 1.,
    'B': pd.Timestamp('20130102'),
    'C': pd.Series(1, index=list(range(4)), dtype='float32'),
    'D': np.array([3] * 4, dtype='int32'),
    'E': pd.Categorical(["test", "train", "test", "train"]),
    'F': 'foo'})

print(df2)

# The columns of the resulting DataFrame have different dtypes.

df2.dtypes

print(df2)

#Here is how to view the top and bottom rows of the frame:

print(df.head())
print(df.tail(3))

#Display the index, columns:

print(df.index)
print(df.columns)

#describe() shows a quick statistic summary of your data:

print(df.describe())

#Sorting by values:

print(df.sort_values(by='B'))

#Selecting a single column, which yields a Series, equivalent to df.A:

print(df['A'])

#Selecting via [], which slices the rows.

print(df[0:100])
print(df['20130102':'20130104'])

# selecting by label 

print(df.loc[dates[0]])

#Selecting on a multi-axis by label:


#Showing label slicing, both endpoints are included:
print(
    df.loc['20130102':'20130104', ['A', 'B']]
    )

#For getting a scalar value:

print(
    df.loc[dates[0], 'A']
    )

#Selection by position

print(
    df.iloc[3]
    )



#By integer slices, acting similar to numpy/python:

print(
    df.iloc[3:5, 0:2]
    )


#By lists of integer position locations, similar to the numpy/python style:

print(
    df.iloc[[1, 2, 4], [0, 2]]
    )

#For slicing rows explicitly:

print(
    df.iloc[1:3, :]
    )

#For slicing columns explicitly:

print(
    df.iloc[:, 1:3]
    )

#Boolean Indexing
#Using a single column’s values to select data.
print(
    df[df['A'] > 0]
    )

#Selecting values from a DataFrame where a boolean condition is met.

print(
    df[df > 0]
    )

#Using the isin() method for filtering:

df2 = df.copy()
df2['E'] = ['one', 'one', 'two', 'three', 'four', 'three']
print(df2)

print(
    df2[df2['E'].isin(['two', 'four'])]
    )

#Setting
#Setting a new column automatically aligns the data by the indexes.

s1 = pd.Series([1, 2, 3, 4, 5, 6], index=pd.date_range('20130102', periods=6))
print(s1)
print(df)
df['F'] = s1
print(df)

#Setting values by label:

df.at[dates[0], 'A'] = 0

#Setting values by position:

df.iat[0, 1] = 0

#Setting by assigning with a NumPy array:

df.loc[:, 'D'] = np.array([5] * len(df))

print(df)

#SET work
#Concat  (UNION ALL)

df = pd.DataFrame(np.random.randn(10, 4))
print(df)
print("seperator")
# break it into pieces
pieces = [df[:3], df[3:7], df[7:]]
print(df[:3])
print(df[3:7])
print(df[7:])

print(pd.concat(pieces))

#JOIN

left = pd.DataFrame({'key': ['foo', 'foo'], 'lval': [1, 2]})
right = pd.DataFrame({'key': ['foo', 'foo'], 'rval': [4, 5]})

print(left)
print(right)

print(
    pd.merge(left, right, on='key')
    )

#getting data in and out

df.to_csv('foo.csv')
pd.read_csv('foo.csv')
df.to_excel('foo.xlsx', sheet_name='Sheet1')
pd.read_excel('foo.xlsx', 'Sheet1', index_col=None, na_values=['NA'])











