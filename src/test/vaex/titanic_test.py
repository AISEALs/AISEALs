import vaex
import vaex.ml
import numpy as np
import pylab as plt


SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

df = vaex.ml.datasets.load_titanic()
df.info()

df = df.sample(frac=1, random_state=31)

# shuffle while exporting
df.export("shuffled.hdf5", shuffle=True)
df = vaex.open("shuffled.hdf5")
df_train, df_test = df.ml.train_test_split(test_size=0.2)

# df_train, df_test = df.ml.train_test_split(test_size=0.2, verbose=False)

df_train.describe()

# Handle missing values

# Age - just do the mean of the training set for now
median_age = df_train.percentile_approx(expression='age', percentage=50.0)
df_train['age'] = df_train.age.fillna(value=median_age)

# Fare: the mean of the 5 most common ticket prices.
fill_fares = df_train.fare.value_counts(dropna=True)
fill_fare = fill_fares.iloc[:5].index.values.mean()
df_train['fare'] = df_train.fare.fillna(value=fill_fare)

# Cabing: this is a string column so let's mark it as "M" for "Missing"
df_train['cabin'] = df_train.cabin.fillna(value='M')

# Embarked: Similar as for Cabin, let's mark the missing values with "U" for unknown
fill_embarked = df_train.embarked.value_counts(dropna=True).index[0]
df_train['embarked'] = df_train.embarked.fillna(value=fill_embarked)


