import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split

data = pd.read_csv("spectrum.csv", sep = ';', header = 0, index_col = 0)
train_data, test_data = train_test_split(data, test_size = 0.3)

train_cols = data.columns[:-2]

logistic = LogisticRegression()
clf = logistic.fit(train_data[train_cols], train_data[]