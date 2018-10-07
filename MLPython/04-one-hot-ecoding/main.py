import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv("adult.data", header=None, index_col=False,
        names=['age', 'workclass', 'fnlwgt', 'education',  'education-num',
               'marital-status', 'occupation', 'relationship', 'race', 'gender',
               'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
               'income'])


data = data[['age', 'workclass', 'education', 'gender', 'hours-per-week',
                 'occupation', 'income']]

print(data.head())
print(data.describe())

print("Original features:\n", list(data.columns), "\n")
data_dummies = pd.get_dummies(data)
print("Features after get_dummies:\n", list(data_dummies.columns))
print(data_dummies.head())

features = data_dummies.ix[:, 'age':'occupation_ Transport-moving']
X = features.values
y = data_dummies['income_ >50K'].values

print("X.shape: {} y.shape: {}".format(X.shape, y.shape))

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LogisticRegression()
lr.fit(X_train_scaled, y_train)
print("Test score: {:.2f}".format(lr.score(X_test_scaled, y_test)))

print("Predicted probabilities:\n{}".format(lr.predict_proba(X_test_scaled)[:10]))
print("Sums {}".format(lr.predict_proba(X_test_scaled)[:10].sum(axis=1)))