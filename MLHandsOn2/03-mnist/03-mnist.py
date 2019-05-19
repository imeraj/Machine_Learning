#%%
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1)
mnist.keys()

#%%
X, y = mnist['data'], mnist['target']
X.shape

#%%
y.shape

#%%
import matplotlib as mpl
import matplotlib.pyplot as plt

some_digit = X[0]
some_digit_image = some_digit.reshape(28,28)

plt.imshow(some_digit_image, cmap = mpl.cm.binary, interpolation="nearest")
plt.axis("off")
plt.show()

#%%
y[0]

#%%
import numpy as np

y = y.astype(np.uint8)
y[0]

#%%
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

#%%
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train)

#%%
sgd_clf.predict([some_digit])

#%%
some_digit_scores = sgd_clf.decision_function([some_digit])
some_digit_scores

#%%
np.argmax(some_digit_scores)

#%%
sgd_clf.classes_

#%%
from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(random_state=42)
forest_clf.fit(X_train, y_train)

#%%
forest_clf.predict([some_digit])

#%%
forest_clf.predict_proba([some_digit])

#%%
from sklearn.model_selection import cross_val_score
cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")

#%%
cross_val_score(forest_clf, X_train, y_train, cv=3, scoring="accuracy")

#%%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
# X_test_scaled = scaler.fit_transform(X_test.astype(np.float64))

#%%
cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")

#%%
cross_val_score(forest_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")

#%%
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
conf_mx = confusion_matrix(y_train, y_train_pred)
plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show()

#%%
y_train_pred = cross_val_predict(forest_clf, X_train_scaled, y_train, cv=3)
conf_mx = confusion_matrix(y_train, y_train_pred)
plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show()