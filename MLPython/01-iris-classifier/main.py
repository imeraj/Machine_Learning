from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
import numpy as np

iris_dataset = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'],
                                                    iris_dataset['target'], random_state=0)
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train_scaled, y_train)

X_new = np.array([[5, 2.9, 1, 0.2]])
prediction = knn.predict(scaler.transform(X_new))
y_pred = knn.predict(X_test_scaled)

print("Keys of iris_dataset: \n{}".format(iris_dataset.keys()))
print("Target names: {}".format(iris_dataset['target_names']))
print("Feature names: \n{}".format(iris_dataset['feature_names']))
print("First five columns of data:\n{}".format(iris_dataset['data'].shape))
print("Shape of target: {}".format(iris_dataset['target'].shape))

print("X_train shape: {}".format(X_train.shape))
print("y_train shape: {}".format(y_train.shape))

print("X_test shape: {}".format(X_test.shape))
print("y_test shape: {}".format(y_test.shape))

print("Prediction: {}".format(prediction))
print("Predicted target name: {}".format(iris_dataset['target_names'][prediction]))

print("Test set predictions:\n {}".format(y_pred))
print("Test set score: {:.2f}".format(knn.score(X_test_scaled, y_test)))

print("Predicted probabilities:\n{}".format(knn.predict_proba(X_test_scaled)[:6]))
print("Sums {}".format(knn.predict_proba(X_test_scaled)[:6].sum(axis=1)))

print("Argmax of decision function: {}".format(np.argmax(knn.predict_proba(X_test_scaled), axis=1)[:10]))
print("Argmax combined with classes_: {}".format(iris_dataset['target_names'][np.argmax(knn.predict_proba(X_test_scaled), axis=1)][:10]))