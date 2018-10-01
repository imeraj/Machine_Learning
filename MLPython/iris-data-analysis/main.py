from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

iris_data = pd.read_csv('iris-data.csv', na_values=['NA'])
print(iris_data.head())
print(iris_data.describe())

sb.pairplot(iris_data.dropna(), hue='class')
plt.show()

iris_data.loc[iris_data['class'] == 'versicolor', 'class'] = 'Iris-versicolor'
iris_data.loc[iris_data['class'] == 'Iris-setossa', 'class'] = 'Iris-setosa'

print(iris_data['class'].unique())

sb.pairplot(iris_data.dropna(), hue='class')
plt.show()

iris_data = iris_data.loc[(iris_data['class'] != 'Iris-setosa') | (iris_data['sepal_width_cm'] >= 2.5)]
print(iris_data.loc[iris_data['class'] == 'Iris-setosa', 'sepal_width_cm'].hist())
plt.show()

iris_data.loc[(iris_data['class'] == 'Iris-versicolor') &
              (iris_data['sepal_length_cm'] < 1.0),
              'sepal_length_cm'] *= 100.0

iris_data.loc[iris_data['class'] == 'Iris-versicolor', 'sepal_length_cm'].hist()
plt.show()

print(iris_data.loc[(iris_data['sepal_length_cm'].isnull()) |
              (iris_data['sepal_width_cm'].isnull()) |
              (iris_data['petal_length_cm'].isnull()) |
              (iris_data['petal_width_cm'].isnull())])

iris_data.loc[iris_data['class'] == 'Iris-setosa', 'petal_width_cm'].hist()
plt.show()

average_petal_width = iris_data.loc[iris_data['class'] == 'Iris-setosa', 'petal_width_cm'].mean()
iris_data.loc[(iris_data['class'] == 'Iris-setosa') &
              (iris_data['petal_width_cm'].isnull()),
              'petal_width_cm'] = average_petal_width
print(iris_data.loc[(iris_data['class'] == 'Iris-setosa') &
                    (iris_data['petal_width_cm'] == average_petal_width)])


iris_data.to_csv('iris-data-clean.csv', index=False)
iris_data_clean = pd.read_csv('iris-data-clean.csv')

sb.pairplot(iris_data_clean, hue='class')
plt.show()

all_inputs = iris_data_clean[['sepal_length_cm', 'sepal_width_cm',
                             'petal_length_cm', 'petal_width_cm']].values
all_labels = iris_data_clean['class'].values

model_accuracies = []

for repetition in range(1000):
    (training_inputs,
     testing_inputs,
     training_classes,
     testing_classes) = train_test_split(all_inputs, all_labels, test_size=0.25)

    random_forest_classifier = RandomForestClassifier(n_estimators=4)
    random_forest_classifier.fit(training_inputs, training_classes)
    classifier_accuracy = random_forest_classifier.score(testing_inputs, testing_classes)
    model_accuracies.append(classifier_accuracy)

print(model_accuracies)
plt.hist(model_accuracies)
plt.show()