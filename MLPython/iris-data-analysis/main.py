from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import numpy as np
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


def plot_cv(cv, features, labels):
    masks = []
    for train, test in cv.split(features, labels):
        mask = np.zeros(len(labels), dtype=bool)
        mask[test] = 1
        masks.append(mask)

    plt.figure(figsize=(15, 15))
    plt.imshow(masks, interpolation='none', cmap='gray_r')
    plt.ylabel('Fold')
    plt.xlabel('Row #')
    plt.show()


plot_cv(StratifiedKFold(n_splits=10), all_inputs, all_labels)

random_forest_classifier = RandomForestClassifier()
cv_scores = cross_val_score(random_forest_classifier, all_inputs, all_labels, cv=10)
plt.hist(cv_scores)
plt.title('Average score: {}'.format(np.mean(cv_scores)))
plt.show()

training_inputs, testing_inputs, training_classes, testing_classes \
        = train_test_split(all_inputs, all_labels, test_size=0.25)

parameter_grid = {
                    'max_depth': [1, 2, 3, 4, 5],
                'max_features' : [1, 2, 3, 4]
                 }
cross_validation = StratifiedKFold(n_splits=10)
grid_search = GridSearchCV(random_forest_classifier, param_grid=parameter_grid, cv=cross_validation)
grid_search.fit(training_inputs, training_classes)
print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))

grid_visualization = grid_search.cv_results_['mean_test_score']
grid_visualization.shape = (5, 4)
sb.heatmap(grid_visualization, cmap='Blues', annot=True)
plt.xticks(np.arange(4) + 0.5, grid_search.param_grid['max_features'])
plt.yticks(np.arange(5) + 0.5, grid_search.param_grid['max_depth'])
plt.xlabel('max_features')
plt.ylabel('max_depth')
plt.show()

parameter_grid = {'criterion': ['gini', 'entropy'],
                  'n_estimators': [10, 25, 50, 100],
                  'max_depth': [1, 2, 3, 4, 5],
                  'max_features': [1, 2, 3, 4]}

cross_validation = StratifiedKFold(n_splits=10)

grid_search = GridSearchCV(random_forest_classifier,
                           param_grid=parameter_grid,
                           cv=cross_validation)

grid_search.fit(training_inputs, training_classes)
print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))
print(grid_search.best_estimator_)

random_forest_classifier = grid_search.best_estimator_
cv_scores = cross_val_score(random_forest_classifier, training_inputs, training_classes, cv=10)
plt.hist(cv_scores)
plt.title('Average score: {}'.format(np.mean(cv_scores)))
plt.show()

rf_df = pd.DataFrame({'accuracy': cross_val_score(random_forest_classifier, training_inputs, training_classes, cv=10),
                       'classifier': ['Random Forest'] * 10})
sb.boxplot(x='classifier', y='accuracy', data=rf_df)
sb.stripplot(x='classifier', y='accuracy', data=rf_df, jitter=True, color='black')
plt.show()

random_forest_classifier.fit(training_inputs, training_classes)
classifier_accuracy = random_forest_classifier.score(testing_inputs, testing_classes)
print("Accuracy = {}".format(classifier_accuracy))