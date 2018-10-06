from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

digits_dataset = load_digits()

fig, axes = plt.subplots(2, 5, figsize=(10, 5),
                             subplot_kw={'xticks':(), 'yticks': ()})
for ax, img in zip(axes.ravel(), digits_dataset.images):
    ax.imshow(img)
plt.show()

print("Keys of dataset: {}".format(digits_dataset.keys()))
print("Target names: {}".format(digits_dataset['target_names']))
print("Shape of data: {}".format(digits_dataset['data'].shape))
print("Shape of target: {}".format(digits_dataset['target'].shape))

pca = PCA(n_components=2)
pca.fit(digits_dataset.data)
# transform the digits data onto the first two principal components digits_pca =
digits_pca = pca.transform(digits_dataset.data)

colors = ["#476A2A", "#7851B8", "#BD3430", "#4A2D4E", "#875525",
              "#A83683", "#4E655E", "#853541", "#3A3120", "#535D8E"]

plt.figure(figsize=(10, 10))
plt.xlim(digits_pca[:, 0].min(), digits_pca[:, 0].max())
plt.ylim(digits_pca[:, 1].min(), digits_pca[:, 1].max())
for i in range(len(digits_dataset.data)):
        # actually plot the digits as text instead of using scatter
        plt.text(digits_pca[i, 0], digits_pca[i, 1], str(digits_dataset.target[i]),
                 color = colors[digits_dataset.target[i]],
                 fontdict={'weight': 'bold', 'size': 9})
plt.xlabel("First principal component")
plt.ylabel("Second principal component")
plt.show()

tsne = TSNE(random_state=42)
digits_tsne = tsne.fit_transform(digits_dataset.data)

plt.figure(figsize=(10, 10))
plt.xlim(digits_tsne[:, 0].min(), digits_tsne[:, 0].max() + 1)
plt.ylim(digits_tsne[:, 1].min(), digits_tsne[:, 1].max() + 1)
for i in range(len(digits_dataset.data)):
        # actually plot the digits as text instead of using scatter
        plt.text(digits_tsne[i, 0], digits_tsne[i, 1], str(digits_dataset.target[i]),
                 color = colors[digits_dataset.target[i]],
                 fontdict={'weight': 'bold', 'size': 9})
plt.xlabel("t-SNE feature 0")
plt.xlabel("t-SNE feature 1")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(digits_dataset['data'], digits_dataset['target'], random_state=0)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train);

y_pred = knn.predict(X_test)

print("Test set predictions:\n {}".format(y_pred))
print("Test set score: {:.5f}".format(knn.score(X_test, y_test)))

print("Predicted probabilities:\n{}".format(knn.predict_proba(X_test)[:6]))
print("Sums {}".format(knn.predict_proba(X_test)[:6].sum(axis=1)))



