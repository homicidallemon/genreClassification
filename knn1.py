"""
================================
Nearest Neighbors Classification
================================

Sample usage of Nearest Neighbors classification.
It will plot the decision boundaries for each class.
"""
print(__doc__)

from fft2 import FourierTransform as fft
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

num_viz = 5
n_freq = 5

fft = fft(n_freq)
X = np.asarray(fft.ultimateVector)[:, :2]
#print X
y = fft.target
#nbrs = NearestNeighbors(n_neighbors=num_viz,algorithm='auto').fit(x)
#iris = datasets.load_iris()

#X = iris.data[:, :2]  # we only take the first two features. We could
                      # avoid this ugly slicing by using a two-dim dataset
#y = iris.target

h = .02  # step size in the mesh

# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

for weights in ['uniform', 'distance']:
    #Modela
    clf = neighbors.KNeighborsClassifier(num_viz, weights=weights)
    clf.fit(X, y)

    #Define as fronteiras do grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    #cria a grid
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    #Achata as multidimensoes
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    #Poe cor e figura etc
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plota
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("3-Class classification (k = %i, weights = '%s')"
              % (num_viz, weights))

plt.show()
