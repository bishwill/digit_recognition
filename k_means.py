import numpy as np
from sklearn.cluster import KMeans
from keras.datasets import mnist
from PCA import pca
import matplotlib.pyplot as plt


(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape((60000, 784))
X_test = X_test.reshape((10000, 784))

reduced = pca(X_test, 2) 


plt.plot(reduced[:,0], reduced[:,1])
plt.show


# kmeans = KMeans(n_clusters=10, verbose=True)
# kmeans.fit(X_train)




# with open('kmeans_output.txt', 'w') as f:
#     np.savetxt(f, kmeans.predict(X_test))
    

