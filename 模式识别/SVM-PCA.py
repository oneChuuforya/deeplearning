import numpy as np
from sklearn.decomposition import RandomizedPCA
from sklearn.datasets import fetch_mldata
from sklearn import svm, metrics
from sklearn import cross_validation

import time

custom_data_home = 'D:\Python-Data\Python\data'
mnist = fetch_mldata('MNIST original', data_home=custom_data_home)

print("Data shape ")
print(mnist.data.shape) #(70000, 784)
print("Length : " + str(len(mnist.data[0])))

X_train, y_train = np.float32(mnist.data[:60000])/ 255., np.float32(mnist.target[:60000])

pca = RandomizedPCA(n_components=90)
pca.fit(X_train)

print("Variance explained")
print(np.sum(pca.explained_variance_ratio_))

train_ext = pca.fit_transform(X_train)

print("Train-set dimensions after PCA")
print(train_ext.shape)

start = int(round(time.time() * 1000))

classifier = svm.SVC(gamma=0.01, C=3, kernel='rbf')
classifier.fit(train_ext,y_train)

print("accuracy")
print(cross_validation.cross_val_score(classifier, train_ext, y_train, cv=5))

end = int(round(time.time() * 1000))
print("cost", (end-start), "ms")

X_test, y_test = np.float32(mnist.data[60000:]) / 255., np.float32(mnist.target[60000:])

test_ext = pca.transform(X_test)
print("Test-set dimensions")
print(test_ext.shape)
expected = y_test
predicted = classifier.predict(test_ext)

print("Classification report %s\n"
      % (metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))