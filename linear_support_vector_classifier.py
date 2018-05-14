# NumPy is the fundamental package for scientific computing with Python. It contains among other things:
#
# a powerful N-dimensional array object
# sophisticated (broadcasting) functions
# tools for integrating C/C++ and Fortran code
# useful linear algebra, Fourier transform, and random number capabilities
import numpy as np

# Matplotlib is a Python 2D plotting library
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")

# Python machine learning library
from sklearn import svm

# A feature is an array of data

# feature1
x = [1, 5, 1.5, 8, 1, 9]
# feature2
y = [2, 8, 1.8, 8, 0.6, 11]

# plot graph
plt.scatter(x,y)

# show visual graph
plt.show()

# An array of features
X = np.array([[1,2],
              [5,8],
              [1.5,1.8],
              [8,8],
              [1,0.6],
              [9,11]])

# Our "targets" or "labels"
# Here I define 1 to mean stock solved  above recommended stock price and 0 to mean sold below stock price
# I am labeling each array plotting with what it's outcome was i.e what it means in the real world.
y = [0,1,0,1,0,1]

# C is a valuation of "how badly" you want to properly classify, or fit, everything. (1) is the default
# Linear growth
clf = svm.SVC(kernel='linear', C = 1.0)

# Sklearn required the data to be re-shaped for fitting
reshaped_X = X.reshape(6, -1)
clf.fit(X,y)

print(clf.predict([0.58,0.76]))
print(clf.predict([10.58,10.76]))