from root_numpy import root2array, tree2array, list_structures
from root_numpy import testdata
import numpy as np
import pandas

import keras
from keras.models import Sequential, load_model
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout, Activation, Flatten, Merge
from keras.layers import Conv2D, MaxPooling2D

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix,accuracy_score,average_precision_score,roc_auc_score

model = load_model("model_arch.h5")
print(model.summary())

#predictionsBkg = [-0.997004, 1.20295, -0.722999, -1.49033, 1.36544, -0.949627, -0.0958932, -0.192436, -0.634542, -1.72239, 1.2252, -0.0850188, -0.269918, 0.645118, 2.78293, 1.00219, -0.940145, -0.4618, -1.64704, 700.]

#predictionsBkgPlot = model.predict(predictionsBkg)

#print(predictionsBkgPlot)
