import numpy as np
import pandas

import keras
from keras.models import Sequential
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix,accuracy_score,average_precision_score,roc_auc_score
from sklearn.metrics import mean_squared_error

def normalize(vector, length):
    for i in range(0,length):
        mean = np.mean(vector[:,i])
        std = np.std(vector[:,i]) 

        if(std!=0):
            vector[:,i] = (vector[:,i] - mean)/std
    
    return vector

def get_label_and_values(sample):
    Values = sample.drop(['Label'], axis=1)
    Labels = sample['Label']

    Values_array = Values.to_numpy()
    Labels_array = Labels.to_numpy()

    return Values_array,Labels_array

##################################

# JUST THESE FOLLOWING LINES TO BE MODIFIED

##################################
 
# Convert a TTree in a ROOT file into a NumPy structured array + pandas Dataframe, with which you can create an array to be given to Keras
branches_to_analyze = ["DNN_MET_eta","DNN_MET_phi","DNN_MET_pt","DNN_gluon1_eta","DNN_gluon1_phi","DNN_gluon1_pt","DNN_gluon2_eta","DNN_gluon2_phi","DNN_gluon2_pt","DNN_lepton_eta","DNN_lepton_phi","DNN_lepton_pt","DNN_leptopjet_eta","DNN_leptopjet_phi","DNN_leptopjet_pt","DNN_ttaggedjet_eta","DNN_ttaggedjet_phi","DNN_ttaggedjet_pt"]
save_numpy_format = False
epochs=100
number_hidden_nodes=[30,15,1]
number_layers=len(number_hidden_nodes)
activations=['tanh','tanh','tanh']
batch_size=128
filenameBkg = 'hadded/uhh2.AnalysisModuleRunner.MC.TTbar.root'
filenameSig = 'hadded/uhh2.AnalysisModuleRunner.MC.TstarTstar_M-Combined.root'
filenameSig_notseen = 'hadded/uhh2.AnalysisModuleRunner.MC.TstarTstar_M-1200.root'
mass_notseen = 1200
train_parametric = True
validation_split = 0.2
split_train_test = 0.8
smear = True

##########################################################################

outfileBkg = 'outBkg.npy'
outfileSig = 'outSig.npy'

Arr_Bkg = np.load(outfileBkg)
Arr_Sig = np.load(outfileSig)

arrBkg = pandas.DataFrame(data=Arr_Bkg,columns=branches_to_analyze)
arrSig = pandas.DataFrame(data=Arr_Sig,columns=branches_to_analyze)

#adding a column for labels
label_bkg = np.zeros(len(arrBkg))
label_sig = np.ones(len(arrSig))

arrBkg.insert(len(branches_to_analyze), "Label", label_bkg, True)
arrSig.insert(len(branches_to_analyze), "Label", label_sig, True) 

frames = [arrBkg, arrSig]
result = pandas.concat(frames)

result = result.sample(frac=1).reset_index(drop=True)

msk = np.random.rand(len(result)) < split_train_test
train_sample = result[msk]
test_sample = result[~msk]

print(test_sample.head(10))

if(smear):

    METsmeared = pandas.DataFrame({'DNN_MET_pt': test_sample.loc[:,'DNN_MET_pt']*10, 'DNN_gluon1_pt': test_sample.loc[:,'DNN_gluon1_pt']*10, 'DNN_gluon2_pt': test_sample.loc[:,'DNN_gluon2_pt']*10})
    test_sample.update(METsmeared)

print(test_sample.head(10))

#create array for Keras implementation
Values_array_train,Labels_array_train = get_label_and_values(train_sample)
Values_array_test,Labels_array_test = get_label_and_values(test_sample)
    
#VERY IMPORTANT! One does need to normalize the features, otherwise training is super difficult!
Values_array_train = normalize(Values_array_train,len(branches_to_analyze))
Values_array_test = normalize(Values_array_test,len(branches_to_analyze))

back_dim = len(result.loc[result['Label'] == 0])
sig_dim = len(result.loc[result['Label'] == 1])
print('We have number of background events equal to '+str(back_dim))
print('We have number of signal events equal to '+str(sig_dim))

print('Variables setup')

print('Defining the Keras model')

import warnings
warnings.filterwarnings('ignore')

from BayesianNN import *

from keras.layers import Input
from keras.models import Model

train_size = len(Values_array_train)
num_batches = train_size / batch_size

kl_weight = 1.0 / num_batches
prior_params = {
    'prior_sigma_1': 1.0, 
    'prior_sigma_2': 1.0, 
    'prior_pi': 0.1 
}

x_in = Input(shape=(len(branches_to_analyze),))
x = DenseVariational(50, kl_weight, **prior_params, activation='tanh')(x_in)
x = DenseVariational(30, kl_weight, **prior_params, activation='tanh')(x)
x = DenseVariational(15, kl_weight, **prior_params, activation='tanh')(x)
x = DenseVariational(1, kl_weight, **prior_params, activation='tanh')(x)

model = Model(x_in, x)

from keras import callbacks, optimizers

noise = 1.0

def neg_log_likelihood(y_obs, y_pred, sigma=noise):
    dist = tfp.distributions.Normal(loc=y_pred, scale=sigma)
    return K.sum(-dist.log_prob(y_obs))

model.compile(loss=neg_log_likelihood, optimizer=optimizers.Adam(lr=0.01), metrics=['accuracy','mse'])
history = model.fit(Values_array_train, Labels_array_train, batch_size=batch_size, epochs=150, verbose=1);

print(model.summary())

print('Plotting stuff')

# summarize history for accuracy
fig_acc = plt.figure(figsize=(12,8))
plt.plot(history.history['accuracy'],color="blue")
#plt.plot(history.history['val_acc'],color="red")
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
fig_acc.savefig('accuracy.png')

# summarize history for loss
fig_loss = plt.figure(figsize=(12,8))
plt.plot(history.history['loss'],color="blue")
#plt.plot(history.history['val_loss'],color="red")
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
fig_loss.savefig('loss.png')

predictions = model.predict(Values_array_test)

#ROC curve
fpr, tpr, _ = roc_curve(Labels_array_test, predictions)
roc_auc = auc(fpr, tpr)
fig_roc = plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.05])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic curve')
print('AUC: %f' % roc_auc)
fig_roc.savefig('ROC.png')

predictionsSig = []
predictionsBkg = []

for i in range(0,len(Labels_array_test)):
    if(Labels_array_test[i]==1):
        predictionsSig.append(predictions[i])
    if(Labels_array_test[i]==0):
        predictionsBkg.append(predictions[i])

predictionsSigPlot = np.asarray(predictionsSig)
predictionsBkgPlot = np.asarray(predictionsBkg)

fout = plt.figure(figsize=(12,8))
plt.hist(predictionsBkgPlot,label='Background',histtype="stepfilled",color="blue",normed=True,alpha=0.5)
plt.hist(predictionsSigPlot,label='Signal',histtype="stepfilled",color="red",normed=True,alpha=0.5)
plt.title("Deep Neural Network output in test sample")
plt.xlabel("Value")
plt.ylabel("Number of events")
#plt.set_yscale('log')
plt.legend(loc='upper right')
fout.savefig('output.png')
