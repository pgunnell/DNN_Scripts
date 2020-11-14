from root_numpy import root2array, tree2array, list_structures
from root_numpy import testdata
import numpy as np
import pandas

import keras
from keras.models import Sequential
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout, Activation, Flatten, Merge
from keras.layers import Conv2D, MaxPooling2D

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix,accuracy_score,average_precision_score,roc_auc_score

def normalize(vector, length):
    for i in range(0,length):
        mean = np.mean(vector[:,i])
        std = np.std(vector[:,i]) 

        if(std!=0):
            vector[:,i] = (vector[:,i] - mean)/std
    
    return vector
        
##################################

# JUST THESE FOLLOWING LINES TO BE MODIFIED

##################################

# Convert a TTree in a ROOT file into a NumPy structured array + pandas Dataframe, with which you can create an array to be given to Keras
branches_to_analyze = ["DNN_MET_eta","DNN_MET_phi","DNN_MET_pt","DNN_gluon1_eta","DNN_gluon1_phi","DNN_gluon1_pt","DNN_gluon2_eta","DNN_gluon2_phi","DNN_gluon2_pt","DNN_lepton_eta","DNN_lepton_phi","DNN_lepton_pt","DNN_leptopjet_eta","DNN_leptopjet_phi","DNN_leptopjet_pt","DNN_ttaggedjet_eta","DNN_ttaggedjet_phi","DNN_ttaggedjet_pt"]
save_numpy_format = False
epochs=100
number_hidden_nodes=[20,15,1]
number_layers=len(number_hidden_nodes)
activations=['tanh','tanh','tanh']
batch_size=512
filenameBkg = 'hadded/uhh2.AnalysisModuleRunner.MC.TTbar.root'
filenameSig = 'hadded/uhh2.AnalysisModuleRunner.MC.TstarTstar_M-1600.root'

#Try to include the mass

##########################################################################

arrBkg = pandas.DataFrame(root2array(filenameBkg, treename='AnalysisTree',branches=branches_to_analyze))
arrSig = pandas.DataFrame(root2array(filenameSig, treename='AnalysisTree',branches=branches_to_analyze))

#save the numpy format
if(save_numpy_format):
    outfileBkg = 'outBkg.npy'
    np.save(outfileBkg,arrBkg) #save as numpy

    outfileSig = 'outSig.npy'
    np.save(outfileSig,arrSig) #save as numpy

#create array for Keras implementation
Arr_Bkg = arrBkg.to_numpy()
Arr_Sig = arrSig.to_numpy()

Values_array = np.concatenate((Arr_Bkg,Arr_Sig),axis=0)

label_bkg = np.zeros(Arr_Bkg.shape[0])
label_sig = np.ones(Arr_Sig.shape[0])

Labels_array = np.concatenate((label_bkg,label_sig),axis=0)

Values_array = normalize(Values_array,len(branches_to_analyze))

print 'Variables setup'

#trying a NN implementation

#transforming arrays in float
Labels_array = Labels_array.astype(np.float)
Values_array = Values_array.astype(np.float)

print('We have number of background events equal to '+str(Arr_Bkg.shape[0]))
print('We have number of signal events equal to '+str(Arr_Sig.shape[0]))

#define train and test arrays for training
total_elements = Values_array.shape[0]

Labels_array_train=[]
Values_array_train=[]

Labels_array_test=[]
Values_array_test=[]

for i in range(0,total_elements/2):
    Labels_array_train.append(Labels_array[i*2])
    Values_array_train.append(Values_array[i*2,:])
    Labels_array_test.append(Labels_array[i*2+1])
    Values_array_test.append(Values_array[i*2+1,:])

Labels_array_train = np.asarray(Labels_array_train).astype(np.float)
Values_array_train = np.asarray(Values_array_train).astype(np.float)

Labels_array_test = np.asarray(Labels_array_test).astype(np.float)
Values_array_test = np.asarray(Values_array_test).astype(np.float)

print 'Defining the Keras model'

#play with the activation functions

model = Sequential()
model.add(Dense(5, input_dim=len(branches_to_analyze), activation='sigmoid'))
for i in range(0,number_layers):
    model.add(Dense(number_hidden_nodes[i], activation=activations[i]))

print 'Compiling the model'

# Compile model
model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])
# Fit the model (this is the training!!!)
history = model.fit(Values_array_train, Labels_array_train, validation_split=0.02,  shuffle = True, epochs=epochs, batch_size=batch_size)
#history = model.fit(Values_array_train, Labels_array_train, epochs=epochs, batch_size=batch_size)

print(model.summary())

##################################
#                                #  
#                                #  
#                                #  
#                                #  
#                                #  
##################################

print('Plotting stuff')

# summarize history for accuracy
fig_acc = plt.figure(figsize=(12,8))
plt.plot(history.history['acc'],color="blue")
#plt.plot(history.history['val_acc'],color="red")
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
fig_acc.savefig('accuracy.png')
#plt.show()

# summarize history for loss
fig_loss = plt.figure(figsize=(12,8))
plt.plot(history.history['loss'],color="blue")
#plt.plot(history.history['val_loss'],color="red")
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
fig_loss.savefig('loss.png')
#plt.show()

predictions = model.predict(Values_array_test)



#save model file
arch = model.to_json()
# save the architecture string to a file somehow, the below will work
with open('architecture.json', 'w') as arch_file:
    arch_file.write(arch)

    model.save("model_arch.h5")
    model.save_weights("model_weights.h5")
print("Saved model to disk")

import pickle

# write python dict to a file
mydict="{"
for i in range(1,len(branches_to_analyze)+1):
    mydict+="\""+str(branches_to_analyze[i-1])+"\": "+str(i)+", "

mydict+="}"                     
output_pkl = open('variables.pkl', 'wb')
pickle.dump(branches_to_analyze, output_pkl)
output_pkl.close()



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

#plotting the quantity we are using for training
for i in range(0,len(branches_to_analyze)):

    f = plt.figure(figsize=(12,8))
    plt.hist(Arr_Bkg[:,i],label='Background',histtype="stepfilled",color="blue",normed=True,alpha=0.5)
    plt.hist(Arr_Sig[:,i],label='Signal',histtype="stepfilled",color="red",normed=True,alpha=0.5)
    plt.xlabel(str(branches_to_analyze[i]))
    plt.ylabel('Entries')
    plt.legend(loc='upper right')
    plt.title('Plotting '+str(branches_to_analyze[i]))
    plt.grid(True)

    f.savefig('feat'+str(i)+'.png')

predictionsSig = []
predictionsBkg = []

for i in range(0,total_elements/2):
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


