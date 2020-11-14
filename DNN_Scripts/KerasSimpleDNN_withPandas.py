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

def define_masses(mass):

    files_mass_point = 'hadded/uhh2.AnalysisModuleRunner.MC.TstarTstar_M-'
    root_file = '.root'
    
    branches_to_analyze = ["run"]
    filename = files_mass_point+str(mass)+root_file
        
    Length = pandas.DataFrame(root2array(filename, treename='AnalysisTree',branches=branches_to_analyze))

    mass_value_array = np.full(len(Length), mass)

    return mass_value_array

def define_mass_bkg(filename):

    branches_to_analyze = ["run"]
    Masses = pandas.DataFrame(columns = ['Mass'])
    Length = pandas.DataFrame(root2array(filename, treename='AnalysisTree',branches=branches_to_analyze))
    mass_value_array = np.random.randint(200,2000,len(Length))
    
    return mass_value_array


def define_mass_notseen(filename,mass):

     branches_to_analyze = ["run"]
     Masses = pandas.DataFrame(columns = ['Mass'])
     Length = pandas.DataFrame(root2array(filename, treename='AnalysisTree',branches=branches_to_analyze))
     mass_value_array = np.full(len(Length), mass)

     return mass_value_array
                 
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
save_numpy_format = True
epochs=100
number_hidden_nodes=[30,15,1]
number_layers=len(number_hidden_nodes)
activations=['tanh','tanh','tanh']
batch_size=512
filenameBkg = 'hadded/uhh2.AnalysisModuleRunner.MC.TTbar.root'
filenameSig = 'hadded/uhh2.AnalysisModuleRunner.MC.TstarTstar_M-Combined.root'
filenameSig_notseen = 'hadded/uhh2.AnalysisModuleRunner.MC.TstarTstar_M-1200.root'
mass_notseen = 1200
train_parametric = True
validation_split = 0.2
split_train_test = 0.8

##########################################################################

arrBkg = pandas.DataFrame(root2array(filenameBkg, treename='AnalysisTree',branches=branches_to_analyze))
arrSig = pandas.DataFrame(root2array(filenameSig, treename='AnalysisTree',branches=branches_to_analyze))

#save the numpy format
if(save_numpy_format):
    outfileBkg = 'outBkg.npy'
    np.save(outfileBkg,arrBkg) #save as numpy

    outfileSig = 'outSig.npy'
    np.save(outfileSig,arrSig) #save as numpy

#adding a column for labels
label_bkg = np.zeros(arrBkg[branches_to_analyze[0]].count())
label_sig = np.ones(arrSig[branches_to_analyze[0]].count())

arrBkg.insert(len(branches_to_analyze), "Label", label_bkg, True)
arrSig.insert(len(branches_to_analyze), "Label", label_sig, True) 

if (train_parametric):

    mass_column_bkg = define_mass_bkg(filenameBkg)
    arrBkg.insert(len(branches_to_analyze), "Mass", mass_column_bkg, True)
    #arrBkg = pandas.merge(arrBkg, mass_column_bkg, right_index=True, left_index=True)
    print(arrBkg.head())

    masses = [700,800,900,1000,1100,1300,1400,1500,1600]
    masses_to_include = []
    
    for i in masses:    
        masses_to_include = np.concatenate((masses_to_include, define_masses(i)),axis=0)  

    masses_to_include = np.asarray(masses_to_include,dtype=int)
    print(masses_to_include.shape[0])
    print(len(arrSig))
    arrSig.insert(len(branches_to_analyze), "Mass", masses_to_include.T, True)
    print(arrSig.head())
    
#concatenate and shuffle the two dataframes
frames = [arrBkg, arrSig]
result = pandas.concat(frames)

#with sklearn:
#from sklearn.utils import shuffle
#df = shuffle(df)

result = result.sample(frac=1).reset_index(drop=True)

msk = np.random.rand(len(result)) < split_train_test
train_sample = result[msk]
test_sample = result[~msk]

#train_sample, test_sample = result[:, :dimension,:], result[:,dimension:,:]

train_sample = train_sample.sample(frac=1).reset_index(drop=True)
test_sample = test_sample.sample(frac=1).reset_index(drop=True)

#create array for Keras implementation
Values_array_train,Labels_array_train = get_label_and_values(train_sample)
Values_array_test,Labels_array_test = get_label_and_values(test_sample)

Values_array_train = normalize(Values_array_train,len(branches_to_analyze))
Values_array_test = normalize(Values_array_test,len(branches_to_analyze))

print 'Variables setup'

#trying a NN implementation

#transforming arrays in float
Labels_array_train = Labels_array_train.astype(np.float)
Values_array_train = Values_array_train.astype(np.float)

back_dim = len(result.loc[result['Label'] == 0])
sig_dim = len(result.loc[result['Label'] == 1])
print('We have number of background events equal to '+str(back_dim))
print('We have number of signal events equal to '+str(sig_dim))

print 'Defining the Keras model'

#play with the activation functions

model = Sequential()
input_dim = len(branches_to_analyze)
if(train_parametric): #because we need to add one element, the mass
    input_dim+=1

print('Neural network input dimension: '+str(input_dim))
model.add(Dense(30, input_dim=input_dim, activation='sigmoid'))
for i in range(0,number_layers):
    model.add(Dense(number_hidden_nodes[i], activation=activations[i]))

print 'Compiling the model'

# Compile model
model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])
# Fit the model (this is the training!!!)
history = model.fit(Values_array_train, Labels_array_train, validation_split=validation_split,  shuffle = True, epochs=epochs, batch_size=batch_size)
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
plt.plot(history.history['val_acc'],color="red")
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
fig_acc.savefig('accuracy.png')
#plt.show()

# summarize history for loss
fig_loss = plt.figure(figsize=(12,8))
plt.plot(history.history['loss'],color="blue")
plt.plot(history.history['val_loss'],color="red")
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
for i in branches_to_analyze:

    print(i)
    
    f = plt.figure(figsize=(12,8))
    plt.hist(result.loc[result['Label']==0, i],label='Background',histtype="stepfilled",color="blue",normed=True,alpha=0.5)
    plt.hist(result.loc[result['Label']==1, i],label='Signal',histtype="stepfilled",color="red",normed=True,alpha=0.5)
    plt.xlabel(str(i))
    plt.ylabel('Entries')
    plt.legend(loc='upper right')
    plt.title('Plotting '+str(i))
    plt.grid(True)

    f.savefig('feat'+str(i)+'.png')

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

#test the model in a not-seen mass point
arrSig_notseen = pandas.DataFrame(root2array(filenameSig_notseen, treename='AnalysisTree',branches=branches_to_analyze))
label_sig_notseen = np.ones(arrSig_notseen[branches_to_analyze[0]].count())

arrSig_notseen.insert(len(branches_to_analyze), "Label", label_sig_notseen, True)

if (train_parametric):
    
    mass_column_bkg = define_mass_notseen(filenameSig_notseen, mass_notseen)
    arrSig_notseen.insert(len(branches_to_analyze), "Mass", mass_column_bkg, True)

#concatenate and shuffle the two dataframes
frames_notseen = [arrBkg, arrSig_notseen]
result_notseen = pandas.concat(frames_notseen)
result_notseen = result_notseen.sample(frac=1).reset_index(drop=True)

#concatenate and shuffle the two dataframes
Values_array_notseen,Labels_array_notseen = get_label_and_values(result_notseen)

print('We are testing the model for '+str(Values_array_notseen.shape[0])+' events')

Values_array_notseen = normalize(Values_array_notseen,len(branches_to_analyze))
predictions_notseen = model.predict(Values_array_notseen)

fout = plt.figure(figsize=(12,8))
plt.hist(predictions_notseen,label='Signal',histtype="stepfilled",color="red",normed=True,alpha=0.5)
plt.title("Deep Neural Network output in test sample")
plt.xlabel("Value")
plt.ylabel("Number of events")
#plt.set_yscale('log')
plt.legend(loc='upper right')
fout.savefig('output_notseen.png')

#ROC curve
fpr, tpr, _ = roc_curve(Labels_array_notseen, predictions_notseen)
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
fig_roc.savefig('ROC_notseen.png')
