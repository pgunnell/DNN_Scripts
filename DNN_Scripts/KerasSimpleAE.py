import numpy as np
import pandas as pd

import keras
from keras.models import Sequential
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

from keras.models import Model
from keras.layers import Input, Dense

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix,accuracy_score,average_precision_score,roc_auc_score,mean_squared_error,log_loss

def normalize(vector, length):
    for i in range(0,length):
        mean = np.mean(vector[:,i])
        std = np.std(vector[:,i]) 

        if(std!=0):
            vector[:,i] = (vector[:,i] - mean)/std
    
    return vector

def normalize_to_integral(vector,length):
    for i in range(0,length):
        integral = np.sum(vector[:,i])
                
        if(integral!=0):
            vector[:,i] = vector[:,i]/integral
            
    return vector

from keras import backend as K,objectives
from keras.losses import mse

def ae_loss(input_img, output):

    # Compute error in reconstruction
    reconstruction_loss = mse(K.flatten(input_img) , K.flatten(output))
    
    # Return the average loss over all images in batch
    total_loss = reconstruction_loss
    return total_loss

from sklearn.decomposition import PCA

def PCA_analysis(input_vector):
    pca = PCA(n_components=18)
    pca.fit(input_vector)
    pca_score = pca.explained_variance_ratio_
    V = pca.components_

    #this is the rotation matrix
    axis = len(input_vector) * V.T

    #input_vector_rotated = pca.transform(input_vector) 

    return pca

def define_ROC(loss_function):

    efficiency = []

    total_integral = 0
    for i in loss_function:
        total_integral+=i
    
    efficiency_point_definition = [0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0]
    for efficiency_point in efficiency_point_definition:
        integral=0
        for i in loss_function:
            if(i>efficiency_point):
                integral+=i

        efficiency.append(integral/total_integral)

    return efficiency
        
    
##################################

# JUST THESE FOLLOWING LINES TO BE MODIFIED

##################################

# Convert a TTree in a ROOT file into a NumPy structured array + pandas Dataframe, with which you can create an array to be given to Keras
branches_to_analyze = ["DNN_MET_eta","DNN_MET_phi","DNN_MET_pt","DNN_gluon1_eta","DNN_gluon1_phi","DNN_gluon1_pt","DNN_gluon2_eta","DNN_gluon2_phi","DNN_gluon2_pt","DNN_lepton_eta","DNN_lepton_phi","DNN_lepton_pt","DNN_leptopjet_eta","DNN_leptopjet_phi","DNN_leptopjet_pt","DNN_ttaggedjet_eta","DNN_ttaggedjet_phi","DNN_ttaggedjet_pt"]
#branches_to_analyze = ["DNN_lepton_pt","DNN_leptopjet_pt","DNN_ttaggedjet_pt","DNN_gluon1_pt","DNN_gluon2_pt","DNN_MET_pt","DNN_gluon2_phi","DNN_lepton_phi"]
save_numpy_format = False
epochs=200
number_hidden_nodes=[20,15,1]
number_layers=len(number_hidden_nodes)
activations=['tanh','tanh','tanh']
batch_size=128
filenameBkg = 'hadded/uhh2.AnalysisModuleRunner.MC.TTbar.root'
filenameSig = 'hadded/uhh2.AnalysisModuleRunner.MC.TstarTstar_M-Combined.root'
split_train_test = 0.7
z_mean = 1.0
z_sigma = 0.1
reduced_training = True
reduced_dimension = 10

##########################################################################

outfileBkg = 'outBkg.npy'
outfileSig = 'outSig.npy'

Arr_Bkg = np.load(outfileBkg)
Arr_Sig = np.load(outfileSig)

arrBkg = pd.DataFrame(data=Arr_Bkg)
arrSig = pd.DataFrame(data=Arr_Sig)

msk = np.random.rand(len(arrBkg)) < split_train_test
train_sample = arrBkg[msk]
test_sample = arrBkg[~msk]
    
#create array for Keras implementation
train_sample = train_sample.to_numpy()
test_sample = test_sample.to_numpy()
Arr_Sig = arrSig.to_numpy()
Arr_Bkg = arrBkg.to_numpy()

#VERY IMPORTANT! One does need to normalize the features, otherwise training is super difficult!
train_sample = normalize(train_sample,len(branches_to_analyze))
test_sample = normalize(test_sample,len(branches_to_analyze))
Arr_Sig = normalize(Arr_Sig,len(branches_to_analyze))
Arr_Bkg = normalize(Arr_Bkg,len(branches_to_analyze))

print('Variables setup')

print('We have number of background events equal to '+str(train_sample.shape[0]))
print('We have number of signal events equal to '+str(Arr_Sig.shape[0]))

print('Defining the Keras model')

#play with the activation functions

# this is the size of our encoded representations
encoding_dim = 5  

# this is our input placeholder
# Fit the model (this is the training!!!)
input_dimension = len(branches_to_analyze)

train_sample_1 = []
test_sample_1 = []

if(reduced_training):
    train_sample = PCA_analysis(train_sample).transform(train_sample)
    test_sample = PCA_analysis(test_sample).transform(test_sample)
    Arr_Sig = PCA_analysis(train_sample).transform(Arr_Sig)
    Arr_Bkg = PCA_analysis(train_sample).transform(Arr_Bkg)
    
    train_sample = train_sample[:,:reduced_dimension] 
    test_sample = test_sample[:,:reduced_dimension]
    input_dimension = reduced_dimension
    Arr_Sig = Arr_Sig[:,:reduced_dimension]
    Arr_Bkg = Arr_Bkg[:,:reduced_dimension]
    
input_img = Input(shape=(input_dimension,))
encoded = Dense(20, activation='relu')(input_img)
encoded = Dense(30, activation='relu')(encoded)
encoded = Dense(5, activation='tanh')(encoded)

decoded = Dense(20, activation='relu')(encoded)
decoded = Dense(40, activation='relu')(decoded)
decoded = Dense(20, activation='relu')(decoded)
decoded = Dense(input_dimension, activation='linear')(decoded)

autoencoder = Model(input_img, decoded)
# Compile model
print('Compiling the model')
autoencoder.compile(optimizer=keras.optimizers.Adam(), loss=ae_loss)
history = autoencoder.fit(train_sample, train_sample, epochs=epochs, batch_size=batch_size, shuffle=True, validation_data=(test_sample, test_sample))

print(autoencoder.summary())

##################################
#                                #  
#                                #  
#                                #  
#                                #  
#                                #  
##################################

print('Plotting stuff')

# summarize history for loss
fig_loss = plt.figure(figsize=(12,8))
plt.plot(history.history['loss'],color="blue")
plt.plot(history.history['val_loss'],color="red")
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
fig_loss.savefig('loss.png')

#Here I try to plot the loss for signal and for background events
predictions_Sig = autoencoder.predict(Arr_Sig)
predictions_Bkg = autoencoder.predict(Arr_Bkg)

loss_signal = []
loss_bkg = []

for i in range(0,len(predictions_Sig)):
    loss_signal.append(mean_squared_error(Arr_Sig[i,], predictions_Sig[i,]))

for i in range(0,len(predictions_Bkg)):
    loss_bkg.append(mean_squared_error(Arr_Bkg[i,], predictions_Bkg[i,]))

# summarize history for loss
fig_loss = plt.figure(figsize=(12,8))
plt.hist(loss_signal,color="blue",histtype="stepfilled",normed=True,alpha=0.5,bins=100)
plt.hist(loss_bkg,histtype="stepfilled",color="red",normed=True,alpha=0.5,bins=100)
plt.title('model loss')
plt.yscale('log')
plt.ylabel('Events')
plt.xlabel('loss')
plt.legend(['signal', 'background'], loc='upper left')
fig_loss.savefig('loss_sigbkg.png')

#plot two distributions from signal true and predicted to see how they look like

for i in range(0, input_dimension):

    print(i)
    
    f = plt.figure(figsize=(12,8))
    plt.hist(predictions_Bkg[:,i],color="blue",histtype="stepfilled",normed=True,alpha=0.5)
    plt.hist(Arr_Bkg[:,i],histtype="stepfilled",color="red",normed=True,alpha=0.5)
    plt.xlabel(str(branches_to_analyze[i]))
    plt.ylabel('Entries')
    plt.legend(['predicted', 'true'], loc='upper left')
    plt.title('Plotting '+str(i))
    plt.grid(True)
    
    f.savefig('predictions_bkg_'+str(branches_to_analyze[i])+'.png')                                           

    fig_loss = plt.figure(figsize=(12,8))
    plt.hist(predictions_Sig[:,i],color="blue",histtype="stepfilled",normed=True,alpha=0.5)
    plt.hist(Arr_Sig[:,i],histtype="stepfilled",color="red",normed=True,alpha=0.5)
    plt.title('Predictions for observable 1')
    #plt.yscale('log')
    plt.ylabel('Entries')
    plt.xlabel(str(branches_to_analyze[i]))
    plt.legend(['predicted', 'true'], loc='upper left')
    fig_loss.savefig('predictions_sig_'+str(branches_to_analyze[i])+'.png')

#save model file
arch = autoencoder.to_json()
# save the architecture string to a file somehow, the below will work
with open('architectureAE.json', 'w') as arch_file:
    arch_file.write(arch)

    autoencoder.save("modelAE_arch.h5")
    autoencoder.save_weights("modelAE_weights.h5")
print("Saved model to disk")

import pickle

# write python dict to a file
mydict="{"
for i in range(1,len(branches_to_analyze)+1):
    mydict+="\""+str(branches_to_analyze[i-1])+"\": "+str(i)+", "

mydict+="}"                     
output_pkl = open('variablesAE.pkl', 'wb')
pickle.dump(branches_to_analyze, output_pkl)
output_pkl.close()

signal_efficiency = define_ROC(loss_signal)
print(signal_efficiency)

bkg_efficiency = define_ROC(loss_bkg)
print(bkg_efficiency)
