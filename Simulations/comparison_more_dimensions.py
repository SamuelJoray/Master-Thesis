#Plots from the Chapter Simulations in the Section "Higher Dimensions"
#Running time of this file can be extensively long depending on the number of epochs used and number of settings considered.
#It includes the following estimator implemented by myself: DeepCF.
#The following estimators are from previous work, they need to be imported before running this file, they can be found in the link mentionned. 
#DeepIV: https://github.com/py-why/EconML
#DeepGMM: https://github.com/CausalML/DeepGMM


import numpy as np
import matplotlib.pyplot as plt
import torch
from ..Deep_CF import DeepCF

import importlib
import keras
from econml.iv import nnet
importlib.reload(nnet)
from econml.iv.nnet import DeepIV
from methods.toy_model_selection_method import ToyModelSelectionMethod




def mse(pred,true):
    return np.mean((pred-true)**2)




def causal_effect(x):
    return abs(x[:,0])+abs(x[:,1])


def linear(i,n):
    Z = np.random.normal(0, 1, size=(n, i))
    H = np.random.normal(0, 1, n)   
    X = Z + np.tile(H.reshape(-1, 1), (1,i)) +np.random.normal(0, 1, size=(n, i))
    Y = np.dot(X,np.ones(i)) + H+np.random.normal(0, 1, n)
    return Z,X,Y


def factor(i,n):
    Z = np.random.normal(0, 1, size=(n, i))
    H = np.random.normal(0, 1, n)
    X = Z + np.tile(H.reshape(-1, 1), (1,i)) +np.random.normal(0, 1, size=(n, i))
    X_new=np.zeros(shape=np.shape(X))
    for j in range(X.shape[1] - 1):
        X_new[:, j] = X[:, j] * X[:, j + 1]
    X_new[:,-1] =X[:,-1]
    Y = np.dot(X_new,np.ones(i)) + H + np.random.normal(0, 1, n)
    return Z,X,Y



def simulate(Z,X,Y):
    
    
    treatment_model = keras.Sequential([keras.layers.Dense(64, activation='relu', input_shape=(2*np.shape(X)[1],)),
                                        #keras.layers.Dropout(0.17),
                                        keras.layers.Dense(32, activation='relu'),
                                        #keras.layers.Dropout(0.17),
                                       keras.layers.Dense(16, activation='relu'),
                                       keras.layers.Dropout(0.17)])
                                        
                                        
    response_model = keras.Sequential([keras.layers.Dense(64, activation='relu', input_shape=(2*np.shape(X)[1],)),#,bias_initializer=Constant(value=1)),
                                      # keras.layers.Dropout(0.17),
                                       keras.layers.Dense(32, activation='relu'),
                                       #keras.layers.Dropout(0.17),
                                       keras.layers.Dense(16, activation='relu'),
                                      keras.layers.Dense(1)])#last row is only one neuron with linear activation
    
    keras_fit_options1 = { "epochs": 100,
                          "validation_split": 0.1,
                          "verbose":0}
    
    keras_fit_options2 = { "epochs": 300,    
                          "validation_split": 0.1,
                          "verbose":0}
    
    
    
    estDeepIV = DeepIV(n_components=2, # Number of gaussians in the mixture density networks
                 m=lambda z, x: treatment_model(keras.layers.concatenate([z, x])), # Treatment model 
                 h=lambda t, x: response_model(keras.layers.concatenate([t, x])), # Response model
                 n_samples=1,# Number of samples used to estimate the response
                 #use_upper_bound_loss = False, # whether to use an approximation to the true loss
                 n_gradient_samples = 1, # number of samples to use in second estimate of the response (to make loss estimate unbiased)
                 optimizer='adam', # Keras optimizer to use for training - see https://keras.io/optimizers/ 
                 first_stage_options = keras_fit_options1, # options for training treatment model
                 second_stage_options = keras_fit_options2) # options for training response model
    
    
    estDeepGMM = ToyModelSelectionMethod(dim=np.shape(X)[1])
    
    estDeepCF = DeepCF()
    
    
    W=np.zeros(np.shape(X))
    estDeepIV.fit(Y=Y, T=X, X=W , Z=Z)
    
    
    train_size = int(n*0.9)
    Z_train = Z[:train_size,:]
    X_train = X[:train_size,:]
    Y_train = Y[:train_size].reshape(-1, 1)
    
    Z_val = Z[train_size:,:]
    X_val = X[train_size:,:]
    Y_val = Y[train_size:].reshape(-1, 1)
    estDeepGMM.fit(torch.from_numpy(X_train), torch.from_numpy(Z_train), torch.from_numpy(Y_train).reshape(-1, 1), 
                torch.from_numpy(X_val), torch.from_numpy(Z_val), torch.from_numpy(Y_val).reshape(-1, 1), 
                g_dev=None, verbose=True)
    
    estDeepCF.fit(Z,X,Y,W)
    
    
    return estDeepCF,estDeepIV,estDeepGMM
    
    
    
    #example with linear function for the treatment
n=2000
mse_deepIV_linear=[]
mse_deepGMM_linear=[]
mse_deepCF_linear=[]
nDim=[2,3,5,8,12,20,30,50,75,100]

for i in nDim:
    Z,X,Y = linear(i,n)
    estDeepCF,estDeepIV,estDeepGMM = simulate(Z,X,Y)
    
    n_test = 1000
    X_test = np.random.normal(0, 1, size=(n, i))
    W_test = np.zeros(np.shape(X))
    
    
    pred_deepIV = estDeepIV.predict(T=X_test,X=W_test)
    pred_deepGMM = estDeepGMM.predict(torch.from_numpy(X_test)).flatten().detach().numpy()
    pred_deepCF = estDeepCF.predict(X_test,W_test,smoothing_param=200,second_stage_additive=True).flatten()
    
    
    mse_deepIV_linear.append( mse(pred_deepIV,np.dot(X_test,np.ones(i)) ))
    mse_deepGMM_linear.append(mse(pred_deepGMM,np.dot(X_test,np.ones(i))))
    mse_deepCF_linear.append(mse(pred_deepCF,np.dot(X_test,np.ones(i))))
    
    print(i)
    
    
plt.plot(nDim,mse_deepCF_linear, color="red", linewidth=2, label="DeepCF", linestyle="-", marker="D")
plt.plot(nDim,mse_deepGMM_linear, color="green", linewidth=2, label="DeepGMM", linestyle="-", marker="D")
plt.plot(nDim,mse_deepIV_linear, color="blue", linewidth=2, label="DeepIV", linestyle="-", marker="D")

plt.ylim(0,45)
plt.xscale("log")
plt.legend(loc="upper right")
plt.xlabel("Number of dimensions")
plt.ylabel("Mean Squared Error")
plt.grid(True)
#Change the following path to save the plot as a pdf
#plt.savefig("PATH\simnDimLinear.pdf", bbox_inches='tight',pad_inches =0)

plt.show()
    

#Example with nonlinear function for the treatment
mse_deepIV_factor=[]
mse_deepGMM_factor=[]
mse_deepCF_factor=[]
mse_GCFN_factor=[]
nDim=[2,3,5,8,12,20,30,50,75,100]
for i in nDim:
    Z,X,Y = factor(i,n)
    estDeepCF,estDeepIV,estDeepGMM= simulate(Z,X,Y)
    
    n_test = 2000
    X_test = np.random.normal(0, 1, size=(n, i))
    W_test=np.zeros(np.shape(X))
    
    pred_deepIV = estDeepIV.predict(T=X_test,X=W_test)
    pred_deepGMM = estDeepGMM.predict(torch.from_numpy(X_test)).flatten().detach().numpy()
    pred_deepCF = estDeepCF.predict(X_test,W_test,smoothing_param=200,second_stage_additive=True).flatten()
    
    
    
    
    X_new=np.zeros(shape=np.shape(X_test))
    for j in range(X.shape[1] - 1):
        X_new[:, j] = X_test[:, j] * X_test[:, j + 1]
    X_new[:,-1] =X_test[:,-1]
    
    mse_deepIV_factor.append( mse(pred_deepIV,np.dot(X_test,np.ones(i))))
    
    mse_deepGMM_factor.append(mse(pred_deepGMM,np.dot(X_new,np.ones(i))))

    mse_deepCF_factor.append(mse(pred_deepCF,np.dot(X_new,np.ones(i))))
    
print("mse_deepGMM_factor",mse_deepGMM_factor)

print("mse_deepCF_factor",mse_deepCF_factor)
print("mse_deepIV_factor",mse_deepIV_factor)

plt.plot(nDim,mse_deepCF_factor, color="red", linewidth=2, label="DeepCF", linestyle="-", marker="D")
plt.plot(nDim,mse_deepGMM_factor, color="green", linewidth=2, label="DeepGMM", linestyle="-", marker="D")
plt.plot(nDim,mse_deepIV_factor, color="blue", linewidth=2, label="DeepIV", linestyle="-", marker="D")

plt.ylim(0,10)
plt.xscale("log")
plt.legend(loc="upper right")
plt.xlabel("Number of dimensions")
plt.ylabel("Mean Squared Error")

plt.grid(True) 
#Change the following path to save the plot as a pdf
#plt.savefig("PATH\simnDimFactorZoom.pdf", bbox_inches='tight',pad_inches =0)

plt.show()
