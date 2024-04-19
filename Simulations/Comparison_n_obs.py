#Plots from the Chapter Simulations in the Section "Mean Squared Error"
#Running time of this file can be extensively long depending on the number of epochs used and number of settings considered.
#It includes the following estimator implemented by myself: CFSpline,Spline2sls,Spline and DeepCF.
#The following estimators are from previous work, they need to be imported before running this file, they can be found in the link mentionned. 
#DeepIV: https://github.com/py-why/EconML
#DeepGMM: https://github.com/CausalML/DeepGMM



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from patsy import dmatrix

import torch
import keras
import importlib

#importing DeepIV
from econml.iv import nnet
importlib.reload(nnet)
from econml.iv.nnet import DeepIV

#importing DeepGMM
from methods.toy_model_selection_method import ToyModelSelectionMethod

#importing GCFN
from ..gcfn import GCFN

from ..spline2sls import Spline2sls
from ..spline import Spline
from ..Deep_CF import DeepCF
from ..Control_Function_spline import CFSpline



def mse(pred,true):
    return np.mean((pred-true)**2)




def generate_data(i,n,n_test):
    if i==0:
        Z = np.random.normal(0, 2, n)
        H = np.random.normal(0, 1, n)
        X = Z + 2*H + np.random.normal(0,0.5,n)
        Y = abs(X)  +  2*H + np.random.normal(0, 0.5, n)
        X_test = np.linspace(-6,6,n_test)
        causal_effect = abs(X_test)
        
    if i==1:
        Z = np.random.uniform(-5,5, n)
        H = np.random.uniform(-3,3, n)
        X = Z + H + np.random.normal(0,1,n)
        Y = 0.1*(X-5)*(X+5)*X + 2*H + np.random.normal(0, 1, n)
        X_test = np.random.uniform(-6,6,n_test)
        causal_effect=0.1*(X_test-5)*(X_test+5)*X_test
        
        
    return Z,X,Y,X_test,causal_effect

def simulate(n_observations,nb_experiments,nb_estimators,n_test,nb_rep):
    mse_est=np.zeros((nb_experiments,len(n_observations),nb_rep,nb_estimators))
    
    for i in range(nb_experiments):
        for j,n in enumerate(n_observations):
            for k in range(nb_rep):
                #generate the data
                Z,X,Y,X_test,causal_effect = generate_data(i+1,n,n_test)
                
    
                #train estimators
                est2sls.fit(Z, X, Y)
                estCF.fit(Z, X, Y)
                #naive_spline.fit(X,Y)
                estDeepIV.fit(Y=Y, T=X, X=np.zeros(n) , Z=Z)
                estDeepCF.fit(Z,X,Y,W=np.zeros(n))
                estGCFN.fit(X,Y,Z)
                
                #DeepGMM
                train_size = int(n*0.9)
                Z_train = Z[:train_size].reshape(-1, 1)
                X_train = X[:train_size].reshape(-1, 1)
                Y_train = Y[:train_size].reshape(-1, 1)
    
                Z_val = Z[train_size:].reshape(-1, 1)
                X_val = X[train_size:].reshape(-1, 1)
                Y_val = Y[train_size:].reshape(-1, 1)
                estDeepGMM.fit(torch.from_numpy(X_train).reshape(-1, 1), torch.from_numpy(Z_train).reshape(-1, 1), torch.from_numpy(Y_train).reshape(-1, 1), 
                           torch.from_numpy(X_val).reshape(-1, 1), torch.from_numpy(Z_val).reshape(-1, 1), torch.from_numpy(Y_val).reshape(-1, 1), 
                           g_dev=None, verbose=True)
            
                pred=np.zeros((n_test,nb_estimators))
                pred[:,0]= est2sls.predict(X_test)
                pred[:,1] = estCF.predict(X_test)
                pred[:,2] = estDeepCF.predict(X_test,W=np.zeros(n_test),smoothing_param=200,second_stage_additive=True).flatten()
                pred[:,3] = estDeepIV.predict(T=X_test,X=np.zeros(n_test))
                pred[:,4] = estDeepGMM.predict(torch.from_numpy(X_test)).flatten().detach().numpy()
                pred[:,5] = estGCFN.predict(X_test)
    
    
                
                for l in range(nb_estimators):
                    mse_est[i,j,k,l] = mse(pred[:,l],causal_effect)
                
            
    return mse_est

   


#define estimators

treatment_model = keras.Sequential([keras.layers.Dense(64, activation='relu', input_shape=(2,)),
                                    #keras.layers.Dropout(0.17),
                                    keras.layers.Dense(32, activation='relu'),
                                    #keras.layers.Dropout(0.17),
                                   keras.layers.Dense(16, activation='relu'),
                                   keras.layers.Dropout(0.17)])
                                    
                                    
response_model = keras.Sequential([keras.layers.Dense(64, activation='relu', input_shape=(2,)),#,bias_initializer=Constant(value=1)),
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



estDeepIV = DeepIV(n_components=2, # Number of gaussians in the mixture density networks)
             m=lambda z, x: treatment_model(keras.layers.concatenate([z, x])), # Treatment model
             h=lambda t, x: response_model(keras.layers.concatenate([t, x])), # Response model
             n_samples=1,# Number of samples used to estimate the responseuse_upper_bound_loss = False, # whether to use an approximation to the true loss
             n_gradient_samples = 1, # number of samples to use in second estimate of the response (to make loss estimate unbiased)
             optimizer='adam', # Keras optimizer to use for training - see https://keras.io/optimizers/ 
             first_stage_options = keras_fit_options1, # options for training treatment model
             second_stage_options = keras_fit_options2) # options for training response model



df = 10
df_first_stage=10
df_second_stage=10

est2sls = Spline2sls(df_second_stage,df_first_stage)
estCF = CFSpline(df)
estDeepCF = DeepCF()
naive_spline = Spline(df)
estDeepGMM = ToyModelSelectionMethod()
estGCFN = GCFN()


n_observations = [150,200,300]#,500,1000,2000,5000]
nb_experiments=1
nb_estimators = 7
n_test =2000
rangey=[50,2,1.5]
nb_rep=1

mse_est = simulate(n_observations,nb_experiments,nb_estimators,n_test,nb_rep)
titles = ["1st Simulation","2nd Simulation","3rd Simulation"]

mean_pred = np.mean(mse_est, axis=2)
std_pred= np.std(mse_est, axis=2)
z_score = 1.96  # For 95% confidence interval
confidence_interval = z_score * (std_pred / np.sqrt(nb_rep))

for i in range(nb_experiments):
    plt.plot(n_observations,mean_pred[i,:,0], color="red", linewidth=2, label="2sls", linestyle="-", marker="o")
    plt.fill_between(n_observations, 
                     mean_pred[i,:,0] - confidence_interval[i,:,0], 
                     mean_pred[i,:,0] + confidence_interval[i,:,0],color="red", alpha=0.3)
    
    plt.plot(n_observations,mean_pred[i,:,1], color="black", linewidth=2, label="CF", linestyle="-", marker="o")
    plt.fill_between(n_observations, 
                     mean_pred[i,:,1] - confidence_interval[i,:,1], 
                     mean_pred[i,:,1] + confidence_interval[i,:,1],color="black", alpha=0.3)
    
    plt.plot(n_observations,mean_pred[i,:,2], color="purple", linewidth=2, label="DeepCF", linestyle="-", marker="o")
    plt.fill_between(n_observations, 
                     mean_pred[i,:,2] - confidence_interval[i,:,2], 
                     mean_pred[i,:,2] + confidence_interval[i,:,2],color="purple", alpha=0.3)
    
    plt.plot(n_observations,mean_pred[i,:,3], color="orange", linewidth=2, label="DeepIV", linestyle="-", marker="o")
    plt.fill_between(n_observations, 
                     mean_pred[i,:,3] - confidence_interval[i,:,3], 
                     mean_pred[i,:,3] + confidence_interval[i,:,3],color="orange", alpha=0.3)
    
    plt.plot(n_observations,mean_pred[i,:,4], color="brown", linewidth=2, label="DeepGMM", linestyle="-", marker="o")
    plt.fill_between(n_observations, 
                     mean_pred[i,:,4] - confidence_interval[i,:,4], 
                     mean_pred[i,:,4] + confidence_interval[i,:,4],color="brown", alpha=0.3)
    
    plt.plot(n_observations,mean_pred[i,:,5], color="green", linewidth=2, label="GCFN", linestyle="-", marker="o")
    plt.fill_between(n_observations, 
                    mean_pred[i,:,5] - confidence_interval[i,:,5], 
                     mean_pred[i,:,5] + confidence_interval[i,:,5],color="green", alpha=0.3)
    
    
    plt.ylim(0,rangey[i])
    plt.xscale("log")
    plt.legend(loc="upper right")
    plt.xlabel("Number of observations")
    plt.ylabel("Mean Squared Error")
    plt.grid(True)
    #Change the following path to save the plot as a pdf
    #plt.savefig("PATH\simnObsZoom"+str(i)+".pdf")

    plt.show()
    
