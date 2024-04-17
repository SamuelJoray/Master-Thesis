#Plots used in the Chapter "Simulations" in Section "Estimated Mean Function".
#Running time of this file can be extensively long depending on the number of epochs used and number of settings considered.
#It includes the following estimators: CFSpline,Spline2sls,Spline and DeepCF.
#The following estimators are from previous work, they need to be imported before running this file, they can be found in the link mentionned. 
#DeepIV: https://github.com/py-why/EconML
#DeepGMM: https://github.com/CausalML/DeepGMM
#GCFN: https://github.com/rajesh-lab/gcfn-code



import numpy as np
import matplotlib.pyplot as plt
from spline2sls import Spline2sls
from spline import Spline
from Deep_CF import DeepCF
from Control_Function_spline import CFSpline


import torch
import keras

#importing DeepIV
import importlib
from econml.iv import nnet
importlib.reload(nnet)
from econml.iv.nnet import DeepIV

#importing DeepGMM
from methods.toy_model_selection_method import ToyModelSelectionMethod

#importing gcfn
from gcfn.gcfn import GCFN

def mse(pred,true):
    return np.mean((pred-true)**2)


    


def generate_data(i,n,n_test,xlim_min,xlim_max):
    if i==0:
        Z = np.random.normal(0, 4, n)
        H = np.random.normal(0, 2, n)
        X = Z + H + np.random.normal(0,1,n)
        Y = abs(X)  +  H + np.random.normal(0, 1, n)
        X_test = np.linspace(min(X), max(X),n_test) 
        causal_effect = abs(X_test)
        
    if i == 1: 
        Z = np.random.normal(0,4,n)
        H = np.random.normal(0,2, n)
        X = Z + H + np.random.normal(0,1,n)
        Y = 0.25*X  + H + np.random.normal(0, 1, n)
        X_test =np.linspace(min(X), max(X),n_test)
        causal_effect=0.25*X_test
        
    if i==2:
        Z = np.random.normal(0,4,n)
        H = np.random.normal(0,2, n)
        X = Z + H + np.random.normal(0,1,n)
        Y = np.sign(X)  + H+ np.random.normal(0, 1, n)
        X_test =np.linspace(min(X), max(X),n_test)
        causal_effect=np.sign(X_test)
        
    if i==3:
        H =np.random.uniform(-1,1, n)
        Z = np.random.uniform(-5,5, n)
        X = Z  + H 
        Y =np.cos(X) + H
        X_test = np.linspace(min(X), max(X),n_test)
        causal_effect=np.cos(X_test)
        
    if i==4:
        H =np.random.uniform(-4,4, n)
        Z = np.random.uniform(-5,5, n)
        X = Z  + H 
        Y =np.cos(X) + H
        X_test = np.linspace(min(X), max(X),n_test)
        causal_effect=np.cos(X_test)
    if i==5:
        H =np.random.uniform(-7,7, n)
        Z = np.random.uniform(-5,5, n)
        X = Z  + H 
        Y =np.cos(X) + H
        X_test = np.linspace(min(X), max(X),n_test)
        causal_effect=np.cos(X_test)
        
        
    if i==6:
        H = np.random.normal(0,2, n)
        Z = np.random.normal(0,4,n)
        X = 0.25*Z + H + np.random.normal(0,1,n)
        Y = X  + X*abs(H)-X*np.mean(abs(H))+np.random.normal(0,1,n)
        X_test = np.linspace(xlim_min[i], xlim_max[i],n_test)
        causal_effect=X_test
    if i==7:
        H = np.random.normal(0,2, n)
        Z = np.random.normal(0,4,n)
        X = 0.25*Z + 0.5*H + np.random.normal(0,1,n)
        Y = X  + abs(X)*H+np.random.normal(0,1,n)
        X_test = np.linspace(xlim_min[i], xlim_max[i],n_test)
        causal_effect=X_test
    if i==8:
        Z = np.random.normal(0,2,n)
        H = np.random.normal(0,2, n)
        X = Z  +Z*H+H+ np.random.normal(0,1,n)
        Y = np.sin(X)  + H  + np.random.normal(0, 1, n)
        X_test = np.linspace(xlim_min[i], xlim_max[i],n_test)
        causal_effect=np.sin(X_test)
   
        
    return Z,X,Y,X_test,causal_effect

def simulateAndPlot(n,experiments,nb_estimators,n_test,titles,nb_rep,ylim_min,ylim_max,xlim_min,xlim_max,xticks,yticks):
    mse_pred=np.zeros((len(experiments),nb_estimators,nb_rep))
    pred=np.zeros((n_test,nb_estimators,nb_rep))
    for i in experiments:
        for j in range(nb_rep):
            
            #generate the data
            Z,X,Y,X_test,causal_effect = generate_data(i,n,n_test,xlim_min,xlim_max)
            
    
            #train estimators
            est2sls.fit(Z, X, Y)
            estCF.fit(Z, X, Y)
            naive_spline.fit(X,Y)
            estDeepIV.fit(Y=Y, T=X, X=np.zeros(n) , Z=Z)
            estDeepCF.fit(Z,X,Y,W=np.zeros(n))
            #estGCFN.fit(X,Y,Z)
            
            
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
            
        
            
            pred[:,0,j]=est2sls.predict(X_test)
            pred[:,1,j] = estCF.predict(X_test)
            pred[:,2,j] = naive_spline.predict(X_test)
            pred[:,3,j] = estDeepCF.predict(X_test,W=np.zeros(n_test),smoothing_param=200).flatten()
            pred[:,4,j] = estDeepIV.predict(T=X_test,X=np.zeros(n_test))
            pred[:,5,j] = estDeepGMM.predict(torch.from_numpy(X_test)).flatten().detach().numpy()
            pred[:,6,j] = estGCFN.predict(X_test)
            
        
            print(j)
            
     
        mean_pred = np.mean(pred, axis=2)
        std_pred= np.std(pred, axis=2)
        z_score = 1.96  # For 95% confidence interval
        confidence_interval = z_score * (std_pred / np.sqrt(nb_rep))
        
       
    
        sorted_indices_test = np.argsort(X_test)
        X_test_sorted = X_test[sorted_indices_test]
        
        mean_pred_sorted = mean_pred[sorted_indices_test,:]
        conf_int_sorted = confidence_interval[sorted_indices_test,:]
        
        #plot all estimators, one plot for one estimator
        plt.scatter(X, Y, marker='o', s=0.25, label="data")
        plt.plot(X_test_sorted, causal_effect[sorted_indices_test], color="green", linewidth=2, label="True")
        plt.plot(X_test_sorted, mean_pred_sorted[:,2], color="black", linewidth=2, label="Naive")
        plt.xlabel("X",fontsize=22)
        plt.xlim(xlim_min[i],xlim_max[i])
        plt.ylim(ylim_min[i],ylim_max[i])
        plt.ylabel("Y",fontsize=22,rotation = 0)
        plt.xticks(xticks[i,:])
        plt.xticks(fontsize=22)
        plt.yticks(yticks[i,:])
        plt.yticks(fontsize=22)
        plt.gcf().set_size_inches(4, 4) 
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        #Change the following path to save the plot as a pdf
        plt.savefig("PATH\simMeant0"+str(i)+".pdf", bbox_inches='tight',pad_inches =0)
        plt.show()
        
        #plot the average causal effect
        plt.plot(X_test_sorted, mean_pred_sorted[:,0], color="black", linewidth=2, label="2SLS")
        #plot the true causal effect
        plt.plot(X_test_sorted, causal_effect[sorted_indices_test], color="green", linewidth=2, label="True")
        #plot the 95% confidence region
        plt.fill_between(X_test_sorted, 
                         mean_pred_sorted[:,0] - conf_int_sorted[:,0], 
                         mean_pred_sorted[:,0] + conf_int_sorted[:,0],color="black", alpha=0.3)
        plt.xlabel("X",fontsize=22)
        plt.xlim(xlim_min[i],xlim_max[i])
        plt.ylim(ylim_min[i],ylim_max[i])
        plt.xticks(xticks[i,:])
        plt.xticks(fontsize=22)
        plt.yticks([])
        plt.gcf().set_size_inches(4, 4) 
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        #Change the following path to save the plot as a pdf
        #plt.savefig("PATH\simMeant1"+str(i)+".pdf", bbox_inches='tight',pad_inches =0)
        plt.show()
        
        plt.plot(X_test_sorted, mean_pred_sorted[:,1], color="black", linewidth=2, label="CF")
        plt.plot(X_test_sorted, causal_effect[sorted_indices_test], color="green", linewidth=2, label="True")
        plt.fill_between(X_test_sorted, 
                         mean_pred_sorted[:,1] - conf_int_sorted[:,1], 
                         mean_pred_sorted[:,1] + conf_int_sorted[:,1],color="black", alpha=0.3)
        plt.xlabel("X",fontsize=22)
        plt.xlim(xlim_min[i],xlim_max[i])
        plt.ylim(ylim_min[i],ylim_max[i])
        plt.xticks(xticks[i,:])
        plt.xticks(fontsize=22)
        plt.yticks([])
        plt.gcf().set_size_inches(4, 4) 
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        #Change the following path to save the plot as a pdf
        #plt.savefig("PATH\simMeant2"+str(i)+".pdf", bbox_inches='tight',pad_inches =0)
        plt.show()
        
        
        
        plt.plot(X_test_sorted, mean_pred_sorted[:,3], color="black", linewidth=2, label="DeepCF")
        plt.plot(X_test_sorted, causal_effect[sorted_indices_test], color="green", linewidth=2, label="True")
        plt.fill_between(X_test_sorted, 
                         mean_pred_sorted[:,3] - conf_int_sorted[:,3], 
                         mean_pred_sorted[:,3] + conf_int_sorted[:,3],color="black", alpha=0.3)
        plt.xlabel("X",fontsize=22)
        plt.xlim(xlim_min[i],xlim_max[i])
        plt.ylim(ylim_min[i],ylim_max[i])
        plt.xticks(xticks[i,:])
        plt.xticks(fontsize=22)
        plt.yticks([])
        plt.gcf().set_size_inches(4, 4) 
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        #Change the following path to save the plot as a pdf
        #plt.savefig("PATH\simMeant4"+str(i)+".pdf", bbox_inches='tight',pad_inches =0)
        plt.show()
        
        plt.plot(X_test_sorted, mean_pred_sorted[:,4], color="black", linewidth=2, label="DeepIV")
        plt.plot(X_test_sorted, causal_effect[sorted_indices_test], color="green", linewidth=2, label="True")
        plt.fill_between(X_test_sorted, 
                         mean_pred_sorted[:,4] - conf_int_sorted[:,4], 
                         mean_pred_sorted[:,4] + conf_int_sorted[:,4],color="black", alpha=0.3)
        plt.xlabel("X",fontsize=22)
        plt.xlim(xlim_min[i],xlim_max[i])
        plt.ylim(ylim_min[i],ylim_max[i])
        plt.xticks(xticks[i,:])
        plt.xticks(fontsize=22)
        plt.yticks([])
        plt.gcf().set_size_inches(4, 4) 
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        #Change the following path to save the plot as a pdf
        #plt.savefig("PATH\simMeant5"+str(i)+".pdf", bbox_inches='tight',pad_inches =0)
        plt.show()
        
        plt.plot(X_test_sorted, mean_pred_sorted[:,5], color="black", linewidth=2, label="DeepGMM")
        plt.plot(X_test_sorted, causal_effect[sorted_indices_test], color="green", linewidth=2, label="True")
        plt.fill_between(X_test_sorted, 
                         mean_pred_sorted[:,5] - conf_int_sorted[:,5], 
                         mean_pred_sorted[:,5] + conf_int_sorted[:,5],color="black", alpha=0.3)
        
        plt.xlim(xlim_min[i],xlim_max[i])
        plt.ylim(ylim_min[i],ylim_max[i])
        plt.xlabel("X",fontsize=22)
        plt.xticks(xticks[i,:])
        plt.xticks(fontsize=22)
        plt.yticks([])
        plt.gcf().set_size_inches(4, 4) 
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        #Change the following path to save the plot as a pdf
        #plt.savefig("PATH\simMeant6"+str(i)+".pdf", bbox_inches='tight',pad_inches =0)
        plt.show()
        
        print(conf_int_sorted[:,6])
        plt.plot(X_test_sorted, mean_pred_sorted[:,6], color="black", linewidth=2, label="GCFN")
        plt.plot(X_test_sorted, causal_effect[sorted_indices_test], color="green", linewidth=2, label="True")
        plt.fill_between(X_test_sorted, 
                         mean_pred_sorted[:,6] - conf_int_sorted[:,6], 
                         mean_pred_sorted[:,6] + conf_int_sorted[:,6],color="black", alpha=0.3)
        
        plt.xlim(xlim_min[i],xlim_max[i])
        plt.ylim(ylim_min[i],ylim_max[i])
       
        plt.xlabel("X",fontsize=22)
        plt.xticks(xticks[i,:])
        plt.xticks(fontsize=22)
        plt.yticks([])
        plt.gcf().set_size_inches(4, 4) 
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        #Change the following path to save the plot as a pdf
        #plt.savefig("PATH\simMeant7"+str(i)+".pdf", bbox_inches='tight',pad_inches =0)
        plt.show()
        
            
   


#define estimators

#define DeepIV
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



df=10
df_first_stage=10
df_second_stage=10

#define other estimators
est2sls = Spline2sls(df_second_stage,df_first_stage)
estCF = CFSpline(df)
estDeepCF = DeepCF()
naive_spline = Spline(df)

estDeepGMM = ToyModelSelectionMethod()
estGCFN = GCFN()



n = 2000
#setting that will generate the data in the function generate_data
experiments=[8]#1,2,3,4,5,6,7
nb_estimators = 7
n_test =2000
nb_rep=1

yticks=np.full((9,3),0)
yticks[0,:] = [0,4,8]
yticks[1,:] = [-6,0,6]
yticks[2,:] = [-1,1,3]
yticks[3,:] = [-1,0,1]
yticks[4,:] = [-3,0,3]
yticks[5,:] = [-5,0,5]
yticks[6,:] = [-15,0,15]
yticks[7,:] = [-15,0,15]
yticks[7,:] = [-5,0,6]

xticks=np.full((9,4),0) 
xticks[0,:]=[-7,-2,2,7]
xticks[1,:]=[-7,-2,2,7]
xticks[2,:]=[-7,-2,2,7]
xticks[3,:]=[-5,-2,2,5]
xticks[4,:]=[-7,-2,2,7]
xticks[5,:]=[-9,-3,3,9]
xticks[6,:]=[-5,-2,2,5]
xticks[7,:]=[-4,-1,1,4]
xticks[7,:]=[-8,-3,3,8]

ylim_min=[-1,-8,-2,-2,-4,-6,-20,-20,-6]
ylim_max=[10,8,4,2,4,6,20,20,8]

xlim_min=[-10,-10,-10,-6,-8,-10,-6,-5,-10]
xlim_max=[10,10,10,6,8,10,6,5,10]

titles = ["Simulation 0","Simulation 1","Simulation 2","Simulation 3","Simulation 4","Simulation 5","Simulation 6",
          "Simulation 7","Simulation 8","Simulation 9","Simulation 10","Simulation 11","Simulation 12","Simulation 13",
          "Simulation 14","Simulation 15","Simulation 16"]


#This function simulate and plot directly all the plots
mse_est = simulateAndPlot(n,experiments,nb_estimators,n_test,titles,nb_rep,ylim_min,ylim_max,xlim_min,xlim_max,xticks,yticks)

    
