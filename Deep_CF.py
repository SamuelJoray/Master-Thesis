# -*- coding: utf-8 -*-

"""
Deep Control Function uses the principle of control function method and use it along with deep learning. 
This allows to weaken some assumptions of Control Function algorithm. 
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scipy.stats import norm
from keras.initializers import Constant



class DeepCF:
    def __init__(self):
        self.model_second_stage= None
        self.H_intercept = None


    def fit(self, Z, X, Y,W):
        
        #Dimensions of entry vector Z and X, Y is supposed to have dimension 1
        d_x, d_z,d_w = [np.shape(a)[1] if np.ndim(a) > 1 else 1 for a in [X,  Z,W]]
        
        #Create a Keras neural network for the first stage
        model_first_stage = Sequential()
        model_first_stage.add(Dense(64, input_dim=d_z+d_w, activation='relu'))
        model_first_stage.add(Dense(32, activation='relu'))
        model_first_stage.add(Dense(16, activation='relu'))
        model_first_stage.add(Dense(d_x, activation='linear'))

        # Compile the model
        model_first_stage.compile(optimizer='adam', loss='mse')

        # Train the model with input Z and output X
        model_first_stage.fit(np.column_stack((Z,W)), X, epochs=100, verbose=1)
        
        # Calculate residuals, which is the control function
        predictions_first_stage = model_first_stage.predict(np.column_stack((Z,W)))
        if X.ndim ==1:
            residuals = X - predictions_first_stage.flatten()
        else:
            #It assumes that the confounding in the first stage is a vector with copy of itself.
            residuals = np.mean(X - predictions_first_stage, axis=1)
        
        
        #Create Keras neural network for the second stage
        self.model_second_stage = Sequential()
        self.model_second_stage.add(Dense(64, input_dim=d_x+d_w+1, activation='relu'))
        self.model_second_stage.add(Dense(32, activation='relu'))
        self.model_second_stage.add(Dense(16, activation='relu'))
        self.model_second_stage.add(Dense(1, activation='linear'))

        # Compile the model
        self.model_second_stage.compile(optimizer='adam', loss='mse')

        # Train the model with input X, residuals and output Y
        self.model_second_stage.fit(np.column_stack((X, residuals,W)), Y,epochs=100, verbose=1)

        #Assuming the mean of h_2(H,N_Y) is zero, we extract the mean of the residuals of the second stage given H=0.
        #It makes sure that the intercept is accounted in the intercept and not in h_2(H,N_Y).
        self.H_intercept = np.mean(Y-self.model_second_stage.predict(np.column_stack((X, np.zeros(np.shape(X)[0]),W))).flatten())
       

    def predict(self,X,W):
        print(self.H_intercept)
        predictions = self.model_second_stage.predict(np.column_stack((X, np.zeros(np.shape(X)[0]),np.zeros(np.shape(X)[0]))))+self.H_intercept
        return predictions
    
       
"""
#Example where X,Z have i diminesions 
i=10
n=2000
Z = np.random.normal(0, 1, size=(n, i))
H = np.random.normal(0, 1, n)
X = Z + np.tile(H.reshape(-1, 1), (1,i)) +np.random.normal(0, 1, size=(n, i))
Y = np.dot(X,np.ones(i)) + H+np.random.normal(0, 1, n)
W=np.zeros((n,i))

estDeepCF = DeepCF()
estDeepCF.fit(Z,X,Y,W)

n_test=1000
X_test =  np.random.normal(0, 1, size=(n, i))
predictions = estDeepCF.predict(X_test)

print("Mean squared error is : ", np.mean((np.dot(X_test,np.ones(i))-Y)**2 ))


"""
"""
#Example where X,Z have 1 dimension
n=2000

Z = np.random.uniform(-5,5,n)
H = np.random.normal(0,2, n)
X = Z + H + np.random.normal(0,1,n)
Y = np.cos(X)  + H  + np.random.normal(0, 1, n)
W=np.zeros(n)

estDeepCF = DeepCF()
estDeepCF.fit(Z,X,Y,W)

n_test=1000
X_test = np.random.uniform(-5, 5, n_test)
W_test = np.zeros(n_test)
predictions = estDeepCF.predict(X_test,W_test)

sorted_indices = np.argsort(X)
X_sorted = X[sorted_indices]
Y_sorted = Y[sorted_indices]


sorted_indices_test = np.argsort(X_test)
X_test_sorted = X_test[sorted_indices_test]


plt.scatter(X_sorted,Y_sorted, label='Actual data', color='blue',s=1)
plt.plot(X_test_sorted, predictions[sorted_indices_test], label='Fitted curve', color='red')
plt.plot(X_test_sorted,np.cos(X_test)[sorted_indices_test],label='true fit', color='green')


#naive estimator, which fits Y on X. It is useful to see if the confounding effect is corrected.
model_naive = Sequential()
model_naive.add(Dense(32, input_dim=1, activation='relu'))
model_naive.add(Dense(16, activation='relu'))
model_naive.add(Dense(1, activation='linear'))

# Compile the model
model_naive.compile(optimizer='adam', loss='mse')

# Train the model with input X and output Y.
model_naive.fit(X, Y, epochs=10, verbose=1)  
predictions_naive = model_naive.predict(X)


plt.plot(X_sorted, predictions_naive[sorted_indices] ,label='naive estimator' ,color='yellow')
plt.legend()
plt.show()
"""
