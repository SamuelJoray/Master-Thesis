
"""
Deep Control Function uses the principle of control function method and use it along with deep learning. 
This allows to weaken some assumptions of Control Function algorithm. More details in the Master thesis pdf document.
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scipy.stats import norm
from keras.initializers import Constant

from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error



  
class DeepCF:
    def __init__(self):
        self.model_second_stage= None
        self.H_intercept = None
        self.residuals = None
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

        # Train the model with input Z and output X, the number of epochs can be changed
        model_first_stage.fit(np.column_stack((Z,W)), X, epochs=100, verbose=0)
        
        # Calculate residuals, which are used as the control function
        predictions_first_stage = model_first_stage.predict(np.column_stack((Z,W)))
        if X.ndim ==1:
            residuals = X - predictions_first_stage.flatten()
        else:
            residuals = np.mean(X - predictions_first_stage, axis=1)
            
        self.residuals = residuals
       
        #Create Keras neural network for the second stage
        self.model_second_stage = Sequential()
        self.model_second_stage.add(Dense(64, input_dim=d_x+d_w+1, activation='relu'))
        self.model_second_stage.add(Dense(32, activation='relu'))
        self.model_second_stage.add(Dense(16, activation='relu'))
        self.model_second_stage.add(Dense(1, activation='linear'))

        # Compile the model
        self.model_second_stage.compile(optimizer='adam', loss='mse')

        # Train the model with input X, residuals and output Y, the number of epochs can be changed
        self.model_second_stage.fit(np.column_stack((X, residuals,W)), Y,epochs=300, verbose=0)

        #Assuming the mean of h_2(H,N_Y) is zero, we extract the mean of the residuals of the second stage given H=0.
        #It makes sure that the intercept is accounted in the intercept and not in h_2(H,N_Y).
        self.H_intercept = np.mean(Y-self.model_second_stage.predict(np.column_stack((X, np.zeros(len(X)),W))).flatten())
        
       

    def predict(self,X,W,smoothing_param=100,second_stage_additive=False):
        #If the second is not additive or it is a nonlinear function of the control function in the econd stage,
        #we have to take the mean over the whole control function to get the counterfactual function.
        if (second_stage_additive == False):
            #When we can write Y as Y = f(X,W,H,N_Y) the true causal effect is E_H[Y|X,H]. The following estimate the latter term.
            predictions = np.zeros(len(X))
            for i in range(smoothing_param):
                index = np.random.choice(range(0, len(self.residuals)), len(X))
                res = self.residuals[index]
                predictions = predictions + np.array(self.model_second_stage.predict(np.column_stack((X, res,W)),verbose=0)).flatten()
            predictions = predictions/smoothing_param


            #smooth the predictions with cross-validation
            # Perform cross-validation for each value of s
            s_values = np.concatenate((np.arange(0.1, 1.1, 0.1),np.arange(0,100), np.arange(100,1000,10), np.arange(1000,10000,100),np.arange(10000,100000,1000)))
            sorted_indices = np.argsort(X)
            
            
            mse_values=[]
            pred=[]
            x_train, x_valid, y_train, y_valid = train_test_split(X, predictions, test_size=0.2)
            
            sorted_indices = np.argsort(x_train)
            x_train_sorted = x_train[sorted_indices]
            y_train_sorted = y_train[sorted_indices]
            
            for s in s_values:
                # Fit the spline to the training data
                spline = UnivariateSpline(x_train_sorted, y_train_sorted, s=s)
    
                # Evaluate the spline on the validation data
                y_pred = spline(x_valid)
                
                # Calculate the mean squared error
                mse = mean_squared_error(y_valid, y_pred)
                
                # Store the mean squared error
                mse_values.append(mse)
                pred.append(spline(X))
            
            # Find the index of the minimum mean squared error
            best_s_index = np.argmin(mse_values)
            return pred[best_s_index]
        
        else:
            #We add H_intercept to make sure the intercept is indeed accounted in the intercept and not in the control function.
            predicitons= self.model_second_stage.predict(np.column_stack((X, np.zeros(len(X)),W)),verbose=0)+self.H_intercept
            return predicitons
"""
#Example

n = 2000

Z = np.random.normal(0, 4, n)
H = np.random.normal(0, 2, n)
X = Z + H + np.random.normal(0,1,n)
Y = abs(X)  +  H+H**2-np.mean(H**2) + np.random.normal(0, 1, n)
W=np.zeros(n)


estimator = DeepCF()
estimator.fit(Z,X, Y,W)
predict = estimator.predict(X,W,second_stage_additive=True)

# Plotting
plt.scatter(X, Y, marker='o', s=0.25, label="True")
orderX = np.argsort(X)

#estimator
plt.plot(X[orderX], predict[orderX], color="red", linewidth=3, label="Predicted")

#true causal effect
plt.plot(X[orderX], np.abs(X)[orderX], color="green", linewidth=3, label="True")

plt.legend(loc="upper right")
plt.show()
"""