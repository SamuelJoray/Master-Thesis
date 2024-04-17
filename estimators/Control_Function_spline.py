#Control function implemented with splines

import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt



class CFSpline:
    def __init__(self, df):
        self.df = df
        self.knots=None

    def ns_basis_function(self, x,knots):
        
        basis_matrix = np.zeros((len(x), self.df))
        d = (np.maximum(0, (x[:, None]- knots[:-1])**3) - np.maximum(0, (x[:, None] - knots[-1])**3)) / (knots[-1] - knots[:-1])
        basis_matrix[:, 1:] = d[:, :-1] - d[:, -1][:, None]
        basis_matrix[:,0] = x
        return basis_matrix

    def fit(self, Z, X, Y):
        knots_Z = np.quantile(Z, np.linspace(0, 1, self.df + 1))
        basis_first_stage = self.ns_basis_function(Z,knots_Z)
        self.knots_X = np.quantile(X, np.linspace(0, 1, self.df + 1))
        self.basis_second_stage = self.ns_basis_function(X,self.knots_X)

        # First stage
        res = sm.OLS(X, sm.add_constant(basis_first_stage)).fit().resid
        

        # Second stage
        predictors = np.column_stack([self.basis_second_stage,res ])
        predictors = sm.add_constant(predictors)
        self.model = sm.OLS(Y, predictors).fit()
     

    def predict(self, X):
        basis_second_stage_new = np.column_stack([self.ns_basis_function(X,self.knots_X), np.zeros(len(X))])
        return self.model.predict(sm.add_constant(basis_second_stage_new))




"""
#example

n = 2000
df =10
df_first_stage=10

Z = np.random.normal(0, 4, n)
H = np.random.normal(0, 2, n)
X = Z + H + np.random.normal(0,1,n)
Y = abs(X)  +  H + np.random.normal(0, 1, n)


estimator = CFSpline(df)
estimator.fit(Z, X, Y)


# Plotting
plt.scatter(X, Y, marker='o', s=0.25, label="True")
orderX = np.argsort(X)

#estimator
plt.plot(X[orderX], estimator.predict(X)[orderX], color="red", linewidth=3, label="Predicted")

#true causal effect
plt.plot(X[orderX], np.abs(X)[orderX], color="green", linewidth=3, label="True")

plt.legend(loc="upper right")
plt.show()
"""
