#Natural cubic spline

import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

class Spline:
    def __init__(self, df):
        self.df = df
        self.knots=None

    def ns_basis_function(self, x,knots):
        
        basis_matrix = np.zeros((len(x), self.df))
        d = (np.maximum(0, (x[:, None]- knots[:-1])**3) - np.maximum(0, (x[:, None] - knots[-1])**3)) / (knots[-1] - knots[:-1])
        basis_matrix[:, 1:] = d[:, :-1] - d[:, -1][:, None]
        basis_matrix[:,0] = x
        return basis_matrix

    def fit(self, X, Y):
        self.knots_X = np.quantile(X, np.linspace(0, 1, self.df + 1))
        self.basis_second_stage = self.ns_basis_function(X,self.knots_X)

        
        # Second stage
        predictors = self.basis_second_stage
        predictors = sm.add_constant(predictors)
        self.model = sm.OLS(Y, predictors).fit()

    def predict(self, X):
        basis_new = self.ns_basis_function(X,self.knots_X)
        return self.model.predict(sm.add_constant(basis_new))


#Example, it gives a baised estimate since we regress Y on X.
"""
n = 2000
df =10
df_first_stage=10

Z = np.random.normal(0, 4, n)
H = np.random.normal(0, 2, n)
X = Z + H + np.random.normal(0,1,n)
Y = abs(X)  +  H + np.random.normal(0, 1, n)


estimator = Spline(df)
estimator.fit(X, Y)


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