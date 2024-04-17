# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 11:31:37 2024

@author: samjo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from patsy import dmatrix
#from statsmodels.stats.outliers_influence import variance_inflation_factor
#from scipy.linalg import svd

class Spline2sls:
    def __init__(self,df_second_stage, df_first_stage):
        self.df_first_stage = df_first_stage
        self.df_second_stage = df_second_stage

    def ns_basis_function(self, x,df):
        knots = np.quantile(x, np.linspace(0, 1, df + 1))
        basis_matrix = np.zeros((len(x), df))
        d = np.maximum(0, (x[:, None]- knots[:-1])**3) - np.maximum(0, (x[:, None] - knots[-1])**3) / (knots[-1] - knots[:-1])
        basis_matrix[:, 1:] = d[:, :-1] - d[:, -1][:, None]
        basis_matrix[:,0] = x
        return basis_matrix

    def fit(self, Z, X, Y):
        
        basis_second_stage = self.ns_basis_function(X,df =self.df_second_stage)

        # First stage
        fitted = np.zeros((len(Z), self.df_second_stage))
        for i in range(self.df_second_stage):
            basis_first_stage = self.ns_basis_function(Z,df =self.df_first_stage+i)
            coefficients = np.linalg.lstsq(basis_first_stage, basis_second_stage[:, i], rcond=None)[0]
            fitted[:, i] = np.dot(basis_first_stage, coefficients)

        # Second stage
        predictors = pd.DataFrame(fitted)
        predictors.columns = [f'f{i}' for i in range(self.df_second_stage)]
        self.model = np.linalg.lstsq(predictors, Y, rcond=None)[0]

    def predict(self, X):
        basis_second_stage_new = self.ns_basis_function(X,self.df_second_stage)
        return np.dot(basis_second_stage_new, self.model)



n = 10000
df = 10
df_first_stage=10


Z = np.random.uniform(-4, 4, n)
H = np.random.uniform(-1, 1, n)
X = Z - 2 * H + np.random.normal(0, 1, n)
Y = np.abs(X) + 4 * H + np.random.normal(0, 1, n)

estimator = Spline2sls(df,df_first_stage)

estimator.fit(Z, X, Y)

# Plotting
plt.scatter(X, Y, marker='o', s=0.25, label="True")
orderX = np.argsort(X)

plt.plot(X[orderX], estimator.predict(X)[orderX], color="red", linewidth=3, label="Predicted")

basis_naive_spline = dmatrix("bs(train, df=df, include_intercept=False)", {"train": X}, return_type='dataframe')
pred = basis_naive_spline @ np.linalg.lstsq(basis_naive_spline, Y, rcond=None)[0]

plt.plot(X[orderX], pred[orderX], color="blue", linewidth=3, label="Naive")

plt.plot(X[orderX], np.abs(X)[orderX], color="green", linewidth=3, label="True (without noise)")

plt.legend(loc="upper right")
plt.show()
