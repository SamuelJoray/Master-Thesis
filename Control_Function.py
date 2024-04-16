# -*- coding: utf-8 -*-



import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

class CF:
    def __init__(self, df):
        self.df = df

    def ns_basis_function(self, x):
        knots = np.quantile(x, np.linspace(0, 1, self.df + 1))
        basis_matrix = np.column_stack([x ** 3 - (np.maximum(0, x - knots[i])) ** 3 / (knots[-1] - knots[i]) for i in range(self.df)])
        basis_matrix[:, 1:] -= basis_matrix[:, :-1]
        return basis_matrix

    def fit(self, Z, X, Y):
        basis_first_stage = self.ns_basis_function(Z)
        basis_second_stage = self.ns_basis_function(X)

        # First stage
        res = sm.OLS(X, sm.add_constant(basis_first_stage)).fit().resid

        # Second stage
        predictors = np.column_stack([basis_second_stage, res])
        predictors = sm.add_constant(predictors)
        self.model = sm.OLS(Y, predictors).fit()

    def predict(self, X):
        basis_second_stage_new = np.column_stack([self.ns_basis_function(X), np.zeros(len(X))])
        return self.model.predict(sm.add_constant(basis_second_stage_new))

np.random.seed(42)

n = 2000
df = 10

Z = np.random.uniform(-3, 3, n)
H = np.random.uniform(-2, 2, n)

X = Z + 2 * np.abs(Z) - 4 * H + np.random.normal(0, 1, n)
Y = np.abs(X) + 5 * np.sin(X) + 4 * H + np.random.normal(0, 1, n)

estimator = CF(df)
estimator.fit(Z, X, Y)

# Plotting
plt.scatter(X, Y, marker='o', s=0.25, label="True")
orderX = np.argsort(X)

plt.plot(X[orderX], estimator.predict(X[orderX]), color="red", linewidth=3, label="Predicted")

basis_naive_spline = estimator.ns_basis_function(X)
plt.plot(X[orderX], sm.OLS(Y, sm.add_constant(basis_naive_spline)).fit().fittedvalues[orderX], color="blue", linewidth=3, label="Naive")

plt.plot(X[orderX], (np.abs(X) + 5 * np.sin(X))[orderX], color="green", linewidth=3, label="True (without noise)")

plt.legend(loc="upper right")
plt.show()
