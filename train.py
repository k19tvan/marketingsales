import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

data = pd.read_csv('marketing_sales_data.csv')
data = data.dropna(axis = 0)

ols_data = data[['Radio', 'Social Media', 'Sales']]

X = ols_data[['Radio', 'Social Media']]
Y = ols_data['Sales']
X = sm.add_constant(X)

OLS = sm.OLS(Y, X)
model = OLS.fit()

intercept = model.params.const
slope_radio = model.params['Radio']
slope_social_media = model.params['Social Media']

xx, yy = np.meshgrid(np.linspace(X.iloc[:, 1].min(), X.iloc[:, 1].max(), 50), np.linspace(X.iloc[:, 2].min(), X.iloc[:, 2].max(), 50))
zz = intercept + slope_radio * xx + slope_social_media * yy

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X.iloc[:, 1], X.iloc[:, 2], Y, c='r', marker='o')
ax.plot_surface(xx, yy, zz, alpha=0.5, cmap='viridis')

ax.set_xlabel('Radio')
ax.set_ylabel('Social Media')
ax.set_zlabel('Sales')
ax.set_title('3D Regression Plot')

plt.show()