import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from GD import predict, cost, gradientDescent

data = pd.read_csv("marketing_sales_data.csv")
data = data.dropna(axis = 0)
data = pd.get_dummies(data, columns=['TV', 'Influencer'], drop_first=True)

Xtrain = data[['Radio', 'Social Media']]
ytrain = data['Sales']

scaler = StandardScaler()
Xtrain = scaler.fit_transform(Xtrain)

Xtrain = sm.add_constant(Xtrain)

theta = gradientDescent(Xtrain, ytrain)
intercept = theta[0]
slope_radio = theta[1] 
slope_social_media = theta[2]

xx, yy = np.meshgrid(np.linspace(Xtrain[:, 1].min(), Xtrain[:, 1].max(), 50),
                     np.linspace(Xtrain[:, 2].min(), Xtrain[:, 2].max(), 50))

zz = intercept + slope_radio * xx + slope_social_media * yy

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(Xtrain[:, 1], Xtrain[:, 2], ytrain, color='red', marker='o', alpha=0.5, label='Data points')

ax.plot_surface(xx, yy, zz, alpha=0.5, cmap='viridis')

ax.set_xlabel('Radio (Standardized)')
ax.set_ylabel('Social Media (Standardized)')
ax.set_zlabel('Sales')
ax.set_title('3D Regression Plot')

plt.legend()
plt.show()
