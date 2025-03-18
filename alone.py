import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

I3 = [0.08,0.15,0.21 ,0.25,0.29,0.34,0.40,0.45,0.49, 0.55,0.60,0.65,0.70,0.75 ]
U3 = [0.453,0.764,1.1,1.3,1.47,1.76,2.02, 2.32, 2.52,2.78,3.02,3.26,3.58,3.78 ]
n_samples = len(I3)

rho =[0]* n_samples
print(rho)
S = np.pi*(0.00017/2)**2 #[m^2]
L = 0.05  #[m]
for i in range(n_samples):
    r = U3[i]/I3[i]
    rho[i] = r*S/L

plt.scatter(I3, U3, color='blue')
plt.show()

plt.scatter(U3, rho, color='red')
plt.show()