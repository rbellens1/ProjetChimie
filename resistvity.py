import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

n_samples = 4

U = [0.524, 2.247,4.150 ,4.848  ]
U2= [0.607, 2.191,4.175,4.930 ]

I = [0.1,  0.5   ,1     ,1.2]
T2= [30,  100   ,195  ,320 ]  # à 320 fil rouge

r   = [0,0,0,0]
rho = [0,0,0,0]

S = np.pi*(0.00017/2)**2 #[m^2]
L = 0.05  #[m]
u_moy =[0,0,0,0]

for i in range(4):
    u_moy[i] = (U[i]+U2[i])/2
    r = u_moy[i]/I[i]
    rho[i] = r*S/L


plt.scatter(u_moy, I, color='blue')
plt.scatter(T2, rho, color='blue')
plt.show()



I3 = [0.08,0.15,0.21 ,0.25,0.29,0.34,0.40,0.45,0.49, 0.55,0.60,0.65,0.70,0.75 ]
U3 = [0.453,0.764,1.1,1.3,1.47,1.76,2.02, 2.32, 2.52,2.78,3.02,3.26,3.58,3.78 ]

plt.scatter(U3, I3, color='blue')
plt.show()