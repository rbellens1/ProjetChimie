import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#mesures
n_samples = 4
U1 = [0.524, 2.247,4.150 ,4.848  ]
U2= [0.607, 2.191,4.175,4.930 ]
I = [0.1,  0.5   ,1     ,1.2]
T= [30,  100   ,195  ,320 ]  # à 320 fil roug
U_moy =[0,0,0,0]
for i in range(4):
    U_moy[i]=(U1[i]+U2[i])/2

def compute_resistance(U, I):
    """
    Computes the resistance of a material based on voltage and current measurements.

    Parameters:
    U (list or np.array): Vector of voltage measurements.
    I (list or np.array): Vector of current measurements.

    Returns:
    tuple: Predicted U values, resistance of the material, mean squared error.
    """
    # Convert lists to numpy arrays if necessary
    U = np.array(U)
    I = np.array(I)

    # Reshape I for sklearn LinearRegression
    I_reshaped = I.reshape(-1, 1)

    # Create and fit the linear regression model
    model = LinearRegression()
    model.fit(I_reshaped, U)

    # Predict U values using the linear model
    U_pred = model.predict(I_reshaped)

    # Calculate the resistance (slope of the line)
    R = model.coef_[0]

    # Calculate the mean squared error
    mse = mean_squared_error(U, U_pred)

    return U_pred, R, mse

def plot_linear_regression(U, I, U_pred,T):
    """
    Plots a linear regression of U(I) based on vectors of measurements.

    Parameters:
    U (list or np.array): Vector of voltage measurements.
    I (list or np.array): Vector of current measurements.
    U_pred (list or np.array): Predicted voltage measurements.
    """
    # Plot the measurements and the linear regression line
    plt.scatter(I, U, color='green', label='Measurements')
    plt.plot(I, U_pred, color='orange', label='Linear Regression')
    plt.xlabel('I [A]')
    plt.ylabel('U [V]')
    plt.title('U(I) Linear Regression with T = ' + str(T)+ '[K]')
    plt.legend()
    plt.grid(True, linestyle='dashed')
    plt.show()


def compute_drhodT(rho,T):
   
    
    rho = np.array(rho)
    T = np.array(T)

    T_reshaped = T.reshape(-1, 1)

    model = LinearRegression()
    model.fit(T_reshaped, rho)

    # Predict U values using the linear model
    rho_pred = model.predict(T_reshaped)

    # Calculate the resistance (slope of the line)
    drhodT = model.coef_[0]

    # Calculate the mean squared error
    mse = mean_squared_error(rho, rho_pred)

    return rho_pred, drhodT, mse

def plot_resistivity(rho, T, rho_pred):
    # Plot the measurements and the linear regression line
    plt.scatter(T, rho, color='green', label='Measurements')
    plt.plot(T, rho_pred, color='orange', label='Linear Regression')
    plt.xlabel('T [°c]')
    plt.ylabel(r'rho [$\Omega$ m]')
    plt.title(r'$\rho(T)$ Linear Regression')
    plt.legend()
    plt.grid(True, linestyle='dashed')
    plt.show()

U_pred, R, mse = compute_resistance(U_moy, I)
plot_linear_regression(U_moy,I,U_pred,0)
S = np.pi*(0.00017/2)**2 #[m^2]
L = 0.05  #[m]
rho_moy = R*S/L
print(rho_moy)


rho = np.zeros(n_samples)
for i in range(n_samples):
    rho[i] = (U_moy[i]/I[i])*S/L

rho_pred,drhodT,mse = compute_drhodT(rho,T)
plot_resistivity(rho,T,rho_pred)



I3 = [0.08,0.15,0.21 ,0.25,0.29,0.34,0.40,0.45,0.49, 0.55,0.60,0.65,0.70,0.75 ]
U3 = [0.453,0.764,1.1,1.3,1.47,1.76,2.02, 2.32, 2.52,2.78,3.02,3.26,3.58,3.78 ]

