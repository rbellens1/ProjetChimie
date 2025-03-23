import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

## ce programme affiche les résultat de la mesure de conductivité à basse température

I = [0.08,0.15,0.21 ,0.25,0.29,0.34,0.40,0.45,0.49, 0.55,0.60,0.65,0.70,0.75 ]
U = [0.453,0.764,1.1,1.3,1.47,1.76,2.02, 2.32, 2.52,2.78,3.02,3.26,3.58,3.78 ]


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

U_pred,R,mse = compute_resistance(U,I)
plot_linear_regression(U,I,U_pred,50)

S = np.pi*(0.00017/2)**2 #[m^2]
L = 0.08  #[m]
rho = R*S/L

print(rho)