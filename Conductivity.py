import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

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

def plot_linear_regression(U, I, U_pred):
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
    plt.title('U(I) Linear Regression')
    plt.legend()
    plt.grid(True, linestyle='dashed')
    plt.show()

def compute_conductivity(R, S, L):
    """
    Computes the conductivity of a material.

    Parameters:
    R (float): Resistance of the material.
    S (float): Cross-sectional area of the material.
    L (float): Length of the material.

    Returns:
    float: Conductivity of the material.
    """
    return (R * S) / L

def plot_conductivity(T, sigma):
    """
    Plots the conductivity as a function of temperature.

    Parameters:
    T (list or np.array): Vector of temperature measurements.
    sigma (list or np.array): Vector of conductivity measurements.
    """
    # Convert lists to numpy arrays if necessary
    T = np.array(T)
    sigma = np.array(sigma)

    # Plot the conductivity as a function of temperature
    plt.scatter(T, sigma, color='blue')
    plt.xlabel('T [°C]')
    plt.ylabel('σ [S/m]')
    plt.title('Conductivity as a function of Temperature')
    plt.grid(True, linestyle='dashed')
    plt.show()

# Example usage:
nbr_of_materials = 3
T = [20, 30, 40]
U = [[1.2, 2.3, 3.1, 4.8, 5.6], [1.2, 2.3, 3.1, 4.8, 5.6], [1.2, 2.3, 3.1, 4.8, 5.6]]
I = [[0.5, 1.0, 1.5, 2.0, 2.5], [0.5, 1.0, 1.5, 2.0, 2.5], [0.5, 1.0, 1.5, 2.0, 2.5]]
S = 1
L = 1

R = np.zeros(nbr_of_materials)
sigma = np.zeros(nbr_of_materials)
mse = np.zeros(nbr_of_materials)
U_pred = [None] * nbr_of_materials

for idx in range(nbr_of_materials):
    U_pred[idx], R[idx], mse[idx] = compute_resistance(U[idx], I[idx])
    plot_linear_regression(U[idx], I[idx], U_pred[idx])
    sigma[idx] = compute_conductivity(R[idx], S, L)

plot_conductivity(T, sigma)

print("R = ", R)
print("mse = ", mse)
print("sigma = ", sigma)