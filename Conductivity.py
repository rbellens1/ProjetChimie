import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def plot_linear_regression(U, I):
    """
    Plots a linear regression of U(I) based on vectors of measurements.

    Parameters:
    U (list or np.array): Vector of voltage measurements.
    I (list or np.array): Vector of current measurements.
    """
    # Convert lists to numpy arrays if necessary
    U = np.array(U) ; I = np.array(I)

    # Reshape I for sklearn LinearRegression
    I_reshaped = I.reshape(-1, 1)

    # Create and fit the linear regression model
    model = LinearRegression() ; model.fit(I_reshaped, U)

    # Predict U values using the linear model
    U_pred = model.predict(I_reshaped)

    R = model.coef_[0]

    #error of the model
    mse = mean_squared_error(U, U_pred)

    # Plot the measurements and the linear regression line
    plt.scatter(I, U, color='blue', label='Measurements')
    plt.plot(I, U_pred, color='red', label='Linear Regression')
    plt.xlabel('Current (I)')
    plt.ylabel('Voltage (U)')
    plt.title('Linear Regression of U(I)')
    plt.legend()
    plt.grid(True)
    plt.show()

    return R, mse

def compute_conductivity(R, S, L):
    return (R * S) / L



# Example usage:
U = [1.2, 2.3, 3.1, 4.8, 5.6]
I = [0.5, 1.0, 1.5, 2.0, 2.5]
S = 1 ; L = 1

R,mse = plot_linear_regression(U, I)

sigma = compute_conductivity(R, S, L)

print("R = ", R)
print("mse = ", mse)
print("sigma = ", sigma)