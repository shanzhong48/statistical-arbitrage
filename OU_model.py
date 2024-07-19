import numpy as np

def generateOUProcess(theta, mu, sigma, X0, dt, T):
    """
    Simulate an Ornstein-Uhlenbeck process using the Euler-Maruyama method.

    Parameters:
    - theta: rate of mean reversion
    - mu: long-term mean
    - sigma: volatility
    - X0: initial value
    - dt: time step size
    - T: total time

    Returns:
    - times: array of time points
    - X: array of simulated values of the OU process
    """
    N = int(T / dt)
    times = np.linspace(0, T, N)
    X = np.zeros(N)
    X[0] = X0

    for i in range(1, N):
        dW = np.sqrt(dt) * np.random.normal()
        X[i] = X[i-1] + theta * (mu - X[i-1]) * dt + sigma * dW

    return times, X
