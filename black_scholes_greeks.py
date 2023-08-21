import numpy as np  # Importing the NumPy library
from scipy.stats import norm  # Importing the 'norm' function from the 'stats' module of the SciPy library

# This Python Script calculates the Option Price
# as well as the Greeks (Delta, Gamma, Theta, Vega and Rho)
# for European Call and Put Options
# The Script is based on the Black–Scholes–Merton model
# by Fischer Black, Myron Scholes and Robert C. Merton
# which was published in the year 1973 in the paper
# "The Pricing of Options and Corporate Liabilities"


def black_scholes(S, K, r, sigma, T, option_type):
    # Convert days to years
    t = T / 365
    # Calculate d1 and d2 values used in Black-Scholes-Merton formula
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)

    # Calculate the option price
    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * t) * norm.cdf(d2)
    elif option_type == "put":
        price = K * np.exp(-r * t) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("Invalid  type")
    # Greeks' calculations
    # Delta: Sensitivity of option price to stock price change
    delta = norm.cdf(d1) if option_type == "call" else -norm.cdf(-d1)
    # Gamma: Rate of change in delta for a change in stock price
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(t))
    # Theta: Sensitivity of option price to passage of time
    if option_type == "call":
        theta = -(S * sigma * norm.pdf(d1)) / (2 * np.sqrt(t)) - r * K * np.exp(
            -r * t) * norm.cdf(d2)
    elif option_type == "put":
        theta = -(S * sigma * norm.pdf(d1)) / (2 * np.sqrt(t)) + r * K * np.exp(
            -r * t) * norm.cdf(-d2)
    # Vega: Sensitivity of option price to changes in volatility
    vega = S * np.sqrt(t) * norm.pdf(d1)
    # Rho: Sensitivity of option price to changes in risk-free rate
    if option_type == "call":
        rho = K * t * np.exp(-r * t) * norm.cdf(d2)
    elif option_type == "put":
        rho = -K * t * np.exp(-r * t) * norm.cdf(-d2)
    return price, delta, gamma, theta, vega, rho

# Set the input parameters
S = 95  # Current Stock Price
K = 90  # Strike Price
r = 0.02  # Risk Free Rate
sigma = 0.22  # Implied Stock Volatility
T = 365  # Days till expiration
option_type = "call"  # call or put (European Option Type)

# Call the Black-Scholes function
price, delta, gamma, theta, vega, rho = black_scholes(S, K, r, sigma, T, option_type)

# Print the Option Price and the Greeks (Delta, Gamma, Theta, Vega, Rho)
print(f"Option price: {price:.4f}")
print(f"Delta: {delta:.4f}")
print(f"Gamma: {gamma:.4f}")
print(f"Theta: {theta:.4f}")
print(f"Vega: {vega:.4f}")
print(f"Rho: {rho:.4f}")
