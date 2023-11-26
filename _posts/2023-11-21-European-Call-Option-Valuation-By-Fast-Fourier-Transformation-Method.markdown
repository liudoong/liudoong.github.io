---
layout: post
title: European Call Option Valuation By Fast Fourier Transformation Method
date: 2023-11-21 00:00:00 +0300
description: European Call Option Valuation Example. # Add post description (optional)
img: Post_Head_001.jpg # Add image post (optional)
fig-caption: # Add figcaption (optional)
tags: [Option, Pricing, FFT] # add tag
---

In this post, I use two diffusion models to illustrate the

### Black-Schole Model
___


The Black-Scholes model is a fundamental concept in modern financial theory and is widely used for pricing European-style options. Developed by Fischer Black, Myron Scholes, and Robert Merton in the early 1970s, this model revolutionized the field of financial economics by providing a mathematical framework for valuing options, a type of derivative security.

Key aspects of the Black-Scholes model include:

European Options: The model is primarily used for pricing European options, which can only be exercised at expiration, as opposed to American options, which can be exercised at any time up to expiration.
Assumptions: The Black-Scholes model makes several simplifying assumptions:
The stock price follows a log-normal distribution and exhibits geometric Brownian motion.
The volatility of the stock (standard deviation of its returns) is constant.
Markets are efficient, meaning that prices reflect all available information.
There are no transaction costs or taxes, and trading of securities is continuous.
The risk-free interest rate is constant and known.
The Model: At its core, the Black-Scholes model is a partial differential equation that describes how the price of an option varies over time. The solution to this equation gives the theoretical price of European call and put options.
Formula: The Black-Scholes formula for a European call option is:
C
(
S
,
K
,
T
,
r
,
σ
)
=
S
Φ
(
d
1
)
−
K
e
−
r
T
Φ
(
d
2
)
C(S,K,T,r,σ)=SΦ(d 
1
​	
 )−Ke 
−rT
 Φ(d 
2
​	
 )
where 
C
C is the call option price, 
S
S is the current stock price, 
K
K is the strike price, 
T
T is the time to expiration, 
r
r is the risk-free interest rate, 
σ
σ is the volatility, 
Φ
Φ is the cumulative distribution function of the standard normal distribution, and 
d
1
d 
1
​	
  and 
d
2
d 
2
​	
  are intermediate calculations based on these variables.
Impact: The Black-Scholes model had a profound impact on the trading of options and led to the rapid growth of markets for these financial instruments. It also provided a foundation for more complex financial models and strategies.
Limitations: Despite its widespread use, the model has limitations, especially concerning its assumptions of constant volatility and market efficiency. Real-world market conditions can deviate significantly from these assumptions, leading to discrepancies between model prices and observed market prices.
The Black-Scholes model remains a cornerstone of financial theory and practice, forming the basis for various financial instruments and strategies, as well as for further research and development in financial mathematics.


<div style="text-align: center;">
$ C(S_0, K, T, r, \sigma) = S_0 \Phi(d_1) - K e^{-rT} \Phi(d_2) $
</div>

<br>
where:
- $ C $ is the price of the call option.
- $ S_0 $ is the current price of the underlying asset.
- $ K $ is the strike price of the option.
- $ T $ is the time to maturity (in years).
- $ r $ is the risk-free interest rate.
- $ \sigma $ is the volatility of the underlying asset's returns.
- $ \Phi $ represents the cumulative distribution function of the standard normal distribution.

<br>
$d_1$ and $d_2$ are given by:

<br>

<div style="text-align: center;">
$ d_1 = \frac{\ln\left(\frac{S_0}{K}\right) + \left(r + \frac{\sigma^2}{2}\right)T}{\sigma\sqrt{T}} $
</div>

<div style="text-align: center;">
$ d_2 = d_1 - \sigma\sqrt{T} $
</div>


<div style="text-align: center;">
$ \phi_{BS}(u; T, S_0, r, \sigma) = \exp\left(iu\left(\ln(S_0) + \left(r - \frac{1}{2}\sigma^2\right)T\right) - \frac{1}{2}\sigma^2 u^2 T\right) $
</div>

<br>
where:
- \( \phi_{BS}(u; T, S_0, r, \sigma) \) is the characteristic function under the Black-Scholes model.
- \( u \) is a complex number, representing the argument of the characteristic function.
- \( T \) is the time to maturity of the option.
- \( S_0 \) is the current stock price.
- \( r \) is the risk-free interest rate.
- \( \sigma \) is the volatility of the stock's returns.
- \( i \) represents the imaginary unit.

{% highlight c %}

import numpy as np
import yfinance as yf
import pandas_datareader.data as web
from scipy import fftpack

#%% data preparation 

# stock data download
ticker = 'AAPL'
data   = yf.download(ticker, start = '2022-01-01', end = '2023-01-01')

# calculate the historical volatility
returns  = np.log(data['Close'] / data['Close'].shift(1))
sigma    = np.std(returns.dropna()) * np.sqrt(252) # Annualize the volatility

# use the most recent closing price as the current stock price
S0 = data['Close'][-1]

# download risk-free rate data from FRED
risk_free_rate_df = web.DataReader('DGS3MO', 'fred', '2022-01-01', '2023-01-01')
r                 = risk_free_rate_df.iloc[-1,0] / 100 # convert the latest value to a decimal

#%% Black-Schole Model

# Parameters for option pricing
K = 100 # strike price
T = 1   # time to maturity in years

# Characteristic function for the Black-Scholes Model
def char_func_BS(u, T, r, sigma):
    return np.exp(1j * u * (np.log(S0) + (r - 0.5 * sigma**2) * T) - 0.5 * (sigma**2) * (u**2) *T)

"""
Given the following inputs
        S0:    Current stock price
        r :    Risk-Free rate
        sigma: Volatility of the stock,
        T:     Time to maturity
        u:     Variable in the complex plane
        
In this equation
        1j:    the imaginary unit
        exp(): the exponential function
        ln():  natural logarithm
        
This characteristic function is derived udner the assumption that the logarithm of the stock price
follows a normal distribution with a mean ln(S0)+(r - 0.5 * sigma**2) * T), and variance  (sigma**2) * T   
as per the Black-Scholes model
"""

# Parameters for the FFT
N     = 2**10 # Number of points in the FFT
delta = 0.25  # Grid spacing
alpha = -1.5  # Damping factor (for numerical stability)

# FFT input array
u = np.arange(N) * delta
x = np.log(S0) + (r - 0.5 * sigma**2) * T - np.pi / delta
j = np.arange(N)
v = char_func_BS(u - (alpha + 1)*1j, T, r, sigma) / (alpha**2 + alpha - u**2 + 1j * (2 * alpha + 1) * u)

# apply FFT
fft_input   = np.exp(1j * u * x) * v * delta
fft_output  = fftpack.fft(fft_input)
call_prices = np.exp(-alpha * np.log(K)) / np.pi * np.real(fft_output)

# Extract the option price
strike_grid = np.exp(x + j * np.pi / (N * delta))
option_price_index = np.searchsorted(strike_grid, K)
option_price = call_prices[option_price_index]

print(f"European Call Option Price (FFT) for {ticker} under Black-Schole Model: {option_price: .2f}")

#%% Heston Model

# Heston model parameters

v0    = sigma**2 # Initial variance
kappa = 1.0      # Speed of mean reversion
theta = sigma**2 # Long run average volatility
rho   = -0.7     # Correlation coefficient between asset and volatility
xi    = 0.1      # Volatility of volatiliy

# Characteristic function for the Heston model

def char_func_Heston(u, T, r, kappa, theta, xi, rho, v0, S0):
    # Complex numbers for the calculation
    
    d = np.sqrt((kappa - 1j * rho * xi * u)**2 + (xi**2) * (1j * u + u**2))
    g = (kappa - 1j * rho * xi * u - d) / (kappa - 1j * rho * xi * u + d)
    
    C = kappa * (kappa - 1j * rho * xi * u - d) * T - 2 * np.log((1 - g * np.exp(-d * T)) / (1 - g))
    D = ((kappa - 1j * rho * xi * u - d) / (xi**2)) * ((1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T)))
    
    return np.exp(1j * u * np.log(S0) + 1j * u * r * T + D * v0 + C * theta)

# Parameters for the FFT
N     = 2**10 # Number of points in the FFT
delta = 0.25  # Grid spacing
alpha = -1.5  # Damping factor (for numerical stability)

# FFT process

u = np.arange(N) * delta
x = np.log(S0) + (r -0.5 * sigma**2) * T - np.pi / delta
j = np.arange(N)
v = char_func_Heston(u - (alpha + 1) * 1j, T, r, kappa, theta, xi, rho, v0, S0) / (alpha**2 + alpha - u**2 + 1j * (2 * alpha +1) * u)

#apply FFT

fft_input   = np.exp(1j * u * x) * v * delta
fft_output  = fftpack.fft(fft_input)
call_price  = np.exp(-alpha * np.log(K)) / np.pi * np.real(fft_output)


# Extract option price

strike_grid = np.exp(x + j * np.pi / (N * delta))
option_price_index = np.searchsorted(strike_grid, K)
option_price = call_prices[option_price_index]

print(f"European Call Option Price (FFT) for {ticker} under Heston Model: {option_price: .2f}")

{% endhighlight %}