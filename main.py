import streamlit as st
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Updated Black-Scholes function with dividend yield (q)
def black_scholes_option_price_and_greeks(S, K, T, r, q, sigma, option_type='call'):
    if T <= 0 or sigma <= 0:
        raise ValueError("Time to maturity and volatility must be positive.")
    
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    
    if option_type.lower() == 'call':
        price = S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
        delta = math.exp(-q * T) * norm.cdf(d1)
        theta = - (S * math.exp(-q * T) * norm.pdf(d1) * sigma) / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * norm.cdf(d2) + q * S * math.exp(-q * T) * norm.cdf(d1)
        rho = K * T * math.exp(-r * T) * norm.cdf(d2)
    elif option_type.lower() == 'put':
        price = K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-q * T) * norm.cdf(-d1)
        delta = -math.exp(-q * T) * norm.cdf(-d1)
        theta = - (S * math.exp(-q * T) * norm.pdf(d1) * sigma) / (2 * math.sqrt(T)) + r * K * math.exp(-r * T) * norm.cdf(-d2) - q * S * math.exp(-q * T) * norm.cdf(-d1)
        rho = -K * T * math.exp(-r * T) * norm.cdf(-d2)
    else:
        raise ValueError("Option type must be 'call' or 'put'.")
    
    gamma = math.exp(-q * T) * norm.pdf(d1) / (S * sigma * math.sqrt(T))
    vega = S * math.exp(-q * T) * norm.pdf(d1) * math.sqrt(T)
    
    # Rho adjustment for put is already handled; for call it's positive, put negative
    if option_type.lower() == 'call':
        rho = K * T * math.exp(-r * T) * norm.cdf(d2)
    else:
        rho = -K * T * math.exp(-r * T) * norm.cdf(-d2)
    
    return {
        'price': price,
        'delta': delta,
        'gamma': gamma,
        'theta': theta,
        'vega': vega,
        'rho': rho
    }

# Streamlit app
st.title("Black-Scholes Option Pricing and Greeks Dashboard (with Dividend Yield)")

# Sidebar for inputs (sliders and selectors)
st.sidebar.header("Parameters")
S = st.sidebar.slider("Underlying Price (S)", min_value=50.0, max_value=150.0, value=100.0, step=1.0)
K = st.sidebar.slider("Strike Price (K)", min_value=50.0, max_value=150.0, value=100.0, step=1.0)
T = st.sidebar.slider("Time to Maturity (T)", min_value=0.01, max_value=5.0, value=1.0, step=0.01)
r = st.sidebar.slider("Risk-Free Rate (r)", min_value=0.0, max_value=0.2, value=0.05, step=0.01)
q = st.sidebar.slider("Dividend Yield (q)", min_value=0.0, max_value=0.2, value=0.0, step=0.01)
sigma = st.sidebar.slider("Volatility (sigma)", min_value=0.01, max_value=1.0, value=0.2, step=0.01)
option_type = st.sidebar.selectbox("Option Type", ["call", "put"])

# Select which Greeks/Payoff to display
st.sidebar.header("Select Plots to Show")
plot_options = ["Delta", "Gamma", "Theta", "Vega", "Rho", "Payoff"]
selected_plots = st.sidebar.multiselect("Choose Greeks/Payoff", plot_options, default=plot_options)

# Compute results for current parameters
try:
    results = black_scholes_option_price_and_greeks(S, K, T, r, q, sigma, option_type)
    premium = results['price']
    greeks = {k: v for k, v in results.items() if k != 'price'}
    
    # Payoff at expiration (assuming current S as S_T) - unaffected by q
    payoff = max(S - K, 0) if option_type == 'call' else max(K - S, 0)
    
    # Display numerical outputs in a cleaner format
    st.header("Computed Values")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Premium (Option Price)", f"{premium:.4f}")
        st.metric("Payoff (at expiration)", f"{payoff:.4f}")
    with col2:
        for greek, value in greeks.items():
            st.metric(greek.capitalize(), f"{value:.4f}")
    
    # Generate data for plots
    S_range = np.linspace(max(50, S - 50), S + 50, 100)  # Dynamic range around current S
    plot_data = {
        'Delta': [],
        'Gamma': [],
        'Theta': [],
        'Vega': [],
        'Rho': [],
        'Payoff': []
    }
    
    for s in S_range:
        res = black_scholes_option_price_and_greeks(s, K, T, r, q, sigma, option_type)
        plot_data['Delta'].append(res['delta'])
        plot_data['Gamma'].append(res['gamma'])
        plot_data['Theta'].append(res['theta'])
        plot_data['Vega'].append(res['vega'])
        plot_data['Rho'].append(res['rho'])
        plot_data['Payoff'].append(max(s - K, 0) if option_type == 'call' else max(K - s, 0))
    
    # Display each selected plot one per line/section for better Streamlit format
    if selected_plots:
        st.header("Selected Plots vs. Underlying Price (S)")
        for plot_name in selected_plots:
            with st.expander(f"{plot_name} vs. S (Click to Expand/Collapse)", expanded=True):
                fig, ax = plt.subplots(figsize=(8, 4))  # Single plot per figure for clean layout
                ax.plot(S_range, plot_data[plot_name], label=plot_name, color='blue')
                ax.set_title(f"{plot_name} vs. Underlying Price (S)")
                ax.set_xlabel('Underlying Price (S)')
                ax.set_ylabel('Value')
                ax.grid(True)
                ax.legend()
                st.pyplot(fig)
except ValueError as e:
    st.error(f"Error: {e}")

# Instructions for deployment
st.sidebar.markdown("### Deployment Notes")
st.sidebar.markdown("Save this as `app.py` (or `main.py`). Create `requirements.txt` with:")
st.sidebar.code("streamlit\nnumpy\nscipy\nmatplotlib")
st.sidebar.markdown("Upload to GitHub and deploy on Streamlit Cloud.")
