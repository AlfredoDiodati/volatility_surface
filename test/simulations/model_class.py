import numpy as np

def compute_normal_scaled_score(return_innovation):
    scaled_score = (return_innovation ** 2) - 1.0
    return scaled_score

def compute_persistence_parameters(alpha, K, decades_covered):
    beta = 10 ** (decades_covered / K)
    exponents = np.arange(K+1)
    x = np.exp(-alpha * beta ** exponents)
    rho = x/(1 + x)
    return rho, beta

def compute_correction_factors(alpha, beta, K):
    c = np.zeros(K + 1)
    c[K] = 1.0
    
    for j in range(1, K + 1):
        sum_term = 0.0
        for l in range(j):
            exponent_part = alpha * (1 - beta ** (l - j))
            power_part = beta ** (-alpha * (j - l))
            sum_term += c[K - l] * power_part * np.exp(exponent_part)
        c[K - j] = 1.0 - sum_term
    return c

def compute_weights(correction_factors, alpha, beta, K):
    exponents = np.arange(K + 1) / (2 * alpha)
    raw_weights = np.abs(correction_factors) * beta ** exponents
    weights = raw_weights / np.sum(raw_weights)
    return weights

def simulate_power_law_model(
    horizon,
    mu_return,
    K,
    alpha,
    decades_covered,
    sigma_xi_squared,
    alpha_scales,
    score_function=None,
    random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    
    if score_function is None:
        score_function = compute_normal_scaled_score
    
    alpha_scales = np.asarray(alpha_scales)
    if alpha_scales.shape[0] != K + 1:
        raise ValueError(f"alpha_scales must have length {K + 1}")
    
    rho, beta = compute_persistence_parameters(alpha, K, decades_covered)
    correction_factors = compute_correction_factors(alpha, beta, K)
    weights = compute_weights(correction_factors, alpha, beta, K)
    
    component_means = np.ones(K + 1)
    component_variances = (alpha_scales ** 2 * sigma_xi_squared) / (1 - rho ** 2)
    
    component_values = component_means + np.sqrt(component_variances) * np.random.standard_normal(K + 1)
    
    simulated_returns = np.zeros(horizon)
    simulated_volatilities = np.zeros(horizon)
    
    for t in range(horizon):
        variance_squared = np.sum(weights * component_values)
        volatility = np.sqrt(variance_squared)
        simulated_volatilities[t] = variance_squared
        
        return_innovation = np.random.standard_normal()
        simulated_returns[t] = mu_return + volatility * return_innovation
        
        scaled_score = score_function(return_innovation)
        
        component_values = (
            component_means + 
            rho * (component_values - component_means) + 
            alpha_scales * scaled_score
        )
    
    return simulated_returns, simulated_volatilities

if __name__ == "__main__":
    K = 500
    alpha = 0.1
    decades_covered = 50
    sigma_xi_squared = 1.0
    alpha_scales = np.ones(K + 1) * 0.15
    mu_return = 0.0005
    horizon_length = 1250
    
    returns, volatility = simulate_power_law_model(
        horizon=horizon_length,
        mu_return=mu_return,
        K=K,
        alpha=alpha,
        decades_covered=decades_covered,
        sigma_xi_squared=sigma_xi_squared,
        alpha_scales=alpha_scales,
        random_seed=42
    )
    
    output_file = "plot/test/simulations/simulation_results.txt"
    
    with open(output_file, "w") as f:
        f.write("Time,Return,Volatility\n")
        for t in range(horizon_length):
            f.write(f"{t},{returns[t]:.8f},{volatility[t]:.8f}\n")
    
    print(f"Simulation complete. Results saved to {output_file}")
    print(f"Mean return: {np.mean(returns):.6f}")
    print(f"Mean volatility: {np.mean(volatility):.6f}")
    print(f"Return standard deviation: {np.std(returns):.6f}")
    print(f"Volatility standard deviation: {np.std(volatility):.6f}")
    
    try:
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        
        pdf_file = "plot/test/simulations/simulation_plot.pdf"
        
        with PdfPages(pdf_file) as pdf:
            fig, axes = plt.subplots(2, 1, figsize=(12, 8))
            
            time_index = np.arange(horizon_length)
            
            axes[0].plot(time_index, returns, linewidth=0.8, color='steelblue')
            axes[0].set_ylabel('Return')
            axes[0].set_title('Simulated Returns')
            axes[0].grid(True, alpha=0.3)
            
            axes[1].plot(time_index, volatility, linewidth=0.8, color='darkred')
            axes[1].set_ylabel('Volatility')
            axes[1].set_xlabel('Time')
            axes[1].set_title('Simulated Volatility')
            axes[1].grid(True, alpha=0.3)
            
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
        
        print(f"Plot saved to {pdf_file}")
    except ImportError:
        print("matplotlib not available, skipping plot generation")