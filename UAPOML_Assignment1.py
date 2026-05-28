import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# ==========================================
# Section 2:(Q10 - Q12)
# ==========================================

print("--- QUESTION 10 ---")
# Stock prices given in the assignment
prices = np.array([
    [100, 108, 103, 115, 110, 119, 125, 121, 130, 127, 135, 140],
    [200, 195, 210, 205, 220, 215, 225, 230, 222, 235, 240, 238]
])

# 10(a) Simple returns using vector slicing (P_t - P_t-1) / P_t-1
returns = (prices[:, 1:] - prices[:, :-1]) / prices[:, :-1]
print(f"Returns array shape: {returns.shape}")

# 10(b) Annualised Mean and Standard Deviation
# Multiply mean by 12, multiply std by sqrt(12)
ann_mean = np.mean(returns, axis=1) * 12
ann_std = np.std(returns, axis=1, ddof=1) * np.sqrt(12)
print(f"Stock A - Ann. Mean: {ann_mean[0]:.4f}, Ann. Std: {ann_std[0]:.4f}")
print(f"Stock B - Ann. Mean: {ann_mean[1]:.4f}, Ann. Std: {ann_std[1]:.4f}")

# 10(c) Sample Covariance matrix
cov_mat = np.cov(returns)
print("Covariance Matrix:")
print(cov_mat)


print("\n--- QUESTION 11 ---")
# 3-asset parameters from Question 6
mu = np.array([0.15, 0.08, 0.05])
sigma = np.array([
    [0.0625, 0.0120, 0.0010],
    [0.0120, 0.0144, 0.00096],
    [0.0010, 0.00096, 0.0016]
])

# 11(a) Equal weight portfolio math
w_eq = np.array([1/3, 1/3, 1/3])
exp_ret_eq = w_eq @ mu
var_eq = w_eq.T @ sigma @ w_eq
print(f"Equal Weight Portfolio -> Return: {exp_ret_eq:.4f}, Variance: {var_eq:.6f}")

# 11(b) 10,000 random portfolios
np.random.seed(42) # Set seed for reproducibility
weights_sim = np.random.dirichlet(np.ones(3), size=10000)

# Vectorised returns and standard deviations
sim_returns = weights_sim @ mu
sim_vars = np.sum(weights_sim * (weights_sim @ sigma), axis=1)
sim_stds = np.sqrt(sim_vars)

# 11(c) Find maximum Sharpe ratio
sharpe_array = sim_returns / sim_stds
max_idx = np.argmax(sharpe_array)
print(f"Max Sharpe Ratio: {sharpe_array[max_idx]:.4f}")
print(f"Optimal Weights: {weights_sim[max_idx]}")


print("\n--- QUESTION 12 ---")
# Setup basic parameters
mu1, s1 = 0.12, 0.20
mu2, s2 = 0.06, 0.10
w1, w2 = 0.6, 0.4

# 12(a) Generate 200 correlation values and calculate risks
rhos = np.linspace(-1, 1, 200)
port_vars = (w1*s1)**2 + (w2*s2)**2 + 2*w1*w2*rhos*s1*s2
port_stds = np.sqrt(port_vars)

# 12(b) Find the minimum risk
min_risk = np.min(port_stds)
min_rho = rhos[np.argmin(port_stds)]
print(f"Minimum Risk is {min_risk:.4f} occurring at correlation {min_rho:.2f}")

# 12(c) proof:
# Since d(variance)/d(rho) = 2*w1*w2*s1*s2, and all those terms are positive,
# the variance strictly increases as rho increases. Therefore, the minimum 
# must sit at the lowest possible value of rho, which is -1.


# ==========================================
# Section 3: (Q13 - Q14)
# ==========================================

print("\n--- QUESTION 13 ---")
# Setup simulated dataset from the assignment skeleton
np.random.seed(0)
dates = pd.date_range('2023-01-02', periods=52, freq='W-MON')
mu_wk = np.array([0.003, 0.002, 0.001, 0.0015])
sig_wk = np.array([0.04, 0.03, 0.02, 0.025])
sim_rets = np.random.normal(mu_wk, sig_wk, (52, 4))
prices_df = pd.DataFrame(100 * np.cumprod(1 + sim_rets, axis=0), index=dates, columns=['AAPL', 'MSFT', 'GOOGL', 'AMZN'])

# 13(a) Compute simple returns
returns_df = prices_df.pct_change().dropna()
print("First 3 rows of weekly returns:")
print(returns_df.head(3))
print(f"Shape: {returns_df.shape}")

# 13(b) Summary statistics
stats = returns_df.describe()
print(f"Highest Mean Return: {stats.loc['mean'].idxmax()}")
print(f"Highest Standard Dev: {stats.loc['std'].idxmax()}")

# 13(c) Sharpe Ratios
rf_weekly = (1 + 0.02)**(1/52) - 1
annual_sharpe = (returns_df.mean() - rf_weekly) / returns_df.std() * np.sqrt(52)
print("Annualised Sharpe Ratios:")
print(annual_sharpe)


print("\n--- QUESTION 14 ---")
# 14(a) Correlation Matrix
corr_df = returns_df.corr()
print("Correlation Matrix:")
print(corr_df)
# Finding the lowest pair
lowest_pair = corr_df.replace(1.0, np.nan).stack().idxmin()
print(f"Lowest correlation pair: {lowest_pair}")

# 14(b) Equal Weight Portfolio Returns
w_series = pd.Series([0.25, 0.25, 0.25, 0.25], index=returns_df.columns)
eq_port_rets = returns_df.dot(w_series)

# 14(c) Resample to Monthly
monthly_rets = eq_port_rets.resample('ME').apply(lambda x: (1 + x).prod() - 1)
print(f"Monthly Resampled Mean: {monthly_rets.mean():.4f}")
print(f"Monthly Resampled Std: {monthly_rets.std():.4f}")


# ==========================================
# Section 4: (Q15)
# ==========================================

print("\nGenerating Plots for Question 15...")

# Set up the figure side-by-side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Portfolio Theory - Week 1 Visualisations', fontsize=16, fontweight='bold')

# --- SUBPLOT 1: Efficient Frontier ---
# Scatter plot of all portfolios
scatter = ax1.scatter(sim_stds, sim_returns, c=sharpe_array, cmap='viridis', s=10, alpha=0.6)
# Mark the optimal Sharpe portfolio
ax1.scatter(sim_stds[max_idx], sim_returns[max_idx], marker='*', color='gold', s=300, edgecolor='black', zorder=5)

# Mark the individual assets
individual_stds = np.sqrt(np.diag(sigma))
asset_names = ['Asset 1', 'Asset 2', 'Asset 3']
for i in range(3):
    ax1.scatter(individual_stds[i], mu[i], color='darkred', s=100, edgecolor='black', zorder=4)
    ax1.annotate(asset_names[i], (individual_stds[i], mu[i]), xytext=(8, -5), textcoords='offset points')

# Formatting Subplot 1
cbar = fig.colorbar(scatter, ax=ax1)
cbar.set_label('Sharpe Ratio')
ax1.xaxis.set_major_formatter(PercentFormatter(1.0))
ax1.yaxis.set_major_formatter(PercentFormatter(1.0))
ax1.set_xlabel('Portfolio Risk (\u03c3)')
ax1.set_ylabel('Expected Return (\u03bc)')
ax1.grid(True, linestyle='--', alpha=0.5)


# --- SUBPLOT 2: Correlation Sensitivity ---
# Plot risk vs correlation
ax2.plot(rhos, port_stds, color='navy', linewidth=2, label='Portfolio Risk')

# Add weighted average line and fill below it
weighted_risk = w1 * s1 + w2 * s2
ax2.axhline(weighted_risk, color='red', linestyle='--', linewidth=2, label='Weighted Avg. Risk')
ax2.fill_between(rhos, port_stds, weighted_risk, where=(port_stds < weighted_risk), color='lightgreen', alpha=0.4, label='Diversification Benefit')

# Formatting Subplot 2
ax2.yaxis.set_major_formatter(PercentFormatter(1.0))
ax2.set_xlabel('Correlation (\u03c1)')
ax2.set_ylabel('Portfolio Risk (\u03c3)')
ax2.legend(loc='lower right')
ax2.grid(True, linestyle='--', alpha=0.5)

# Save and show
plt.tight_layout()
plt.savefig('week1_plots.png', dpi=150)
plt.show()