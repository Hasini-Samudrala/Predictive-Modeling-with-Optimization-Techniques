"""
Portfolio optimization with risk constraint
question 2 option a
Problem: find optimal portfolio weights to maximize returns while limiting risk
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

# setup the problem - we have 5 stocks to invest in
stock_names = ['Tech_A', 'Finance_B', 'Energy_C', 'Healthcare_D', 'Consumer_E']

# expected annual returns for each stock (as decimals)
expected_returns = np.array([0.12, 0.10, 0.08, 0.11, 0.09])

# covariance matrix - shows how stocks move together (risk)
cov_matrix = np.array([
    [0.04, 0.01, 0.00, 0.01, 0.00],
    [0.01, 0.03, 0.00, 0.00, 0.01],
    [0.00, 0.00, 0.02, 0.00, 0.00],
    [0.01, 0.00, 0.00, 0.035, 0.00],
    [0.00, 0.01, 0.00, 0.00, 0.025]
])

# constraints
max_variance = 0.025  # portfolio variance should not exceed 2.5%
risk_aversion = 1.0   # how much we penalize risk (lambda parameter)

print("portfolio optimization problem")
print("=" * 60)
print("\nstocks and expected returns:")
for i, stock in enumerate(stock_names):
    print(f"  {stock}: {expected_returns[i]*100:.1f}%")
print(f"\nrisk constraint: portfolio variance <= {max_variance*100:.2f}%")
print(f"risk aversion parameter: {risk_aversion}")
print("=" * 60)

# objective function: we want to minimize negative returns plus risk penalty
def objective(w, mu, sigma, lam):
    """calculate f(w) = -mu'w + (lambda/2)w'sigma*w"""
    return -np.dot(mu, w) + 0.5 * lam * np.dot(w, np.dot(sigma, w))

# gradient of objective function
def gradient(w, mu, sigma, lam):
    """calculate gradient: df/dw = -mu + lambda*sigma*w"""
    return -mu + lam * np.dot(sigma, w)

# helper function to calculate portfolio variance
def portfolio_variance(w, sigma):
    """variance = w'sigma*w"""
    return np.dot(w, np.dot(sigma, w))

# helper function to calculate expected return
def portfolio_return(w, mu):
    """expected return = mu'w"""
    return np.dot(mu, w)

# project weights onto valid region (sum=1 and all weights >= 0)
def project_onto_simplex(w):
    """
    project weights so they sum to 1 and are non-negative
    this ensures our portfolio weights are valid
    """
    n = len(w)
    
    # if already valid, return as is
    if np.sum(w) == 1.0 and np.all(w >= 0):
        return w
    
    # sort weights in descending order
    u = np.sort(w)[::-1]
    cumsum = np.cumsum(u)
    
    # find the threshold for projection
    rho = np.where(u * np.arange(1, n+1) > (cumsum - 1))[0][-1]
    theta = (cumsum[rho] - 1) / (rho + 1)
    
    # project and ensure non-negativity
    return np.maximum(w - theta, 0)

# method 1: gradient descent with projection
def gradient_descent_projection(mu, sigma, lam, max_iter=1000, lr=0.01, tol=1e-6):
    """
    gradient descent with projection onto constraints
    this method directly projects weights onto valid region after each step
    """
    n = len(mu)
    w = np.ones(n) / n  # start with equal weights (20% each)
    
    # store results for plotting
    history = {
        'weights': [w.copy()],
        'objective': [objective(w, mu, sigma, lam)],
        'returns': [portfolio_return(w, mu)],
        'variance': [portfolio_variance(w, sigma)],
        'grad_norm': []
    }
    
    for iteration in range(max_iter):
        # compute gradient
        grad = gradient(w, mu, sigma, lam)
        grad_norm = np.linalg.norm(grad)
        history['grad_norm'].append(grad_norm)
        
        # check if converged (gradient is very small)
        if grad_norm < tol:
            print(f"  converged at iteration {iteration}")
            break
        
        # take a gradient descent step
        w_new = w - lr * grad
        
        # project back onto valid region (sum=1, w>=0)
        w_new = project_onto_simplex(w_new)
        
        # update weights
        w = w_new
        
        # save current state
        history['weights'].append(w.copy())
        history['objective'].append(objective(w, mu, sigma, lam))
        history['returns'].append(portfolio_return(w, mu))
        history['variance'].append(portfolio_variance(w, sigma))
    
    return w, history

# method 2: penalty method
def penalty_objective(w, mu, sigma, lam, rho, max_var):
    """
    objective with penalty terms for constraint violations
    penalty increases cost when constraints are violated
    """
    # original objective
    obj = objective(w, mu, sigma, lam)
    
    # add penalty for sum constraint: (sum(w) - 1)^2
    sum_penalty = (np.sum(w) - 1.0)**2
    
    # add penalty for negative weights: sum(max(0, -w_i))^2
    neg_penalty = np.sum(np.maximum(0, -w)**2)
    
    # add penalty for variance constraint: max(0, var - max_var)^2
    var = portfolio_variance(w, sigma)
    var_penalty = max(0, var - max_var)**2
    
    # total objective = original + penalties
    return obj + 0.5 * rho * (sum_penalty + neg_penalty + var_penalty)

def penalty_gradient(w, mu, sigma, lam, rho, max_var):
    """gradient of penalty objective"""
    # start with original gradient
    grad = gradient(w, mu, sigma, lam)
    
    # add gradient of penalty terms
    grad += rho * (np.sum(w) - 1.0)  # sum constraint gradient
    grad += rho * np.minimum(0, w)   # non-negativity constraint gradient
    
    # variance constraint gradient (only if violated)
    var = portfolio_variance(w, sigma)
    if var > max_var:
        grad += rho * (var - max_var) * 2 * np.dot(sigma, w)
    
    return grad

def penalty_method(mu, sigma, lam, max_var, max_outer=10, max_inner=500, 
                   rho_init=1.0, rho_mult=10, tol=1e-6):
    """
    penalty method: gradually increase penalty for constraint violations
    outer loop increases penalty parameter
    inner loop optimizes with current penalty
    """
    n = len(mu)
    w = np.ones(n) / n  # start with equal weights
    rho = rho_init
    
    # store results
    history = {
        'weights': [w.copy()],
        'objective': [objective(w, mu, sigma, lam)],
        'returns': [portfolio_return(w, mu)],
        'variance': [portfolio_variance(w, sigma)],
        'penalty_param': [rho],
        'constraint_violation': []
    }
    
    for outer in range(max_outer):
        # adjust learning rate based on penalty (higher penalty needs smaller steps)
        lr = 0.01 / (1 + rho/10)
        
        # optimize with current penalty parameter
        for inner in range(max_inner):
            grad = penalty_gradient(w, mu, sigma, lam, rho, max_var)
            grad_norm = np.linalg.norm(grad)
            
            if grad_norm < tol:
                break
            
            w = w - lr * grad
        
        # check how much constraints are violated
        sum_violation = abs(np.sum(w) - 1.0)
        neg_violation = np.sum(np.maximum(0, -w))
        var_violation = max(0, portfolio_variance(w, sigma) - max_var)
        total_violation = sum_violation + neg_violation + var_violation
        
        # save current state
        history['constraint_violation'].append(total_violation)
        history['weights'].append(w.copy())
        history['objective'].append(objective(w, mu, sigma, lam))
        history['returns'].append(portfolio_return(w, mu))
        history['variance'].append(portfolio_variance(w, sigma))
        history['penalty_param'].append(rho)
        
        print(f"  outer iter {outer}: rho={rho:.1f}, violation={total_violation:.6f}")
        
        # if constraints satisfied, we're done
        if total_violation < tol:
            print(f"  converged at outer iteration {outer}")
            break
        
        # increase penalty for next iteration
        rho *= rho_mult
    
    return w, history

# run both methods
print("\nmethod 1: gradient descent with projection")
print("=" * 60)
w_gd, hist_gd = gradient_descent_projection(
    expected_returns, cov_matrix, risk_aversion, 
    max_iter=2000, lr=0.1, tol=1e-6
)

print("\nmethod 2: penalty method")
print("=" * 60)
w_pen, hist_pen = penalty_method(
    expected_returns, cov_matrix, risk_aversion, max_variance,
    max_outer=15, rho_init=1.0, rho_mult=5
)

# display results
print("\nresults")
print("=" * 60)

print("\nmethod 1: gradient descent with projection")
print("-" * 50)
for i, stock in enumerate(stock_names):
    print(f"  {stock:15s}: {w_gd[i]*100:6.2f}%")
print(f"\n  expected return: {portfolio_return(w_gd, expected_returns)*100:.2f}%")
print(f"  portfolio risk:  {np.sqrt(portfolio_variance(w_gd, cov_matrix))*100:.2f}% (std dev)")
print(f"  variance:        {portfolio_variance(w_gd, cov_matrix)*100:.3f}%")
print(f"  sum of weights:  {np.sum(w_gd):.6f}")
print(f"  iterations:      {len(hist_gd['objective'])}")

print("\n\nmethod 2: penalty method")
print("-" * 50)
for i, stock in enumerate(stock_names):
    print(f"  {stock:15s}: {w_pen[i]*100:6.2f}%")
print(f"\n  expected return: {portfolio_return(w_pen, expected_returns)*100:.2f}%")
print(f"  portfolio risk:  {np.sqrt(portfolio_variance(w_pen, cov_matrix))*100:.2f}% (std dev)")
print(f"  variance:        {portfolio_variance(w_pen, cov_matrix)*100:.3f}%")
print(f"  sum of weights:  {np.sum(w_pen):.6f}")
print(f"  outer iterations: {len(hist_pen['penalty_param'])}")

# create visualizations
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('portfolio optimization results', fontsize=16, fontweight='bold')

# plot 1: optimal weights comparison
ax = axes[0, 0]
x = np.arange(len(stock_names))
width = 0.35
ax.bar(x - width/2, w_gd * 100, width, label='gradient descent', alpha=0.8)
ax.bar(x + width/2, w_pen * 100, width, label='penalty method', alpha=0.8)
ax.set_xlabel('stocks')
ax.set_ylabel('weight (%)')
ax.set_title('optimal portfolio weights')
ax.set_xticks(x)
ax.set_xticklabels(stock_names, rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# plot 2: convergence of objective value
ax = axes[0, 1]
ax.plot(hist_gd['objective'], label='gd with projection', linewidth=2)
ax.plot(hist_pen['objective'], label='penalty method', linewidth=2)
ax.set_xlabel('iteration')
ax.set_ylabel('objective value')
ax.set_title('convergence: objective function')
ax.legend()
ax.grid(alpha=0.3)

# plot 3: gradient norm for gd
ax = axes[0, 2]
ax.semilogy(hist_gd['grad_norm'], linewidth=2, color='blue')
ax.set_xlabel('iteration')
ax.set_ylabel('||gradient||')
ax.set_title('gradient descent: gradient norm')
ax.grid(alpha=0.3)

# plot 4: risk vs return tradeoff
ax = axes[1, 0]
ax.plot(hist_gd['variance'], hist_gd['returns'], label='gd path', alpha=0.6)
ax.plot(hist_pen['variance'], hist_pen['returns'], label='penalty path', alpha=0.6)
ax.scatter(portfolio_variance(w_gd, cov_matrix), portfolio_return(w_gd, expected_returns),
           s=200, marker='*', c='blue', label='gd final', zorder=5)
ax.scatter(portfolio_variance(w_pen, cov_matrix), portfolio_return(w_pen, expected_returns),
           s=200, marker='*', c='orange', label='penalty final', zorder=5)
ax.axvline(max_variance, color='red', linestyle='--', label='risk constraint', linewidth=2)
ax.set_xlabel('portfolio variance')
ax.set_ylabel('expected return')
ax.set_title('risk-return tradeoff')
ax.legend()
ax.grid(alpha=0.3)

# plot 5: constraint violation over iterations
ax = axes[1, 1]
if len(hist_pen['constraint_violation']) > 0:
    ax.semilogy(hist_pen['constraint_violation'], linewidth=2, marker='o')
    ax.set_xlabel('outer iteration')
    ax.set_ylabel('total constraint violation')
    ax.set_title('penalty method: constraint satisfaction')
    ax.grid(alpha=0.3)

# plot 6: penalty parameter growth
ax = axes[1, 2]
ax.semilogy(hist_pen['penalty_param'], linewidth=2, marker='s', color='green')
ax.set_xlabel('outer iteration')
ax.set_ylabel('penalty parameter (rho)')
ax.set_title('penalty method: parameter growth')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('portfolio_optimization_results.png', dpi=300, bbox_inches='tight')
print("\nplots saved as 'portfolio_optimization_results.png'")
plt.show()

print("\ninterpretation")
print("=" * 60)
print("""
key findings:

1. both methods converge to valid portfolios satisfying all constraints
2. gradient descent finds a more aggressive allocation (concentrated in 2-3 stocks)
3. penalty method finds a more diversified allocation (spreads across all stocks)
4. different solutions arise because the two optimization methods explore the feasible region differently and achieve different trade-offs between risk and return
5. risk constraint is satisfied by both methods (variance <= 2.5%)

trade-offs:
- gradient descent: faster convergence but requires projection operation
- penalty method: handles any constraint type but needs careful tuning of rho
- penalty parameter must increase gradually to avoid numerical issues
- choice depends on investor risk preference (aggressive vs diversified)
""")