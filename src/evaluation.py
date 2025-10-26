import matplotlib.pyplot as plt

def evaluate_portfolio(returns, weights):
    portfolio_returns = (returns * weights).sum(axis=1)
    cumulative = (1 + portfolio_returns).cumprod()
    
    sharpe = portfolio_returns.mean() / portfolio_returns.std()
    
    plt.figure(figsize=(10, 5))
    plt.plot(cumulative, label="Dynamic ESG Portfolio", linewidth=2)
    plt.title("Portfolio Growth (Dynamic ESG + Markowitz)")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    print(f"Sharpe Ratio: {sharpe:.3f}")
    return sharpe
