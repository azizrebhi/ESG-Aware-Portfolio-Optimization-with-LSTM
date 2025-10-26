from src.data_loader import load_price_data, get_esg_scores
from src.preprocessing import compute_returns, compute_covariance
from src.optimizer import optimize_portfolio
from src.lstm_model import LSTMReturnPredictor, prepare_data, train_lstm
from src.evaluation import evaluate_portfolio
import torch

# Step 1: Load Data
tickers = ["AAPL", "MSFT", "GOOGL", "TSLA"]
data = load_price_data(tickers)
esg_scores = get_esg_scores(tickers)

# Step 2: Preprocessing
returns = compute_returns(data)
Sigma = compute_covariance(returns)
mu = returns.mean().values

# Step 3: ESG-Aware Optimization (Baseline)
weights = optimize_portfolio(mu, Sigma, esg_scores)
print("\n✅ Optimal Portfolio Weights:\n", weights.round(3))

# Step 4: Evaluate
evaluate_portfolio(returns, weights)

# Step 5: LSTM Model
X_tensor, y_tensor, scaler = prepare_data(returns)
model = LSTMReturnPredictor(input_dim=len(tickers))
model = train_lstm(model, X_tensor, y_tensor)

# Step 6: Predict & Re-optimize
predicted_returns = model(X_tensor[-1:].detach()).detach().numpy().flatten()
final_weights = optimize_portfolio(predicted_returns, returns.cov().values, esg_scores, alpha=0.2, beta=0.15)
print("\n✅ Optimal LSTM-Driven Portfolio:\n", final_weights.round(3))
