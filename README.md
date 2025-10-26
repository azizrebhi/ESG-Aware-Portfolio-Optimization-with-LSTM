# ESG-Aware Portfolio Optimization with LSTM

> Dynamic portfolio optimization integrating ESG (Environmental, Social, Governance) scores using LSTM-based return prediction and convex optimization.

Traditional portfolio optimization (e.g., Markowitz) often ignores ESG factors and uses static expected returns. This project implements a dynamic, ESG-aware portfolio using LSTM networks to predict future asset returns, convex optimization (CVXPY) to maximize expected return, minimize risk, and incorporate ESG scores, and backtesting to evaluate cumulative returns, Sharpe ratio, and ESG alignment.

---

## Pipeline

Market Data → Feature Engineering → LSTM Return Prediction → ESG-Aware Optimization → Evaluation & Visualization

**Steps:**
1. Download historical price data and ESG scores for selected tickers.
2. Compute returns, smooth with rolling averages, and calculate covariance.
3. Train an LSTM model to predict next-day returns.
4. Use CVXPY to find optimal portfolio weights balancing return, risk, and ESG.
5. Backtest and visualize portfolio performance.

---

## Tech Stack

| Category       | Tools / Libraries                      |
|----------------|---------------------------------------|
| Data           | yfinance, pandas, numpy               |
| Modeling       | PyTorch (LSTM)                        |
| Optimization   | CVXPY                                 |
| Visualization  | Matplotlib                             |
| Environment    | Jupyter Notebook, Python 3.10+        |

---

## Installation

Clone the repository:

```bash
git clone https://github.com/azizrebhi/ESG-Aware-Portfolio-Optimization-with-LSTM.git
cd ESG-Aware-Portfolio-Optimization-with-LSTM
