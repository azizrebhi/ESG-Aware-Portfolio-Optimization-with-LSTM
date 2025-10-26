import cvxpy as cp
import pandas as pd
import numpy as np

def optimize_portfolio(mu, Sigma, esg_scores, alpha=0.3, beta=0.15):
    n = len(mu)
    w = cp.Variable(n)
    
    risk = cp.quad_form(w, Sigma)
    ret = mu @ w
    
    esg_norm = (esg_scores - esg_scores.min()) / (esg_scores.max() - esg_scores.min())
    esg = esg_norm.values
    
    objective = cp.Maximize(ret - alpha * risk + beta * (esg @ w))
    constraints = [cp.sum(w) == 1, w >= 0]
    
    prob = cp.Problem(objective, constraints)
    prob.solve()
    
    return pd.Series(w.value, index=esg_scores.index)
