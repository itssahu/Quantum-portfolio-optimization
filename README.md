# Quantum-portfolio-optimization
This project combines Quantum Generative Adversarial Networks (QGANs) and Variational Quantum Eigensolver (VQE) for portfolio optimization using real financial time series. The goal is to generate plausible synthetic returns using QGANs and perform quantum-enhanced risk minimization using VQE, comparing outcomes with classical optimization (cvxpy).
## 1. Dataset

- **Assets Used**:
  - Equities: `SPY`, `AAPL`, `MSFT`
  - Cryptocurrencies: `BTC-USD`, `ETH-USD`
- **Source**: Yahoo Finance API
- **Time Range**: `2022-01-01` to `2024-12-31`
- **Transformation**: 
  - Adjusted close prices extracted
  - Log returns computed: `log(1 + pct_change)`
- **Saved Files**:
  - `data/historical_prices.csv`
  - `data/log_returns.csv`

## 2. Synthetic Data Generation using QGAN
### 2.1 Architecture
- **Quantum Generator**: 
  - `AngleEmbedding` + `BasicEntanglerLayers` (Pennylane)
  - 4 qubits, 3 layers
  - Output from PauliZ expectation on one qubit
- **Classical Discriminator**:
  - 1 hidden layer MLP

### 2.2 Training Setup
- **Epochs**: 1000
- **Loss Function**: Binary Cross-Entropy
- **Optimizers**: Adam (lr = 0.003)
- **Output**: 500 synthetic samples per asset
### 2.3 Real vs Generated Distribution
![image](https://github.com/user-attachments/assets/1ba006e2-2e69-439f-9615-76d659bb80e0)
![image](https://github.com/user-attachments/assets/d21a8f22-1166-4b15-8b56-fd6504c1a99c)
![image](https://github.com/user-attachments/assets/a3834b48-f430-47ca-9a90-2e8e0dd670c8)
![image](https://github.com/user-attachments/assets/1b03abfc-9989-4ca7-984f-47a513ff799f)
![image](https://github.com/user-attachments/assets/60efda99-1285-406d-a6a3-6aa83bcb3ab9)
## 3. Covariance Matrix Computation

- **Input**: QGAN-generated returns
- **Output**: `cov_matrix.npy` used in VQE
![image](https://github.com/user-attachments/assets/46898862-1232-4367-b5f8-34d55425580f)
## 4.Portfolio Optimization using VQE
### 4.1 Objective Function
Minimize portfolio variance:
\[
\min_w \; w^\top \Sigma w
\]

### 4.2 Circuit Details
- **Ansatz**: RY rotations + CNOTs
- **Qubits**: 5 (1 per asset)
- **Optimizer**: Nesterov Momentum
- **Steps**: 100

### 4.3 Portfolio Weights (Final)

| Asset     | Weight |
|-----------|--------|
| SPY       | 0.1119 |
| AAPL      | 0.6641 |
| MSFT      | 0.1729 |
| BTC-USD   | 0.0402 |
| ETH-USD   | 0.0110 |
![image](https://github.com/user-attachments/assets/2f87d0b3-a114-442e-8101-4eff3e4ca87d)
## 5. Backtesting & Benchmarking
### 5.1 Quantum Portfolio Performance
- Cumulative returns using VQE-optimized weights
- Real returns used for backtesting
![image](https://github.com/user-attachments/assets/8d6ccb6d-34fd-4520-af1f-0733ad8f59f4)
### 5.2 Comparison with Classical CVXPY Portfolio

| Method     | Description                            |
|------------|----------------------------------------|
| VQE        | Quantum risk-minimized portfolio       |
| CVXPY      | Convex optimization (classical)        |
![image](https://github.com/user-attachments/assets/3a9c2370-af9b-49f1-8901-f5191253bf2d)
## 6. Advanced Analyses
### 6.1 Correlation Matrix (QGAN Data)
![image](https://github.com/user-attachments/assets/48b63ff3-3e88-4bb0-8515-67287f15ef3a)
### 6.2 PCA on Generated Returns
![image](https://github.com/user-attachments/assets/b50030fe-8584-482f-ad18-e95a1c1e75ed)
- **Explained Variance**:
  - PC1: 70.2%
  - PC2: 25.4%
## 7. Performance Metrics

| Metric                       | Quantum VQE | Classical CVXPY |
|-----------------------------|-------------|------------------|
| Sharpe Ratio                | 0.028       | 0.030            |
| Max Drawdown                | -35.31%     | -26.19%          |
## 8.  Industry Applications

- **Asset Management**: Synthetic stress-testing and low-risk portfolio design
- **Risk Modeling**: Simulate alternative scenarios in volatile markets
- **Fintech R&D**: Explore quantum solutions for investment management
- **Hybrid Modeling**: Combines quantum computing with deep learning techniques for financial engineering
## 9. Technologies Used

- **Quantum**: Pennylane, Qiskit
- **Deep Learning**: PyTorch
- **Classical Optimization**: CVXPY
- **Visualization**: Matplotlib, Seaborn
- **Finance API**: yfinance

## 12. Summary

This project demonstrates the potential of quantum hybrid models in finance. By combining QGAN for synthetic return simulation and VQE for portfolio optimization, we establish a robust and interpretable pipeline that is benchmarked against classical methods.

While QGANs require further tuning to match the distribution of real returns exactly, the portfolio optimization framework delivers plausible, backtestable strategies, highlighting the industrial potential of quantum computing in financial modeling.


