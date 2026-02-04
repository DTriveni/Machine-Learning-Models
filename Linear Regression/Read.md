# Linear Regression: Manual Implementation & scikit-learn

This repository contains a Jupyter Notebook demonstrating **linear regression** using both a **manual implementation** (pseudo-inverse) and **scikit-learn's LinearRegression**. The notebook uses the **Diabetes dataset** from `sklearn.datasets` as an example.

---

## 🧩 Project Overview

The notebook covers:

1. Loading and exploring the **Diabetes dataset**.
2. Preparing **feature and target matrices** for modeling.
3. Adding an **intercept column** for linear regression.
4. Implementing **manual linear regression** using the pseudo-inverse (`np.linalg.pinv`).
5. Training and predicting with **scikit-learn LinearRegression**.
6. Comparing predictions and evaluating the model using:
   - **Mean Squared Error (MSE)**
   - **Root Mean Squared Error (RMSE)**
   - **R² score**
7. Visualizing predictions vs actual values using **scatter plots**.

---

## 📦 Requirements

- Python 3.x  
- Jupyter Notebook / JupyterLab  
- Libraries:
  bash
  numpy
  pandas
  matplotlib
  scikit-learn
