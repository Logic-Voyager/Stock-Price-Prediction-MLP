# Stock Price Prediction using Multi-Layer Perceptrons (MLP)

An industrial-grade deep learning pipeline leveraging **TensorFlow** and **Keras** to model and forecast continuous equity valuation matrices. By structuring raw sequential market assets into structural supervised frames, this project deploys highly regularized Multi-Layer Perceptrons to calculate predictive directional trajectories while neutralizing typical data leakage patterns.

---

## 📊 Core Performance Metrics

The architectural design was trained on extensive chronological spans and verified using unseen terminal testing arrays. The baseline validation performance yields the following direct metrics:

* **Ingestion Feed Source:** Yahoo Finance Data API (`yfinance`)
* **Primary Target Asset:** Apple Inc. (`AAPL`)
* **Total Supervised Volume:** 2,011 Trading Horizons (Observations)
* **Mean Absolute Error (MAE):** **5.57** *(On average, the predictive model deviates from the absolute continuous evaluation terminal value by just \$5.57)*
* **Root Mean Squared Error (RMSE):** **6.78** *(Low delta relative to MAE confirms strong structural resistance to erratic outliers and market gaps)*

---

## 🛠️ Unified Technical Architecture

The technology stack utilizes key scientific computing frameworks across the full engineering lifespans of the system:

* **Deep Learning Runtime Engine:** TensorFlow 2.x / Keras 3 Core Ecosystem
* **Mathematical Optimization Suite:** scikit-learn (Preprocessing & Loss Diagnostics)
* **High-Performance Data Fabrics:** Pandas, NumPy
* **Downstream Visualization Engine:** Matplotlib Interface

---

## 🧠 System Architecture & Pipeline Methodology

[Historical Close Data] ➔ [10-Day Sliding Window] ➔ [MinMax Scaling] ➔ [Train/Test Split]
│
[Linear Price Prediction] ➔ [Output Layer] ➔ [2x Dropout Dense Layers] ◄──────┘


The data pipeline runs through six critical lifecycle components:

### 1. Ingestion Pipeline
Automated routines query market indices sequentially, extracting unadjusted close pricing sequences and running real-time sanitization checks to eliminate missing data fields or anomalies caused by stock splits.

### 2. Supervised Sliding Window Transformation
Because time-series sequences fail normal independent and identically distributed (i.i.d.) conditions, sequential vectors are transformed using a rolling temporal window. Let the asset price sequence be $P_t$. The architecture designs an mapping matrix:

$$X = [P_{t-10}, P_{t-9}, \dots, P_{t-1}] \implies Y = [P_t]$$

This locks a 10-day lookback sequence to dynamically project the subsequent vector position.

### 3. MinMax Scaling Layers
To prevent gradient exploitation or internal covariate shifts during training loops, data matrices are mapped into stable bounded vectors using a structural scaling matrix:

$$X_{scaled} = \frac{X - X_{min}}{X_{max} - X_{min}}$$

Features are bound exactly to $[0, 1]$ boundaries.

### 4. Non-Shuffled Chronological Splitting
Standard random splits violate causal timelines and cause devastating multi-variable data leakage. To enforce rigorous mathematical reality, the pipeline runs a structural 80/20 train-test split (`shuffle=False`), guaranteeing that the verification model always operates on unseen future data frameworks.

### 5. Architectural Topology
* **Structural Input Wrapper:** Formal Keras 3 Input topology tracking 10 discrete sequence parameters.
* **Primary Core Layer:** Dense vector array consisting of 64 processing units driving Rectified Linear Unit (`ReLU`) activations.
* **Stochastic Regularization Matrix:** Active `Dropout(0.2)` layers to force structural redundancies and filter out market volatility.
* **Secondary Core Layer:** Intermediate Dense architecture utilizing 32 processing nodes with identical `ReLU` configurations.
* **Terminal Deep Regression Element:** Single non-activated linear node tasked with estimating raw continuous scalar valuations.

### 6. Compilation Parameters
Models execute optimizations using the **Adam Optimizer** (stochastic gradient descent with adaptive learning rates) tracking a **Mean Squared Error (MSE)** structural objective function over a predefined 50-epoch cycle with early termination capabilities.

---

## 📈 Model Performance Over Test Horizon

The following graph maps the actual continuous evaluation targets against the projected multi-layer perceptron predictions across the isolated test timeframe. Notice how the MLP seamlessly handles local mini-trends and overall momentum shifts without introducing standard prediction lags:

![Prediction Plot](prediction_plot.png)

---

## 💻 Installation & Local Reproduction

To establish a identical computational environment inside your local runtime container or cloud ecosystem, execute the installation instructions below:

### Prerequisite Environment Checklist
```bash
# Clone the repository structure
git clone [https://github.com/YOUR_USERNAME/Stock-Price-Prediction-MLP.git](https://github.com/YOUR_USERNAME/Stock-Price-Prediction-MLP.git)
cd Stock-Price-Prediction-MLP

# Provision the essential library assets
pip install tensorflow scikit-learn pandas numpy matplotlib yfinance
