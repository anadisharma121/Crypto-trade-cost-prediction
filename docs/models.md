# Model and Algorithm Documentation

##  Almgren-Chriss Market Impact Model

### Purpose

The Almgren-Chriss model provides a framework for optimal trade execution that minimizes total cost by balancing market impact and execution risk.

### Core Concepts

* **Temporary Market Impact:** Transient effect that decays after execution.
  $T(v) = \eta \cdot v^\alpha$
* **Permanent Market Impact:** Lasting change in price due to executed volume.
  $P(v) = \gamma \cdot v^\beta$
* **Execution Risk:** Uncertainty due to price volatility while the trade is being executed.
* **Hamiltonian Cost Function:** Combines all components to find optimal execution.

### Parameters

* $\alpha, \beta$: curvature parameters
* $\eta, \gamma$: impact sensitivity coefficients
* $\lambda$: risk aversion factor
* $\sigma$: price volatility
* $\Delta t$: time step duration

### Formula

The model minimizes the following objective function:

$$
	x^* = \arg\min_x \left[ \mathbb{E}[C(x)] + \lambda \cdot \text{Var}[C(x)] \right]
$$

Where:

* $C(x)$: cost of execution
* $\lambda$: risk aversion parameter
* $x$: execution schedule

### Implementation Notes

* We simulate execution across time intervals with variable trade sizes
* Optimal schedule minimizes expected cost + risk

---

##  Algorithmic Techniques Used

### Dynamic Programming

Used in Almgren-Chriss to compute optimal trade schedule across time periods while accounting for market impact and volatility.

### Linear Regression

Used for slippage estimation based on quantity, volatility, and tier. Trained using synthetic or historical data.

### Logistic Regression

Used to estimate the likelihood of an order being filled as a maker or taker. Influences fee calculation and net execution cost.

### Vectorized NumPy Computation

* Optimizes runtime performance
* Reduces memory usage compared to iterative loops

---

##  Market Impact Calculation Methodology

We use the **Almgren-Chriss optimal execution framework** to estimate market impact, which splits a large order into smaller child orders across time steps.

### Formula

Total market impact cost $C$ is calculated as:

$$
C = \sum_{t=1}^{T} \left( \eta \cdot x_t^2 + \gamma \cdot \sum_{s=1}^{t} x_s \right)
$$

Where:

* $T$: total time steps
* $x_t$: quantity traded at time $t$
* $\eta$: temporary impact coefficient
* $\gamma$: permanent impact coefficient

### Method Summary

* **Permanent Impact:** Models how large trades shift the market price permanently.
* **Temporary Impact:** Models short-term price changes due to liquidity consumption.

### Practical Assumptions

* Linear price impact (for tractability)
* Constant volatility across execution horizon
* Constant spread and liquidity per time step

----

##  Slippage Estimation Model

### Purpose

To estimate additional cost due to execution inefficiency using a regression approach.

### Technique

* **Linear Regression**
* Inputs: Volatility, fee tier, order size
* Output: Predicted slippage cost in USD

### Benefits

* Fast to train and evaluate
* Interpretable coefficients

---

##  Maker/Taker Participation Prediction

### Purpose

To estimate the probability of an order being executed as a maker vs. taker, influencing fee computation.

### Technique

* **Logistic Regression**
* Features: Order size, volatility
* Output: Maker/taker participation probability

### Usage

* Used to compute weighted transaction fee in the cost model

---

## Summary of Model Choices

| Model Type             | Technique Used      | Reasoning                           |
| ---------------------- | ------------------- | ----------------------------------- |
| Market Impact          | Almgren-Chriss      | Robust optimal execution            |
| Slippage Estimation    | Linear Regression   | Simplicity and speed                |
| Maker/Taker Prediction | Logistic Regression | Probabilistic output and efficiency |

