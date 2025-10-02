# Optimal Crypto Trade Cost Simulator

## 📌 Overview

This project simulates the execution cost of large cryptocurrency market orders by integrating financial models with machine learning predictions. It combines:

* The **Almgren-Chriss optimal execution model** for market impact.
* Regression-based techniques to estimate **slippage** and **maker/taker ratios**.
* A web interface (Streamlit) for user interaction.

The goal is to provide traders, quants, and researchers with insights into how trade size, volatility, and market fees impact cost.

---

## ✨ Features

* ✅ Real-time simulation of crypto trade execution
* 📉 Estimates slippage, market impact, fees, and net cost
* 📊 Visualizes execution paths over time
* 🕒 Measures and displays latency metrics
* 🧠 Uses regression models for slippage prediction and maker/taker classification
* 📡 Fully functional via a web-based frontend

---

## 🧰 Technologies Used

| Layer    | Tech Stack                        |
| -------- | --------------------------------- |
| Frontend | Streamlit (Python)                |
| Backend  | FastAPI, scikit-learn, numpy      |
| Hosting  | Google Colab, Uvicorn, pyngrok    |
| Modeling | Almgren-Chriss, Linear Regression |

---

## 🚀 Getting Started

### 1. Run Backend (Google Colab)

* Open `backend.ipynb` in Google Colab
* Run all cells (sets up FastAPI + exposes via ngrok)
* Copy the ngrok public URL (e.g., `https://xyz.ngrok-free.app`)

### 2. Run Frontend (Locally)

```bash
pip install streamlit requests
streamlit run frontend.py
```

* Use the ngrok URL as the backend URL
* Enter simulation parameters and click **"Run Simulation"**

---

## 🧱 Project Structure

```
├── frontend.py               # Streamlit app UI
├── backend.ipynb             # Backend (FastAPI) in Colab
├── docs/
│   ├── models.md             # Model and algorithm explanation
│   ├── performance.md        # Optimization and latency benchmarking
│   ├── api.md                # API endpoint usage
│   └── deployment.md         # Step-by-step deployment guide
├── README.md
```

---

## 👥 Contributors

* Anadi Sharma

---

## 📄 License

This project is licensed under the **MIT License**. Feel free to reuse and contribute!
