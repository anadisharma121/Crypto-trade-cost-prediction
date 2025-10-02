# Deployment Guide

This document explains how to deploy both the backend and frontend components of the crypto trade cost simulator.

---

## Backend Deployment (Google Colab)

### Prerequisites

* Google Account
* Colab environment access
* Ngrok account (for stable public URL)

### Steps

1. Open the provided notebook `backend.ipynb` in Google Colab.
2. Run each cell sequentially:

   * Install dependencies (`fastapi`, `uvicorn`, `nest_asyncio`, `pyngrok`, `sklearn`, etc.)
   * Initialize FastAPI app
   * Launch server using `uvicorn` and expose via `pyngrok`
3. Copy the public URL printed by ngrok (e.g., `https://abc123.ngrok-free.app`).

> ⚠️ Keep the notebook running during use. Colab will shut down inactive sessions.

---

## Frontend Deployment (Local Machine)

### Prerequisites

* Python 3.8+
* `streamlit` and `requests` libraries installed

### Steps

1. Clone or download the project locally.
2. Open terminal and navigate to the project root.
3. Run the Streamlit frontend:

   ```bash
   streamlit run frontend.py
   ```
4. When prompted or in code, set `backend_url` to the ngrok URL from the backend.
5. Enter parameters in sidebar and run simulation.

---

## Optional: Local Backend Deployment

If running backend outside Colab (e.g., on a VPS):

1. Install dependencies using `pip install -r requirements.txt`
2. Run FastAPI server locally:

   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```
3. Update the frontend `backend_url` to point to `http://localhost:8000` or public IP.

---

## Notes

* For production-grade deployment, consider:

  * Using Docker
  * Setting up HTTPS and reverse proxies
  * Deploying on AWS/GCP with autoscaling

* Ensure CORS settings allow frontend requests.

## Health Check

* Hit `/compute` with test payload and verify JSON response.
* Confirm frontend response renders in under 1 second for small inputs.

