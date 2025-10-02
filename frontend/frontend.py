import streamlit as st
import requests
import json
import os

st.set_page_config("Trade Cost Simulator", layout="wide")
st.title("Optimal Crypto Trade Cost Simulator")

backend_url = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000")

st.sidebar.header("Input Parameters")
exchange = st.sidebar.selectbox("Exchange", ["okx"])
asset = st.sidebar.selectbox("Spot Asset", ["BTC-USDT", "ETH-USDT"])
order_type = "market"
quantity = st.sidebar.number_input("Order Quantity (USD)", 100.0)
volatility = st.sidebar.slider("Volatility", 0.0, 1.0, 0.3)
fee_tier = st.sidebar.selectbox("Fee Tier", ["Tier 1", "Tier 2", "Tier 3"])


if st.sidebar.button("Run Simulation"):
    with st.spinner("Computing execution cost..."):
        if backend_url:
            payload = {
                "exchange": exchange,
                "asset": asset,
                "order_type": order_type,
                "quantity_usd": quantity,
                "volatility": volatility,
                "fee_tier": fee_tier
            }

            try:
                compute_url = f"{backend_url}/compute"
                response = requests.post(compute_url, json=payload)
                response.raise_for_status()
                result = response.json()

                if "error" in result:
                    st.error(result["error"])
                else:
                    st.subheader("Simulation Results")
                    st.metric("Expected Slippage", f"{result['expected_slippage']:.4f} USD")
                    st.metric("Expected Fees", f"{result['expected_fees']:.4f} USD")
                    st.metric("Market Impact", f"{result['expected_market_impact']:.4f} USD")
                    st.metric("Net Cost", f"{result['net_cost']:.4f} USD")
                    st.metric("Maker/Taker Ratio", f"{result['maker_taker_ratio']:.2%}")
                    st.metric("Internal Latency", f"{result['latency_ms']:.2f} ms")
                    st.subheader("Execution Path Over Time")
                    st.line_chart(result["execution_path"])
                    
            except requests.exceptions.RequestException as e:
                st.error(f"Request to backend failed: {e}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
        else:
            st.warning("Backend URL is not configured.")