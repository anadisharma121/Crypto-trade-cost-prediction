from fastapi import FastAPI
from pydantic import BaseModel
import asyncio
import websockets
import time
import json
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression

# ========== Almgren-Chriss Model Functions ==========

def temporary_impact(volume, alpha, eta):
    return eta * volume ** alpha

def permanent_impact(volume, beta, gamma):
    return gamma * volume ** beta

def hamiltonian(inventory, sell_amount, risk_aversion, alpha, beta, gamma, eta, volatility=0.3, time_step=0.5):
    temp_impact = risk_aversion * sell_amount * permanent_impact(sell_amount / time_step, beta, gamma)
    perm_impact = risk_aversion * (inventory - sell_amount) * time_step * temporary_impact(sell_amount / time_step, alpha, eta)
    exec_risk = 0.5 * (risk_aversion ** 2) * (volatility ** 2) * time_step * ((inventory - sell_amount) ** 2)
    return temp_impact + perm_impact + exec_risk

def optimal_execution(time_steps, total_shares, risk_aversion, alpha, beta, gamma, eta, volatility=0.3, mid_price=1.0):
    value_function = np.zeros((time_steps, total_shares + 1), dtype="float64")
    best_moves = np.zeros((time_steps, total_shares + 1), dtype="int")
    time_step_size = 0.5
    for shares in range(total_shares + 1):
        value_function[time_steps - 1, shares] = np.exp(shares * temporary_impact(shares / time_step_size, alpha, eta))
        best_moves[time_steps - 1, shares] = shares
    for t in range(time_steps - 2, -1, -1):
        for shares in range(total_shares + 1):
            best_value = float("inf")
            best_share_amount = 0
            for n in range(shares + 1):
                if shares - n >= 0:
                    h = hamiltonian(shares, n, risk_aversion, alpha, beta, gamma, eta, volatility)
                    cost = value_function[t + 1, shares - n] * np.exp(h)
                    if cost < best_value:
                        best_value = cost
                        best_share_amount = n
            value_function[t, shares] = best_value
            best_moves[t, shares] = best_share_amount
    current_inventory = total_shares
    execution_path = []
    for t in range(time_steps):
        move = best_moves[t, current_inventory]
        execution_path.append(move)
        current_inventory -= move
        if current_inventory <= 0:
            break
    total_cost = 0
    inventory = total_shares
    for sell in execution_path:
        temp_cost = temporary_impact(sell / time_step_size, alpha, eta)
        perm_cost = permanent_impact(sell / time_step_size, beta, gamma)
        exec_risk = 0.5 * (risk_aversion ** 2) * (volatility ** 2) * time_step_size * (inventory ** 2)
        cost_per_step = temp_cost + perm_cost + exec_risk
        total_cost += cost_per_step
        inventory -= sell
    market_impact_usd = total_cost * mid_price
    return execution_path, market_impact_usd


# ========== FastAPI Setup ==========

app = FastAPI()

class InputParams(BaseModel):
    exchange: str
    asset: str
    order_type: str
    quantity_usd: float
    volatility: float
    fee_tier: str

FEE_TIERS = {
    "Tier 1": 0.001,
    "Tier 2": 0.0007,
    "Tier 3": 0.0005
}

async def fetch_orderbook(asset, retries=5):
    url = f"wss://ws.gomarket-cpp.goquant.io/ws/l2-orderbook/okx/{asset}-SWAP"
    for _ in range(retries):
        try:
            async with websockets.connect(url) as ws:
                raw = await ws.recv()
                data = json.loads(raw)
                asks = [[float(p), float(q)] for p, q in data.get("asks", [])[:5]]
                bids = [[float(p), float(q)] for p, q in data.get("bids", [])[:5]]
                return asks, bids
        except:
            await asyncio.sleep(2)
    raise RuntimeError("Orderbook fetch failed.")

@app.post("/compute")
async def compute_metrics(params: InputParams):
    start_time = time.perf_counter()
    asset = params.asset
    qty_usd = params.quantity_usd
    volatility = params.volatility
    fee = FEE_TIERS.get(params.fee_tier, 0.001)

    try:
        asks, bids = await fetch_orderbook(asset)
    except Exception as e:
        return {"error": str(e)}

    mid_price = (asks[0][0] + bids[0][0]) / 2
    quantity_asset = qty_usd / mid_price

    # Slippage
    prices = np.array([p for p, _ in asks] + [p for p, _ in bids])
    sizes = np.array([q for _, q in asks] + [q for _, q in bids])
    model = LinearRegression().fit(sizes.reshape(-1, 1), prices)
    slippage = model.predict(np.array([[quantity_asset]]))[0] - mid_price

    # Fees
    expected_fees = qty_usd * fee

    # Market Impact
    execution_path, market_impact = optimal_execution(
      time_steps=10,
      total_shares=int(quantity_asset),
      risk_aversion=0.01,
      alpha=1,
      beta=1,
      gamma=0.05,
      eta=0.05,
      volatility=volatility,
      mid_price=mid_price
    )

    # Net Cost
    net_cost = slippage + expected_fees + market_impact

    # Maker/Taker
    maker_taker_model = LogisticRegression().fit(np.random.rand(10, 2), np.random.randint(0, 2, 10))
    maker_prob = maker_taker_model.predict_proba([[volatility, qty_usd]])[0][1]

    # Latency
    latency = (time.perf_counter() - start_time) * 1000

    return {
        "expected_slippage": float(slippage),
        "expected_fees": float(expected_fees),
        "expected_market_impact": float(market_impact),
        "net_cost": float(net_cost),
        "maker_taker_ratio": float(maker_prob),
        "latency_ms": float(latency),
        "execution_path": [int(x) for x in execution_path]
    }

