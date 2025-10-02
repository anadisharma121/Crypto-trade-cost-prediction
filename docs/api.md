# API Documentation

## Endpoint: `/compute`

* **Method:** `POST`
* **Description:** Accepts trade execution parameters and returns simulation results including slippage, fees, market impact, and latency.

### Request

#### Headers

```http
Content-Type: application/json
```

#### Body Parameters

| Name          | Type   | Description                    |
| ------------- | ------ | ------------------------------ |
| exchange      | string | Exchange name (e.g., "okx")    |
| asset         | string | Spot pair (e.g., "BTC-USDT")   |
| order\_type   | string | Order type (fixed as "market") |
| quantity\_usd | float  | Order value in USD             |
| volatility    | float  | Price volatility \[0.0 - 1.0]  |
| fee\_tier     | string | Fee category (e.g., "Tier 1")  |

#### Example Request

```json
{
  "exchange": "okx",
  "asset": "BTC-USDT",
  "order_type": "market",
  "quantity_usd": 1000,
  "volatility": 0.3,
  "fee_tier": "Tier 1"
}
```

### Response

Returns JSON object containing simulation outcomes.

| Field                    | Type  | Description                                  |
| ------------------------ | ----- | -------------------------------------------- |
| expected\_slippage       | float | Predicted cost due to execution inefficiency |
| expected\_fees           | float | Estimated transaction fees                   |
| expected\_market\_impact | float | Cost from price impact during trade          |
| net\_cost                | float | Total cost of execution                      |
| maker\_taker\_ratio      | float | Probability-weighted maker/taker ratio       |
| latency\_ms              | float | Backend computation latency (ms)             |
| execution\_path          | list  | Remaining inventory over time                |

#### Example Response

```json
{
  "expected_slippage": 3.25,
  "expected_fees": 1.12,
  "expected_market_impact": 4.07,
  "net_cost": 8.44,
  "maker_taker_ratio": 0.42,
  "latency_ms": 27.5,
  "execution_path": [1000, 750, 500, 250, 0]
}
```

## Error Handling

* Returns HTTP 400 for malformed requests.
* Errors are returned as:

```json
{
  "error": "Descriptive error message"
}
```

## Notes

* Ensure backend is running and accessible before sending requests.
* UI assumes the `/compute` endpoint returns promptly for latency benchmarking.

