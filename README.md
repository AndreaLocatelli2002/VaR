# VaR

This repository provides a minimal example for computing Value at Risk (VaR) for a stock portfolio.

## Data loading

`load_data.load_prices` uses `yfinance` to download close prices:

```python
import load_data

prices = load_data.load_prices(["AAPL", "MSFT"], period="1y", interval="1d")
```

## Usage

```bash
python main.py
```

Update `main.py` with your ticker list, weights, and desired date/interval parameters.
