from __future__ import annotations

import load_data
from portfolio_var import compute_portfolio_var


def main() -> None:
    # Example usage: update tickers and weights for your portfolio
    tickers = ["AAPL", "MSFT"]
    prices_df = load_data.load_prices(tickers, period="1y", interval="1d")

    weights = [1 / len(tickers)] * len(tickers)

    var_95 = compute_portfolio_var(
        prices_df.values,
        weights,
        confidence=0.95,
        method="historical",
    )
    print(f"95% historical VaR: {var_95:.4f}")

    var_95_param = compute_portfolio_var(
        prices_df.values,
        weights,
        confidence=0.95,
        method="parametric",
    )
    print(f"95% parametric VaR: {var_95_param:.4f}")


if __name__ == "__main__":
    main()
