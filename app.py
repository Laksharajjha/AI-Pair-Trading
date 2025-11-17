from flask import Flask, render_template, request, redirect, session
import os

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "devkey")

# Lazy imports so app doesn't crash on startup
def get_db():
    from db import users_col, strategies_col
    return users_col, strategies_col

@app.route("/health")
def health():
    return "ok", 200

@app.route("/", methods=["GET", "POST"])
def index():
    try:
        users_col, strategies_col = get_db()

        if request.method == "POST":
            from strategy import run_strategy
            ticker1 = request.form.get("ticker1") or "AAPL"
            ticker2 = request.form.get("ticker2") or "MSFT"
            period = request.form.get("period") or "6mo"
            interval = request.form.get("interval") or "1d"
            result = run_strategy(ticker1, ticker2, period, interval)
            return render_template(
                "results.html",
                ticker1=ticker1,
                ticker2=ticker2,
                period=period,
                interval=interval,
                model_accuracy=result.get("model_accuracy", 0),
                sharpe=result["stats"].get("sharpe", 0),
                max_drawdown=result["stats"].get("max_drawdown", 0),
                total_return=result["stats"].get("total_return", 0),
                price_img=result.get("price_img", ""),
                equity_img=result.get("equity_img", ""),
                spread_img=result.get("spread_img", ""),
                beta=result.get("beta", 0),
            )
        return render_template("index.html")
    except Exception as e:
        return f"App crashed: {str(e)}", 500

if __name__ == "__main__":
    app.run()