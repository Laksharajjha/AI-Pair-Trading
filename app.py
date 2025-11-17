from flask import Flask, render_template, request, redirect, session
from strategy import run_strategy, run_backtest, fetch_data, compute_hedge_ratio, build_features, train_model
from db import users_col, strategies_col
import os

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "default_secret")


@app.route("/")
def home():
    if "user" not in session:
        return redirect("/login")
    return render_template("index.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        pwd = request.form.get("password")

        user = users_col.find_one({"email": email})
        if user and user["password"] == pwd:
            session["user"] = email
            return redirect("/")

        return render_template("login.html", error="Invalid credentials")

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect("/login")


@app.route("/run", methods=["POST"])
def run():
    if "user" not in session:
        return redirect("/login")

    ticker1 = request.form.get("ticker1")
    ticker2 = request.form.get("ticker2")
    period = request.form.get("period") or "6mo"
    interval = request.form.get("interval") or "1d"

    result = run_strategy(ticker1, ticker2, period, interval)

    return render_template(
        "results.html",
        ticker1=ticker1,
        ticker2=ticker2,
        model_accuracy=result["model_accuracy"],
        sharpe=result["stats"]["sharpe"],
        total_return=result["stats"]["total_return"],
        max_drawdown=result["stats"]["max_drawdown"],
        price_img=result["price_img"],
        spread_img=result["spread_img"],
        equity_img=result["equity_img"],
        beta=result["beta"]
    )


@app.route("/health")
def health():
    return {"status": "ok"}, 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)