from flask import Flask, render_template, request, jsonify, Response, redirect, url_for, session
from strategy import run_strategy, run_backtest, fetch_data, compute_hedge_ratio, build_features, train_model
from flask_bcrypt import Bcrypt
from functools import wraps
from bson import ObjectId
from db import users_col, strategies_col
import time, csv, io

app = Flask(__name__)
app.secret_key = "super_secret_quant_key"
bcrypt = Bcrypt(app)

last_backtest_csv = None

# AUTH DECORATOR
def login_required(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if "user_id" in session:
            return f(*args, **kwargs)
        return redirect("/login")
    return wrap


@app.route("/", methods=["GET", "POST"])
@login_required
def index():
    if request.method == "POST":
        ticker1 = request.form.get("ticker1") or "AAPL"
        ticker2 = request.form.get("ticker2") or "MSFT"
        period = request.form.get("period") or "6mo"
        interval = request.form.get("interval") or "1d"
        try:
            result = run_strategy(ticker1, ticker2, period, interval)
            return render_template(
                "results.html",
                ticker1=ticker1,
                ticker2=ticker2,
                period=period,
                interval=interval,
                model_accuracy=round(result["model_accuracy"], 3),
                sharpe=round(result["stats"]["sharpe"], 3),
                max_drawdown=round(result["stats"]["max_drawdown"], 3),
                total_return=round(float(result["stats"]["total_return"]), 3),
                price_img=result["price_img"],
                equity_img=result["equity_img"],
                spread_img=result["spread_img"],
                beta=round(float(result["beta"]), 3),
            )
        except Exception as e:
            return render_template("index.html", error=str(e))

    return render_template("index.html")


# ---------- AUTH ROUTES ----------
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        email = request.form.get("email").lower()
        password = request.form.get("password")

        if users_col.find_one({"email": email}):
            return render_template("signup.html", error="User already exists")

        hash_pw = bcrypt.generate_password_hash(password).decode()
        users_col.insert_one({"email": email, "password": hash_pw})

        return redirect("/login")

    return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email").lower()
        password = request.form.get("password")

        user = users_col.find_one({"email": email})
        if not user:
            return render_template("login.html", error="User not found")

        if not bcrypt.check_password_hash(user["password"], password):
            return render_template("login.html", error="Wrong password")

        session["user_id"] = str(user["_id"])
        session["email"] = email

        return redirect("/")

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")


# ---------- BACKTEST ----------
@app.route("/backtest", methods=["GET", "POST"])
@login_required
def backtest_view():
    global last_backtest_csv
    results = []
    portfolio = None

    if request.method == "POST":
        strategy_type = request.form.get("strategy_type") or "ml"
        z_entry = float(request.form.get("z_entry") or 1.0)
        prob_threshold = float(request.form.get("prob_threshold") or 0.6)
        lookback = int(request.form.get("lookback") or 20)
        horizon = int(request.form.get("horizon") or 5)

        pairs_text = request.form.get("pairs") or ""
        lines = [l.strip() for l in pairs_text.splitlines() if l.strip()]

        for line in lines:
            parts = [p.strip() for p in line.replace(";", ",").split(",") if p.strip()]
            if len(parts) != 2:
                continue

            t1, t2 = parts
            try:
                stats = run_backtest(
                    t1, t2,
                    strategy_type=strategy_type,
                    z_entry=z_entry,
                    prob_threshold=prob_threshold,
                    lookback=lookback,
                    horizon=horizon
                )
                results.append(stats)

            except Exception as e:
                results.append({
                    "ticker1": t1,
                    "ticker2": t2,
                    "sharpe": None,
                    "total_return": None,
                    "max_drawdown": None,
                    "model_accuracy": None,
                    "n_trades": 0,
                    "error": str(e)
                })

        if results:
            valid = [r for r in results if r.get("sharpe") is not None]
            if valid:
                portfolio_return = sum(r["total_return"] for r in valid)
                avg_sharpe = sum(r["sharpe"] for r in valid) / len(valid)
                portfolio = {"portfolio_return": portfolio_return, "avg_sharpe": avg_sharpe}

            output = io.StringIO()
            fieldnames = ["ticker1", "ticker2", "sharpe", "total_return", "max_drawdown", "model_accuracy", "n_trades"]
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()

            for r in results:
                writer.writerow({
                    "ticker1": r.get("ticker1", ""),
                    "ticker2": r.get("ticker2", ""),
                    "sharpe": r.get("sharpe", ""),
                    "total_return": r.get("total_return", ""),
                    "max_drawdown": r.get("max_drawdown", ""),
                    "model_accuracy": r.get("model_accuracy", ""),
                    "n_trades": r.get("n_trades", "")
                })

            last_backtest_csv = output.getvalue()

    return render_template("backtest.html", results=results, portfolio=portfolio)


@app.route("/export_csv")
@login_required
def export_csv():
    global last_backtest_csv
    if not last_backtest_csv:
        return "No backtest data to export", 400

    return Response(
        last_backtest_csv,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=backtest_results.csv"}
    )


@app.route("/strategies")
@login_required
def list_strategies():
    user_id = ObjectId(session["user_id"])
    strategies = list(strategies_col.find({"user_id": user_id}))
    return render_template("strategies.html", strategies=strategies)


@app.route("/save_strategy", methods=["POST"])
@login_required
def save_strategy():
    user_id = ObjectId(session["user_id"])

    strategy_name = request.form.get("strategy_name") or "Unnamed"
    strategy_type = request.form.get("strategy_type") or "ml"
    z_entry = request.form.get("z_entry") or "1.0"
    prob_threshold = request.form.get("prob_threshold") or "0.6"
    lookback = request.form.get("lookback") or "20"
    horizon = request.form.get("horizon") or "5"

    cfg = {
        "user_id": user_id,
        "name": strategy_name,
        "strategy_type": strategy_type,
        "z_entry": z_entry,
        "prob_threshold": prob_threshold,
        "lookback": lookback,
        "horizon": horizon
    }

    strategies_col.insert_one(cfg)
    return redirect("/strategies")


@app.route("/live")
@login_required
def live():
    ticker1 = request.args.get("t1")
    ticker2 = request.args.get("t2")
    return render_template("live.html", ticker1=ticker1, ticker2=ticker2)


@app.route("/live_data")
@login_required
def live_data():
    try:
        t1 = request.args.get("t1")
        t2 = request.args.get("t2")
        df = fetch_data(t1, t2)
        beta = compute_hedge_ratio(df)

        spread = df["p1"] - beta * df["p2"]
        z = (spread - spread.rolling(20).mean()) / spread.rolling(20).std()
        last_z = float(z.iloc[-1]) if not z.iloc[-1] != z.iloc[-1] else 0.0

        features = build_features(df, beta)
        model, acc, X = train_model(features)
        prob = float(model.predict_proba(X)[-1][1])

        signal = "HOLD"
        if abs(last_z) > 1 and prob > 0.6:
            signal = "SHORT" if last_z > 0 else "LONG"

        return jsonify({
            "zscore": round(last_z, 3),
            "prob": round(prob, 3),
            "signal": signal,
            "price1": float(df["p1"].iloc[-1]),
            "price2": float(df["p2"].iloc[-1]),
            "ts": time.time()
        })

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True)