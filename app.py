from flask import Flask, render_template, request, redirect, session, make_response
from flask_bcrypt import Bcrypt
from functools import wraps
from db import users_col, strategies_col
from strategy import run_backtest, run_strategy
import os
import io
import csv

app = Flask(__name__)
bcrypt = Bcrypt(app)

app.secret_key = os.getenv("SECRET_KEY", "dev-secret")


# ---------- AUTH GUARD ----------
def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "user" not in session:
            return redirect("/login")
        return f(*args, **kwargs)
    return wrapper


# ---------- HOME = BATCH BACKTEST ----------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "user" not in session:
            return redirect("/login")

        pairs_raw = request.form.get("pairs", "").strip()
        if not pairs_raw:
            return render_template("index.html", results=[], error="Enter pairs.")

        pairs = [p for p in pairs_raw.splitlines() if "," in p]

        strategy_type = request.form.get("strategy_type", "ml")
        z_entry = float(request.form.get("z_entry") or 1.0)
        prob_threshold = float(request.form.get("prob_threshold") or 0.6)
        lookback = int(request.form.get("lookback") or 20)
        horizon = int(request.form.get("horizon") or 5)

        results = []
        for line in pairs:
            t1, t2 = [x.strip() for x in line.split(",")]
            try:
                out = run_backtest(
                    t1, t2,
                    strategy_type=strategy_type,
                    z_entry=z_entry,
                    prob_threshold=prob_threshold,
                    lookback=lookback,
                    horizon=horizon
                )
                results.append(out)
            except Exception as e:
                results.append({
                    "ticker1": t1,
                    "ticker2": t2,
                    "sharpe": None,
                    "total_return": None,
                    "max_drawdown": None,
                    "model_accuracy": None,
                    "n_trades": None,
                    "error": str(e)
                })

        valid = [r for r in results if r.get("total_return") is not None]

        portfolio_return = sum(r["total_return"] for r in valid) if valid else 0.0
        avg_sharpe = (sum(r["sharpe"] for r in valid) / len(valid)) if valid else 0.0

        session["last_results"] = results

        return render_template(
            "index.html",
            results=results,
            portfolio={
                "portfolio_return": round(portfolio_return, 3),
                "avg_sharpe": round(avg_sharpe, 3)
            }
        )

    return render_template("index.html", results=None, portfolio=None)



# ---------- RUN STRATEGY: CHARTS & STATS ----------
@app.route("/run", methods=["GET", "POST"])
@login_required
def run_single():
    if request.method == "POST":
        t1 = request.form.get("ticker1") or "AAPL"
        t2 = request.form.get("ticker2") or "MSFT"
        period = request.form.get("period") or "6mo"
        interval = request.form.get("interval") or "1d"

        try:
            r = run_strategy(t1, t2, period, interval)

            return render_template(
                "results.html",
                result=True,
                ticker1=t1,
                ticker2=t2,
                sharpe=round(r["stats"]["sharpe"], 3),
                total_return=round(r["stats"]["total_return"], 3),
                max_drawdown=round(r["stats"]["max_drawdown"], 3),
                model_accuracy=round(r["model_accuracy"], 3),
                price_img=r["price_img"],
                equity_img=r["equity_img"],
                spread_img=r["spread_img"],
                beta=round(float(r["beta"]), 3),
            )

        except Exception as e:
            return render_template(
                "results.html",
                result=None,
                error=f"Strategy failed: {str(e)}"
            )

    # GET request → show input UI
    return render_template("run.html")



# ---------- LIVE SIGNALS (placeholder) ----------
@app.route("/live", methods=["GET", "POST"])
@login_required
def live():
    if request.method == "POST":
        t1 = request.form.get("ticker1", "AAPL")
        t2 = request.form.get("ticker2", "MSFT")

        try:
            df = fetch_data(t1, t2)
            beta = compute_hedge_ratio(df)

            # only last values
            p1 = df["p1"].iloc[-1]
            p2 = df["p2"].iloc[-1]

            spread = p1 - beta * p2

            # simple z-score calc
            spread_series = df["p1"] - beta * df["p2"]
            z = (spread - spread_series.mean()) / spread_series.std()

            # trading rule
            if z > 1:
                sig = "SELL"
            elif z < -1:
                sig = "BUY"
            else:
                sig = "WAIT"

            return render_template(
                "live.html",
                ticker1=t1,
                ticker2=t2,
                signal=sig,
                zscore=round(z, 3),
                spread=round(spread, 3),
            )

        except Exception as e:
            return render_template(
                "live.html",
                error=str(e)
            )

    return render_template("live.html")



# ---------- SIGNUP ----------
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        email = request.form["email"].strip().lower()
        password = request.form["password"]

        if users_col.find_one({"email": email}):
            return render_template("signup.html", error="Email exists")

        hashed = bcrypt.generate_password_hash(password).decode("utf-8")
        users_col.insert_one({"email": email, "password": hashed})

        return redirect("/login")

    return render_template("signup.html")



# ---------- LOGIN ----------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"].strip().lower()
        password = request.form["password"]

        user = users_col.find_one({"email": email})
        if not user:
            return render_template("login.html", error="User not found")

        if not bcrypt.check_password_hash(user["password"], password):
            return render_template("login.html", error="Incorrect password")

        session["user"] = email
        return redirect("/")

    return render_template("login.html")



# ---------- LOGOUT ----------
@app.route("/logout")
@login_required
def logout():
    session.clear()
    return redirect("/login")



# ---------- SAVE STRATEGY CONFIG ----------
@app.route("/save_strategy", methods=["POST"])
@login_required
def save_strategy():
    strategies_col.insert_one({
        "user": session["user"],
        "name": request.form.get("strategy_name"),
        "strategy_type": request.form.get("strategy_type"),
        "z_entry": request.form.get("z_entry"),
        "prob_threshold": request.form.get("prob_threshold"),
        "lookback": request.form.get("lookback"),
        "horizon": request.form.get("horizon")
    })
    return redirect("/strategies")



# ---------- VIEW SAVED STRATEGIES ----------
@app.route("/strategies")
@login_required
def strategies():
    rows = list(strategies_col.find({"user": session["user"]}))
    return render_template("saved.html", strategies=rows)



# ---------- EXPORT CSV ----------
@app.route("/export_csv")
@login_required
def export_csv():
    results = session.get("last_results", [])
    if not results:
        return "No results", 400

    si = io.StringIO()
    writer = csv.writer(si)
    writer.writerow(["Ticker1", "Ticker2", "Sharpe", "TotalReturn",
                     "MaxDrawdown", "ModelAccuracy", "Trades", "Error"])
    for r in results:
        writer.writerow([
            r.get("ticker1"),
            r.get("ticker2"),
            r.get("sharpe"),
            r.get("total_return"),
            r.get("max_drawdown"),
            r.get("model_accuracy"),
            r.get("n_trades"),
            r.get("error")
        ])

    output = make_response(si.getvalue())
    output.headers["Content-Disposition"] = "attachment; filename=results.csv"
    output.headers["Content-Type"] = "text/csv"
    return output



# ---------- HEALTH ----------
@app.route("/health")
def health():
    return "ok", 200



if __name__ == "__main__":
    app.run(debug=True)