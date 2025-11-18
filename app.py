from flask import Flask, render_template, request, redirect, session, make_response
from flask_bcrypt import Bcrypt
from functools import wraps
from db import users_col, strategies_col
from strategy import run_backtest, run_strategy
import os, io, csv

app = Flask(__name__)
bcrypt = Bcrypt(app)

app.secret_key = os.getenv("SECRET_KEY", "dev-secret")


# -------- AUTH GUARD --------
def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "user" not in session:
            return redirect("/login")
        return f(*args, **kwargs)
    return wrapper


# -------- HOME + BATCH BACKTEST --------
@app.route("/", methods=["GET", "POST"])
@login_required
def index():
    if request.method == "POST":
        pairs_raw = request.form.get("pairs", "").strip()

        if not pairs_raw:
            return render_template("index.html", results=[], error="No pairs provided")

        pairs = [p.split(",") for p in pairs_raw.splitlines() if "," in p]

        results = []
        strategy_type = request.form.get("strategy_type", "ml")
        z_entry = float(request.form.get("z_entry") or 1.0)
        prob_threshold = float(request.form.get("prob_threshold") or 0.6)
        lookback = int(request.form.get("lookback") or 20)
        horizon = int(request.form.get("horizon") or 5)

        for t1, t2 in pairs:
            t1, t2 = t1.strip(), t2.strip()
            try:
                out = run_backtest(
                    t1,
                    t2,
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
                    "error": str(e)
                })

        # portfolio stats
        valid = [r for r in results if "total_return" in r]
        portfolio_return = sum(r["total_return"] for r in valid)
        avg_sharpe = (
            sum(r["sharpe"] for r in valid) / len(valid)
            if len(valid) > 0 else 0
        )

        # store for CSV export
        session["last_results"] = results

        return render_template(
            "index.html",
            results=results,
            portfolio={"portfolio_return": portfolio_return, "avg_sharpe": avg_sharpe}
        )

    return render_template("index.html")


# -------- SIGNUP --------
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        if users_col.find_one({"email": email}):
            return render_template("signup.html", error="Email already registered")

        hashed = bcrypt.generate_password_hash(password).decode("utf-8")
        users_col.insert_one({"email": email, "password": hashed})

        return redirect("/login")

    return render_template("signup.html")


# -------- LOGIN --------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        user = users_col.find_one({"email": email})
        if not user:
            return render_template("login.html", error="User not found")

        if not bcrypt.check_password_hash(user["password"], password):
            return render_template("login.html", error="Incorrect password")

        session["user"] = email
        return redirect("/")

    return render_template("login.html")


# -------- LOGOUT --------
@app.route("/logout")
@login_required
def logout():
    session.clear()
    return redirect("/login")


# -------- SAVE STRATEGY --------
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
        "horizon": request.form.get("horizon"),
    })

    return redirect("/strategies")


# -------- VIEW SAVED STRATEGIES --------
@app.route("/strategies")
@login_required
def strategies():
    rows = list(strategies_col.find({"user": session["user"]}))
    return render_template("saved.html", strategies=rows)


# -------- EXPORT CSV --------
@app.route("/export_csv")
@login_required
def export_csv():
    results = session.get("last_results", [])
    if not results:
        return "No results", 400

    si = io.StringIO()
    writer = csv.writer(si)
    writer.writerow(["Ticker1", "Ticker2", "Sharpe", "Return", "Drawdown", "Accuracy", "Trades"])

    for r in results:
        writer.writerow([
            r.get("ticker1"),
            r.get("ticker2"),
            r.get("sharpe"),
            r.get("total_return"),
            r.get("max_drawdown"),
            r.get("model_accuracy"),
            r.get("n_trades"),
        ])

    output = make_response(si.getvalue())
    output.headers["Content-Disposition"] = "attachment; filename=results.csv"
    output.headers["Content-type"] = "text/csv"
    return output


# -------- HEALTH CHECK --------
@app.route("/health")
def health():
    return "ok", 200


if __name__ == "__main__":
    app.run(debug=True)