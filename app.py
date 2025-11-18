from flask import Flask, render_template, request, redirect, session
from flask_bcrypt import Bcrypt
from db import users_col
from strategy import run_strategy
import os

app = Flask(__name__)
bcrypt = Bcrypt(app)

app.secret_key = os.environ.get("SECRET_KEY", "dev-secret")


# ---------- AUTH GUARD ----------
def login_required(route):
    def wrapper(*args, **kwargs):
        if "user" not in session:
            return redirect("/login")
        return route(*args, **kwargs)
    wrapper.__name__ = route.__name__
    return wrapper


# ---------- HOME + PAIR TRADING ----------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        ticker1 = request.form.get("ticker1")
        ticker2 = request.form.get("ticker2")
        period = request.form.get("period") or "6mo"
        interval = request.form.get("interval") or "1d"

        result = run_strategy(ticker1, ticker2, period, interval)

        return render_template("results.html", **result, ticker1=ticker1, ticker2=ticker2)

    return render_template("index.html")


# ---------- SIGNUP ----------
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        if users_col.find_one({"email": email}):
            return render_template("signup.html", error="User already exists")

        hashed_pw = bcrypt.generate_password_hash(password).decode()
        users_col.insert_one({"email": email, "password": hashed_pw})

        return redirect("/login")

    return render_template("signup.html")


# ---------- LOGIN ----------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        user = users_col.find_one({"email": email})

        if not user:
            return render_template("login.html", error="No such user")

        if not bcrypt.check_password_hash(user["password"], password):
            return render_template("login.html", error="Wrong password")

        session["user"] = email
        return redirect("/dashboard")

    return render_template("login.html")


# ---------- LOGOUT ----------
@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")


# ---------- DASHBOARD ----------
@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html", user=session["user"])


# ---------- ENTRY POINT ----------
if __name__ == "__main__":
    app.run()