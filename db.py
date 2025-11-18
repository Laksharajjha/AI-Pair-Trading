import os
from pymongo import MongoClient

MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)

db = client["pairs_trading_ai"]

users_col = db["users"]
strategies_col = db["strategies"]