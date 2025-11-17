import os
from pymongo import MongoClient

MONGO_URI = os.environ.get("MONGO_URI")

client = MongoClient(MONGO_URI)
db = client["pairs_trading"]

# Collections
users_col = db["users"]
strategies_col = db["strategies"]