from pymongo import MongoClient

MONGO_URI = "mongodb+srv://laksharajjha:9863173888@aipairtrading01.1qymhum.mongodb.net/?appName=AIPAIRTRADING01"

client = MongoClient(MONGO_URI)
db = client["pairs_trading"]

users_col = db["users"]
strategies_col = db["strategies"]