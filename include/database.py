import datetime

try:
    import pymongo
    _PYMONGO_AVAILABLE = True
except ImportError:
    _PYMONGO_AVAILABLE = False

class DatabaseLayer:
    def __init__(self):
        self.client     = None
        self.collection = None
        self.online     = False
        self._init_connection()

    def _init_connection(self):
        if not _PYMONGO_AVAILABLE:
            print("[DB] pymongo not installed — running in Standalone/Offline Mode.")
            return
        try:
            self.client = pymongo.MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=2000)
            self.client.server_info()
            db = self.client["fake_news_db"]
            self.collection = db["predictions"]
            self.online = True
            print("[DB] MongoDB connected.")
        except Exception as exc:
            print(f"[DB] MongoDB unavailable ({exc}) — Running Offline Mode.")
            self.client = None
            self.online = False

    def insert_prediction(self, text: str, prediction: str, confidence: float):
        if not self.online: return
        try:
            self.collection.insert_one({
                "text":        text,
                "prediction":  prediction,
                "confidence":  round(confidence, 4),
                "timestamp":   datetime.datetime.utcnow().isoformat() + "Z",
            })
        except Exception as exc:
            print(f"[DB] Insert failed: {exc}")

    def purge_collection(self):
        if not self.online: return 0
        try:
            return self.collection.delete_many({}).deleted_count
        except Exception as exc:
            print(f"[DB] Purge failed: {exc}")
            return 0