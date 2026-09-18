"""Central configuration for the stock prediction pipeline."""
import os
from dotenv import load_dotenv

load_dotenv()

TICKERS = [t.strip().upper() for t in
           os.getenv("STOCK_TICKERS", "AAPL,MSFT,TSLA,NVDA").split(",") if t.strip()]
BENCHMARK = os.getenv("STOCK_BENCHMARK", "SPY")
DB_PATH = os.getenv("STOCK_DB_PATH", "market.db")
PLOT_DIR = os.getenv("STOCK_PLOT_DIR", "plots")
EXPORT_DIR = os.getenv("STOCK_EXPORT_DIR", "exports")

DAYS = int(os.getenv("STOCK_DAYS", "1500"))          # history to download
EPOCHS = int(os.getenv("STOCK_EPOCHS", "25"))
WINDOW = int(os.getenv("STOCK_WINDOW", "60"))        # look-back days per sample
HORIZON = int(os.getenv("STOCK_HORIZON", "1"))       # days ahead to predict
TEST_SPLIT = float(os.getenv("STOCK_TEST_SPLIT", "0.2"))
TRANSACTION_COST = float(os.getenv("STOCK_TXN_COST", "0.0005"))  # 5 bps per trade
SEED = int(os.getenv("STOCK_SEED", "42"))

# Where the model runs. On machines without tensorflow we fall back to a
# gradient-boosting model so the project never hard-fails.
BACKEND = os.getenv("STOCK_BACKEND", "auto").lower()  # auto | lstm | gbdt

API_HOST = os.getenv("STOCK_API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("STOCK_API_PORT", "8002"))
