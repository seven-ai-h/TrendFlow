"""
Central configuration — the single source of truth for TrendFlow's data universe.

Nothing about *which* assets we track or *how* headlines map to them is hardcoded
inside the analysis or collection code. It all lives here, and can be overridden
without touching any Python:

  * drop a `tickers.json` in the project root, OR
  * set TRENDFLOW_TICKERS=/path/to/your.json

Either file is a JSON object of:
    { "NVDA": {"name": "NVIDIA", "keywords": ["nvidia", "gpu", ...]}, ... }
"""
import os
import json

# ── Default universe (used when no override file is present) ──────────────────
_DEFAULT_TICKERS = {
    "NVDA":   {"name": "NVIDIA",    "keywords": ["nvidia", "nvda", "gpu", "cuda", "jensen huang", "geforce", "h100", "blackwell"]},
    "MSFT":   {"name": "Microsoft", "keywords": ["microsoft", "msft", "azure", "copilot", "openai", "gpt", "chatgpt", "windows"]},
    "GOOGL":  {"name": "Alphabet",  "keywords": ["google", "googl", "alphabet", "gemini", "deepmind", "android", "chrome", "waymo"]},
    "AAPL":   {"name": "Apple",     "keywords": ["apple", "aapl", "iphone", "ios", "macos", "ipad", "vision pro", "mac"]},
    "META":   {"name": "Meta",      "keywords": ["meta", "facebook", "instagram", "llama", "metaverse", "whatsapp", "zuckerberg"]},
    "AMZN":   {"name": "Amazon",    "keywords": ["amazon", "amzn", "aws", "bezos", "prime", "alexa"]},
    "TSLA":   {"name": "Tesla",     "keywords": ["tesla", "tsla", "musk", "electric vehicle", "autopilot", "cybertruck", "ev"]},
    "AMD":    {"name": "AMD",       "keywords": ["amd", "ryzen", "radeon", "epyc", "lisa su"]},
    "BTC-USD":{"name": "Bitcoin",   "keywords": ["bitcoin", "btc", "crypto", "satoshi", "halving"]},
    "ETH-USD":{"name": "Ethereum",  "keywords": ["ethereum", "eth", "ether", "vitalik", "smart contract"]},
}

# Optional seed hint: a plausible starting price for the synthetic demo only.
# Real runs fetch actual prices via yfinance and never touch this.
SEED_START_PRICE = {
    "NVDA": 120, "MSFT": 420, "GOOGL": 175, "AAPL": 220, "META": 560,
    "AMZN": 185, "TSLA": 250, "AMD": 160, "BTC-USD": 65000, "ETH-USD": 3400,
}
DEFAULT_SEED_PRICE = 100.0


def _load_tickers() -> dict:
    """Load the ticker universe from an override file if present, else defaults."""
    path = os.getenv("TRENDFLOW_TICKERS")
    candidates = [p for p in (path, "tickers.json") if p]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            try:
                with open(candidate) as fh:
                    data = json.load(fh)
                if isinstance(data, dict) and data:
                    # basic validation
                    cleaned = {}
                    for tk, meta in data.items():
                        cleaned[tk] = {
                            "name": meta.get("name", tk),
                            "keywords": [k.lower() for k in meta.get("keywords", [])],
                        }
                    print(f"[config] Loaded {len(cleaned)} tickers from {candidate}")
                    return cleaned
            except Exception as e:
                print(f"[config] Could not read {candidate}: {e} — using defaults")
    return _DEFAULT_TICKERS


TICKERS = _load_tickers()

# ── Derived views the rest of the app imports ────────────────────────────────
TRACKED_TICKERS = list(TICKERS.keys())
TICKER_NAMES = {tk: meta["name"] for tk, meta in TICKERS.items()}
TICKER_KEYWORDS = {tk: meta["keywords"] for tk, meta in TICKERS.items()}


def seed_start_price(ticker: str) -> float:
    return float(SEED_START_PRICE.get(ticker, DEFAULT_SEED_PRICE))
