# Btc-sma-dashboard

# BTC/USD SMA SIGNAL MONITOR
# Install: pip install ccxt pandas

import time
import logging
from pathlib import Path

import ccxt
import pandas as pd

# ===== SETTINGS =====
SYMBOL = "BTC/USD"
TIMEFRAME = "1h"

SHORT_MA = 10
LONG_MA = 50

STOP_LOSS_PERCENT = 2.0
TAKE_PROFIT_PERCENT = 4.0

# Hypothetical monitoring settings
POLL_SECONDS = 60
COOLDOWN_CANDLES = 2
MAX_DATA_CANDLES = 200

# Local files store simulated state and alerts
STATE_FILE = Path("btc_signal_state.csv")
LOG_FILE = "btc_signal_monitor.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)

# Public market data only; no API keys and no orders.
exchange = ccxt.binance({
    "enableRateLimit": True,
})

in_position = False
entry_price = None
entry_candle = None
last_processed_candle = None
last_exit_candle = None


def get_price_data():
    """Fetch candles and calculate indicators using closed candles."""
    candles = exchange.fetch_ohlcv(
        SYMBOL, TIMEFRAME, limit=MAX_DATA_CANDLES
    )

    if not candles or len(candles) < LONG_MA + 3:
        raise ValueError("Not enough candle data.")

    df = pd.DataFrame(
        candles,
        columns=["time", "open", "high", "low", "close", "volume"],
    )

    # The newest OHLCV candle may still be forming.
    df = df.iloc[:-1].copy()

    if df.empty:
        raise ValueError("No completed candles available.")

    df["short_ma"] = df["close"].rolling(SHORT_MA).mean()
    df["long_ma"] = df["close"].rolling(LONG_MA).mean()

    # Average true range helps describe recent volatility.
    previous_close = df["close"].shift(1)
    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    df["atr"] = true_range.rolling(14).mean()

    df = df.dropna().reset_index(drop=True)

    if len(df) < 2:
        raise ValueError("Not enough completed indicator data.")

    return df


def save_state():
    """Persist simulated position information across restarts."""
    pd.DataFrame([{
        "in_position": in_position,
        "entry_price": entry_price,
        "entry_candle": entry_candle,
        "last_processed_candle": last_processed_candle,
        "last_exit_candle": last_exit_candle,
    }]).to_csv(STATE_FILE, index=False)


def load_state():
    """Restore the previous simulated state, if available."""
    global in_position, entry_price, entry_candle
    global last_processed_candle, last_exit_candle

    if not STATE_FILE.exists():
        return

    try:
        state = pd.read_csv(STATE_FILE).iloc[0]

        in_position = bool(state["in_position"])

        entry_price = (
            float(state["entry_price"])
            if pd.notna(state["entry_price"])
            else None
        )

        entry_candle = (
            int(state["entry_candle"])
            if pd.notna(state["entry_candle"])
            else None
        )

        last_processed_candle = (
            int(state["last_processed_candle"])
            if pd.notna(state["last_processed_candle"])
            else None
        )

        last_exit_candle = (
            int(state["last_exit_candle"])
            if pd.notna(state["last_exit_candle"])
            else None
        )

    except (ValueError, IndexError, KeyError, OSError) as exc:
        logging.error("Could not restore state: %s", exc)
        raise RuntimeError(
            "State file is invalid. Inspect it before restarting."
        ) from exc


def process_latest_candle(df):
    global in_position, entry_price, entry_candle
    global last_processed_candle, last_exit_candle

    previous = df.iloc[-2]
    current = df.iloc[-1]

    candle_time = int(current["time"])
    price = float(current["close"])
    short_ma = float(current["short_ma"])
    long_ma = float(current["long_ma"])
    atr = float(current["atr"])

    if candle_time == last_processed_candle:
        return

    # Avoid repeatedly processing the same closed candle.
    last_processed_candle = candle_time

    cross_up = (
        previous["short_ma"] <= previous["long_ma"]
        and short_ma > long_ma
    )

    cross_down = (
        previous["short_ma"] >= previous["long_ma"]
        and short_ma < long_ma
    )

    timestamp = exchange.iso8601(candle_time)

    logging.info(
        "%s | close=%.2f | SMA%d=%.2f | SMA%d=%.2f | ATR14=%.2f",
        timestamp, price, SHORT_MA, short_ma, LONG_MA, long_ma, atr,
    )

    # Existing simulated position: check exits first.
    if in_position and entry_price is not None:
        stop_price = entry_price * (
            1 - STOP_LOSS_PERCENT / 100
        )
        target_price = entry_price * (
            1 + TAKE_PROFIT_PERCENT / 100
        )

        # Candle lows/highs can reveal a threshold was touched,
        # but OHLCV cannot establish the exact intrabar sequence.
        stop_hit = float(current["low"]) <= stop_price
        target_hit = float(current["high"]) >= target_price

        if stop_hit and target_hit:
            logging.warning(
                "EXIT REVIEW: both thresholds touched in one candle. "
                "Intrabar order is unknown; outcome is ambiguous."
            )
            in_position = False
            entry_price = None
            entry_candle = None
            last_exit_candle = candle_time

        elif stop_hit:
            logging.warning(
                "HYPOTHETICAL STOP ALERT: threshold %.2f",
                stop_price,
            )
            in_position = False
            entry_price = None
            entry_candle = None
            last_exit_candle = candle_time

        elif target_hit:
            logging.info(
                "HYPOTHETICAL TARGET ALERT: threshold %.2f",
                target_price,
            )
            in_position = False
            entry_price = None
            entry_candle = None
            last_exit_candle = candle_time

        elif cross_down:
            logging.info("SELL SIGNAL: bearish SMA crossover")
            in_position = False
            entry_price = None
            entry_candle = None
            last_exit_candle = candle_time

    # Only consider a new entry if flat after exit processing.
    if not in_position:
        cooldown_complete = (
            last_exit_candle is None
            or candle_time - last_exit_candle
            >= COOLDOWN_CANDLES * 60 * 60 * 1000
        )

        if cross_up and cooldown_complete:
            logging.info(
                "BUY SIGNAL: bullish SMA crossover. "
                "This is a hypothetical signal only."
            )
            in_position = True
            entry_price = price
            entry_candle = candle_time

        elif not cross_up:
            logging.info("HOLD / NO NEW ENTRY SIGNAL")

    save_state()


def main():
    logging.info("Starting signals-only BTC/USD monitor.")
    logging.info("No orders will be submitted.")

    load_state()

    while True:
        try:
            df = get_price_data()
            process_latest_candle(df)
        except (ccxt.NetworkError, ccxt.ExchangeError) as exc:
            logging.warning("Exchange data error: %s", exc)
        except Exception:
            logging.exception("Unexpected monitoring error")

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
