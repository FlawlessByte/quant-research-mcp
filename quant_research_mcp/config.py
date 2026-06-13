"""Central configuration: universe, screener, regime and decision thresholds."""

# Liquid, high-beta US names suited to intraday research.
DEFAULT_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AMD",
    "AVGO", "NFLX", "CRM", "ORCL", "INTC", "MU", "QCOM", "SMCI",
    "PLTR", "COIN", "MARA", "RIOT", "XYZ", "SHOP", "UBER", "ABNB",
    "BA", "JPM", "GS", "XOM", "CVX", "WMT", "COST", "LLY",
    "UNH", "CAT", "DE", "GE", "F", "GM", "DAL", "AAL",
]

# --- Screener -------------------------------------------------------------
MIN_PRICE = 5.0
MIN_DOLLAR_VOLUME = 50_000_000          # 20-day avg daily dollar volume
TOP_N = 5

# --- Data / strategy ------------------------------------------------------
DAILY_LOOKBACK = "9mo"
INTRADAY_INTERVAL = "5m"
INTRADAY_PERIOD = "5d"
OPENING_RANGE_BARS = 3                   # 3 x 5m = 15-minute opening range

HURST_MAX_LAG = 20
HURST_WINDOW = 120
HURST_TREND = 0.55                       # H >= -> trending (momentum playbook)
HURST_REVERT = 0.45                      # H <= -> mean-reverting (fade playbook)

MIN_REL_VOLUME = 1.2
RISK_REWARD = 2.0
VWAP_FADE_SIGMA = 2.0
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

# --- Method parameters ------------------------------------------------------
DONCHIAN_ENTRY = 55              # breakout lookback (days)
DONCHIAN_EXIT = 20               # exit-channel lookback (days)
DONCHIAN_ATR_TRAIL = 2.5         # ATR-multiple trailing stop

XS_MOM_LOOKBACK = 252            # 12-month window
XS_MOM_SKIP = 21                 # skip most recent month (reversal noise)
XS_MOM_TOP_N = 5                 # names held long

RSI2_ENTRY = 10.0                # RSI(2) below -> long (Connors)
RSI2_ENTRY_SHORT = 90.0          # RSI(2) above -> short (below 200d SMA)
RSI2_EXIT_MA = 5                 # exit when close crosses 5d SMA
RSI2_TREND_MA = 200              # only long above 200d SMA

PAIRS_LOOKBACK = 252             # cointegration estimation window
PAIRS_ENTRY_Z = 2.0
PAIRS_EXIT_Z = 0.0
PAIRS_STOP_Z = 3.5
PAIRS_ADF_PVALUE = 0.05          # spread must be stationary at 5%

# --- Backtesting ------------------------------------------------------------
BACKTEST_COSTS_BPS = 5.0         # one-way slippage+fees, basis points
BACKTEST_EQUITY = 100_000.0

# --- Events / portfolio risk -------------------------------------------------
EARNINGS_VETO_DAYS = 3           # veto swing/position entries this close to earnings
MAX_PORTFOLIO_HEAT = 0.02        # total open risk <= 2% of equity
MAX_POSITIONS = 8
CORRELATION_FLAG = 0.7           # pairwise 90d return correlation worth flagging
CORRELATION_WINDOW = 90          # days of returns for the correlation matrix

# --- Decision helper ------------------------------------------------------
# Composite-score weights (must sum to 1.0). Each sub-factor is scaled to
# 0..1 and combined; threshold gates ENTRY vs NO_ENTRY.
DECISION_WEIGHTS = {
    "reward_risk": 0.25,
    "regime_strength": 0.20,
    "volume_confirmation": 0.20,
    "momentum_position": 0.15,
    "stop_quality": 0.20,
}
DECISION_THRESHOLD = 0.55                # score >= -> ENTRY (absent any veto)
NOISE_STOP_ATR_FRACTION = 0.30           # stop < this x daily ATR -> size haircut
NOISE_STOP_SIZE_HAIRCUT = 0.60           # multiply size by this when stop is noise-tight
DEFAULT_ACCOUNT_RISK_PCT = 0.005         # 0.5% of equity risked per trade
