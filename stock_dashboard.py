"""
Stock Analyst Dashboard - colourful browser app.
Type a company name or ticker and get charts, trend, fundamentals, risk and a
3-month to 1-year forecast range.

Setup:  pip install streamlit plotly yfinance pandas numpy
Run:    python -m streamlit run stock_dashboard.py --theme.base dark

Educational tool only. Not financial advice.
"""
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Stock Analyst", page_icon="📈", layout="wide")

GREEN, RED, AMBER, BLUE, PURPLE = "#26d07c", "#ff5470", "#ffb830", "#4da3ff", "#a78bfa"
HORIZONS = {"3 months": 63, "6 months": 126, "1 year": 252}
TRANSPARENT = "rgba(0,0,0,0)"

st.markdown(
    """
<style>
.block-container{padding-top:1.4rem;}
.hero{background:linear-gradient(120deg,#1e3a8a,#7c3aed 55%,#db2777);padding:22px 28px;
      border-radius:18px;margin-bottom:14px}
.hero h1{margin:0;font-size:1.9rem;color:#fff}
.hero p{margin:4px 0 0;color:#ece8ff}
.card{background:#171b26;border:1px solid #2a3042;border-radius:14px;padding:14px 16px;height:100%}
.card .l{font-size:.72rem;color:#8d96ad;text-transform:uppercase;letter-spacing:.06em}
.card .v{font-size:1.3rem;font-weight:700;color:#f2f4fa}
.card .s{font-size:.8rem}
.verdict{border-radius:14px;padding:16px 20px;font-size:1.05rem;color:#fff}
.chip{display:inline-block;padding:6px 12px;margin:4px 6px 4px 0;border-radius:20px;
      font-size:.85rem;color:#fff}
</style>
""",
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------- helpers --
def card(label, value, sub="", color="#8d96ad"):
    return (f'<div class="card"><div class="l">{label}</div><div class="v">{value}</div>'
            f'<div class="s" style="color:{color}">{sub}</div></div>')


def chip(v, text):
    bg = "#1c8c5a" if v > 0 else "#b4234a" if v < 0 else "#8a6d12"
    return f'<span class="chip" style="background:{bg}">{text}</span>'


def label(s):
    if s >= 0.5: return "Strongly Bullish"
    if s >= 0.2: return "Bullish"
    if s > -0.2: return "Neutral"
    if s > -0.5: return "Bearish"
    return "Strongly Bearish"


def score_color(s):
    return GREEN if s >= 0.2 else RED if s <= -0.2 else AMBER


def big(x):
    if not isinstance(x, (int, float)):
        return "n/a"
    for u, v in (("T", 1e12), ("B", 1e9), ("M", 1e6)):
        if abs(x) >= v:
            return f"{x / v:.2f}{u}"
    return f"{x:,.0f}"


def num(x, d=2):
    return "n/a" if not isinstance(x, (int, float)) or np.isnan(x) else f"{x:,.{d}f}"


def pct_change(c, days):
    return c.iloc[-1] / c.iloc[-days - 1] - 1 if len(c) > days else np.nan


# -------------------------------------------------------------------- data --
@st.cache_data(ttl=900, show_spinner=False)
def resolve_ticker(query):
    import yfinance as yf
    q = query.strip()
    if " " not in q and len(q) <= 8:
        try:
            if not yf.Ticker(q.upper()).history(period="5d").empty:
                return q.upper()
        except Exception:
            pass
    try:
        hits = yf.Search(q, max_results=8).quotes
        eq = [h for h in hits if h.get("quoteType") in ("EQUITY", "ETF")]
        if eq:
            return eq[0]["symbol"]
    except Exception:
        pass
    return q.upper()


@st.cache_data(ttl=900, show_spinner=False)
def fetch(ticker):
    import yfinance as yf
    t = yf.Ticker(ticker)
    df = t.history(period="5y", auto_adjust=True)
    try:
        info = t.info or {}
    except Exception:
        info = {}
    if not df.empty:
        df.index = df.index.tz_localize(None)
    return df, info


# --------------------------------------------------------------- indicators --
def add_indicators(df):
    df = df.copy()
    c, h, l = df["Close"], df["High"], df["Low"]
    df["SMA20"] = c.rolling(20).mean()
    df["SMA50"] = c.rolling(50).mean()
    df["SMA200"] = c.rolling(200).mean()
    d = c.diff()
    gain = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    df["RSI"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    df["MACD"] = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    df["MACDsig"] = df["MACD"].ewm(span=9, adjust=False).mean()
    sd = c.rolling(20).std()
    df["BBup"], df["BBlo"] = df["SMA20"] + 2 * sd, df["SMA20"] - 2 * sd
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()
    return df


def technical_score(df):
    r, c = df.iloc[-1], df["Close"]
    sig = [
        (1 if r.Close > r.SMA50 else -1, "Price above 50-day average" if r.Close > r.SMA50 else "Price below 50-day average"),
        (1 if r.Close > r.SMA200 else -1, "Above 200-day average" if r.Close > r.SMA200 else "Below 200-day average"),
        (1 if r.SMA50 > r.SMA200 else -1, "Golden-cross zone (50d > 200d)" if r.SMA50 > r.SMA200 else "Death-cross zone (50d < 200d)"),
        (1 if r.MACD > r.MACDsig else -1, "MACD bullish" if r.MACD > r.MACDsig else "MACD bearish"),
    ]
    if r.RSI > 70:
        sig.append((-0.5, f"RSI {r.RSI:.0f}: overbought"))
    elif r.RSI < 30:
        sig.append((0.5, f"RSI {r.RSI:.0f}: oversold"))
    else:
        sig.append((0.5 if r.RSI >= 50 else -0.5, f"RSI {r.RSI:.0f}: {'positive' if r.RSI >= 50 else 'weak'} momentum"))
    m6 = pct_change(c, 126)
    if not np.isnan(m6):
        sig.append((1 if m6 > 0 else -1, f"6-month momentum {m6:+.1%}"))
    return float(np.mean([v for v, _ in sig])), sig


def fundamental_score(info):
    def get(k):
        v = info.get(k)
        return v if isinstance(v, (int, float)) and not np.isnan(v) else None

    sig = []
    fpe, g, pm = get("forwardPE"), get("revenueGrowth"), get("profitMargins")
    peg = get("trailingPegRatio") or get("pegRatio")
    roe, de, fcf = get("returnOnEquity"), get("debtToEquity"), get("freeCashflow")
    if fpe and fpe > 0: sig.append((1 if fpe < 22 else -1 if fpe > 40 else 0, f"Forward P/E {fpe:.1f}"))
    if peg and peg > 0: sig.append((1 if peg < 1.5 else -1 if peg > 2.5 else 0, f"PEG {peg:.2f}"))
    if g is not None: sig.append((1 if g > 0.10 else -1 if g < 0 else 0, f"Revenue growth {g:+.1%}"))
    if pm is not None: sig.append((1 if pm > 0.15 else -1 if pm < 0 else 0, f"Profit margin {pm:.1%}"))
    if roe is not None: sig.append((1 if roe > 0.15 else -1 if roe < 0 else 0, f"Return on equity {roe:.1%}"))
    if de is not None: sig.append((1 if de < 100 else -1 if de > 200 else 0, f"Debt/Equity {de:.0f}"))
    if fcf is not None: sig.append((1 if fcf > 0 else -1, "Positive free cash flow" if fcf > 0 else "Negative free cash flow"))
    if not sig:
        return None, []
    return float(np.mean([v for v, _ in sig])), sig


def analyst_score(info, price):
    sig, parts = [], []
    tgt = info.get("targetMeanPrice")
    if isinstance(tgt, (int, float)) and price:
        up = tgt / price - 1
        parts.append(max(-1, min(1, up / 0.25)))
        sig.append((1 if up > 0.05 else -1 if up < -0.05 else 0, f"Mean target {tgt:,.2f} ({up:+.1%} vs price)"))
    rec = info.get("recommendationMean")
    if isinstance(rec, (int, float)):
        parts.append((3 - rec) / 2)
        n = info.get("numberOfAnalystOpinions")
        sig.append((1 if rec < 2.5 else -1 if rec > 3.5 else 0,
                    f"Consensus {rec:.1f}/5 ({info.get('recommendationKey', 'n/a')}" + (f", {n} analysts)" if n else ")")))
    if not parts:
        return None, []
    return float(np.mean(parts)), sig


def risk_stats(c):
    r = np.log(c / c.shift()).dropna().iloc[-252:]
    vol = r.std() * np.sqrt(252)
    sharpe = (r.mean() * 252 - 0.04) / vol if vol > 0 else np.nan
    yr = c.iloc[-252:]
    return vol, sharpe, (yr / yr.cummax() - 1).min()


def monte_carlo(c, n_paths=10000, seed=7):
    rng = np.random.default_rng(seed)
    r = np.log(c / c.shift()).dropna().iloc[-504:]
    mu = 0.5 * float(np.clip(r.mean() * 252, -0.20, 0.30)) + 0.5 * 0.08
    sigma = float(r.std() * np.sqrt(252))
    dt, steps = 1 / 252, max(HORIZONS.values())
    z = rng.standard_normal((n_paths, steps))
    paths = c.iloc[-1] * np.exp(np.cumsum((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z, axis=1))
    return paths, mu, sigma


# ------------------------------------------------------------------- charts --
def style(fig, h=420):
    fig.update_layout(template="plotly_dark", height=h, paper_bgcolor=TRANSPARENT,
                      plot_bgcolor=TRANSPARENT, margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", y=1.08))
    return fig


def gauge(score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=round(score * 100),
        title={"text": label(score), "font": {"size": 22, "color": score_color(score)}},
        gauge={"axis": {"range": [-100, 100]}, "bar": {"color": "white", "thickness": 0.25},
               "steps": [{"range": [-100, -50], "color": "#b4234a"}, {"range": [-50, -20], "color": "#f0773c"},
                         {"range": [-20, 20], "color": "#c9a227"}, {"range": [20, 50], "color": "#5cc98a"},
                         {"range": [50, 100], "color": "#1fa463"}]}))
    fig.update_layout(template="plotly_dark", height=260, paper_bgcolor=TRANSPARENT,
                      margin=dict(l=25, r=25, t=60, b=10))
    return fig


def price_chart(d):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.76, 0.24], vertical_spacing=0.03)
    fig.add_trace(go.Scatter(x=d.index, y=d.BBup, line=dict(width=0), showlegend=False, hoverinfo="skip"), 1, 1)
    fig.add_trace(go.Scatter(x=d.index, y=d.BBlo, line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(77,163,255,0.10)", name="Bollinger", hoverinfo="skip"), 1, 1)
    fig.add_trace(go.Candlestick(x=d.index, open=d.Open, high=d.High, low=d.Low, close=d.Close,
                                 increasing_line_color=GREEN, decreasing_line_color=RED, name="Price"), 1, 1)
    fig.add_trace(go.Scatter(x=d.index, y=d.SMA50, name="SMA 50", line=dict(color=AMBER, width=1.6)), 1, 1)
    fig.add_trace(go.Scatter(x=d.index, y=d.SMA200, name="SMA 200", line=dict(color=PURPLE, width=1.6)), 1, 1)
    cols = [GREEN if c >= o else RED for c, o in zip(d.Close, d.Open)]
    fig.add_trace(go.Bar(x=d.index, y=d.Volume, marker_color=cols, opacity=0.6, name="Volume"), 2, 1)
    fig.update_layout(xaxis_rangeslider_visible=False)
    return style(fig, 640)


def fan_chart(c, paths, target):
    hist = c.iloc[-252:]
    fut = pd.bdate_range(c.index[-1], periods=paths.shape[1] + 1)[1:]
    p10, p25, p50, p75, p90 = np.percentile(paths, [10, 25, 50, 75, 90], axis=0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist.values, name="History", line=dict(color=BLUE, width=2)))
    fig.add_trace(go.Scatter(x=fut, y=p90, line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fut, y=p10, line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(167,139,250,0.20)", name="80% range"))
    fig.add_trace(go.Scatter(x=fut, y=p75, line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fut, y=p25, line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(167,139,250,0.40)", name="50% range"))
    fig.add_trace(go.Scatter(x=fut, y=p50, name="Median", line=dict(color=AMBER, width=2.5, dash="dash")))
    if isinstance(target, (int, float)):
        fig.add_trace(go.Scatter(x=[fut[-1]], y=[target], mode="markers", name="Analyst target",
                                 marker=dict(size=15, symbol="star", color=GREEN)))
    return style(fig, 460)


def scenario_chart(table, price):
    fig = go.Figure()
    for name, col, key in (("Bear (10th pct)", RED, "bear"), ("Base (median)", AMBER, "base"), ("Bull (90th pct)", GREEN, "bull")):
        fig.add_trace(go.Bar(name=name, x=list(table), y=[(table[h][key] / price - 1) * 100 for h in table],
                             marker_color=col, text=[f"{(table[h][key] / price - 1):+.1%}" for h in table],
                             textposition="outside"))
    fig.update_layout(barmode="group", yaxis_title="% change vs today")
    return style(fig, 360)


def rsi_chart(d):
    fig = go.Figure()
    fig.add_hrect(y0=70, y1=100, fillcolor=RED, opacity=0.12, line_width=0)
    fig.add_hrect(y0=0, y1=30, fillcolor=GREEN, opacity=0.12, line_width=0)
    fig.add_trace(go.Scatter(x=d.index, y=d.RSI, name="RSI (14)", line=dict(color=PURPLE, width=2)))
    fig.update_yaxes(range=[0, 100])
    fig.update_layout(title="RSI: above 70 overbought, below 30 oversold")
    return style(fig, 300)


def macd_chart(d):
    hist = d.MACD - d.MACDsig
    fig = go.Figure()
    fig.add_trace(go.Bar(x=d.index, y=hist, marker_color=[GREEN if v >= 0 else RED for v in hist],
                         opacity=0.6, name="Histogram"))
    fig.add_trace(go.Scatter(x=d.index, y=d.MACD, name="MACD", line=dict(color=BLUE, width=1.8)))
    fig.add_trace(go.Scatter(x=d.index, y=d.MACDsig, name="Signal", line=dict(color=AMBER, width=1.8)))
    fig.update_layout(title="MACD momentum")
    return style(fig, 300)


# --------------------------------------------------------------------- page --
with st.form("search"):
    c1, c2 = st.columns([6, 1])
    q = c1.text_input("Stock", value=st.session_state.get("q", ""), label_visibility="collapsed",
                      placeholder="Type a company name or ticker: Apple, Nvidia, Maybank, 1155.KL ...")
    submitted = c2.form_submit_button("Analyse 🔍", use_container_width=True)
if submitted and q.strip():
    st.session_state["q"] = q.strip()
query = st.session_state.get("q")

if not query:
    st.markdown('<div class="hero"><h1>📈 Stock Analyst</h1>'
                '<p>Type a stock above to see charts, trend, fundamentals, risk and a 3-month to 1-year outlook.</p></div>',
                unsafe_allow_html=True)
    st.info("Examples: **Apple**, **Tesla**, **Nvidia**, **Maybank**, **1155.KL** (Malaysian stocks end with .KL)")
    st.stop()

with st.spinner("Fetching data and crunching numbers ..."):
    ticker = resolve_ticker(query)
    raw, info = fetch(ticker)

if raw is None or len(raw.dropna(subset=["Close"])) < 250:
    st.error(f"Not enough price history for '{query}' (tried ticker {ticker}). Check the name or try the ticker symbol.")
    st.stop()

df = add_indicators(raw.dropna(subset=["Close"]))
c = df["Close"]
price, prev = float(c.iloc[-1]), float(c.iloc[-2])
day = price / prev - 1
cur = info.get("currency", "")

# --- Hero + cards
st.markdown(f'<div class="hero"><h1>{info.get("longName", ticker)} · {ticker}</h1>'
            f'<p>{info.get("sector", "")} {"· " + info["industry"] if info.get("industry") else ""}'
            f' &nbsp;|&nbsp; data as of {df.index[-1].date()}</p></div>', unsafe_allow_html=True)

lo, hi = float(c.iloc[-252:].min()), float(c.iloc[-252:].max())
rate = info.get("dividendRate")
dy = f"{rate / price * 100:.2f}%" if isinstance(rate, (int, float)) and rate else "none"
cols = st.columns(6)
cards = [
    card("Price", f"{price:,.2f} {cur}", f"{day:+.2%} today", GREEN if day >= 0 else RED),
    card("Market cap", big(info.get("marketCap"))),
    card("52-week range", f"{lo:,.2f} – {hi:,.2f}", f"now at {(price - lo) / (hi - lo):.0%} of range", BLUE),
    card("P/E (trailing / fwd)", f"{num(info.get('trailingPE'), 1)} / {num(info.get('forwardPE'), 1)}"),
    card("Dividend yield", dy),
    card("Beta", num(info.get("beta")), "market sensitivity"),
]
for col, html in zip(cols, cards):
    col.markdown(html, unsafe_allow_html=True)

st.write("")

# --- Scores
t_score, t_sig = technical_score(df)
f_score, f_sig = fundamental_score(info)
a_score, a_sig = analyst_score(info, price)
parts = [(t_score, 0.40), (f_score, 0.35), (a_score, 0.25)]
used = [(s, w) for s, w in parts if s is not None]
overall = sum(s * w for s, w in used) / sum(w for _, w in used)
vol, sharpe, mdd = risk_stats(c)
paths, mu, sigma = monte_carlo(c)
table = {}
for h, d_ in HORIZONS.items():
    end = paths[:, d_ - 1]
    q10, q50, q90 = np.percentile(end, [10, 50, 90])
    table[h] = dict(bear=q10, base=q50, bull=q90, gain=float((end > price).mean()),
                    up20=float((end > price * 1.2).mean()), dn20=float((end < price * 0.8).mean()))

g1, g2 = st.columns([1, 1.6])
with g1:
    st.plotly_chart(gauge(overall), use_container_width=True)
with g2:
    names, vals = [], []
    for n, s in (("Technical", t_score), ("Fundamental", f_score), ("Analyst", a_score)):
        if s is not None:
            names.append(n); vals.append(s * 100)
    bar = go.Figure(go.Bar(x=vals, y=names, orientation="h", marker_color=[score_color(v / 100) for v in vals],
                           text=[f"{v:+.0f}" for v in vals], textposition="outside"))
    bar.update_xaxes(range=[-110, 110])
    st.plotly_chart(style(bar, 170), use_container_width=True)
    t3, t12 = table["3 months"], table["1 year"]
    near = "positive" if t_score > 0.2 else "negative" if t_score < -0.2 else "mixed"
    st.markdown(
        f'<div class="verdict" style="background:linear-gradient(120deg,{score_color(overall)}33,#171b26);'
        f'border:1px solid {score_color(overall)}">'
        f'<b>{label(overall)}.</b> Short-term trend is <b>{near}</b>. Simulated 3-month range '
        f'<b>{t3["bear"]:,.2f} – {t3["bull"]:,.2f}</b> (median {t3["base"]:,.2f}); 1-year range '
        f'<b>{t12["bear"]:,.2f} – {t12["bull"]:,.2f}</b> (median {t12["base"]:,.2f}). '
        f'Chance of being higher in a year: <b>{t12["gain"]:.0%}</b>.</div>', unsafe_allow_html=True)

st.write("")

# --- Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Price & Trend", "🎯 Forecast", "📊 Indicators",
                                         "🧾 Fundamentals & Analysts", "⚠️ Risk"])

with tab1:
    ranges = {"3M": 63, "6M": 126, "1Y": 252, "2Y": 504, "5Y": 1260}
    pick = st.radio("Range", list(ranges), index=2, horizontal=True, label_visibility="collapsed")
    st.plotly_chart(price_chart(df.iloc[-ranges[pick]:]), use_container_width=True)
    st.markdown("**Technical signals**")
    st.markdown("".join(chip(v, n) for v, n in t_sig), unsafe_allow_html=True)
    r = df.iloc[-1]
    s1, s2, s3 = st.columns(3)
    s1.markdown(card("Support (3M low)", f"{c.iloc[-63:].min():,.2f}", color=GREEN), unsafe_allow_html=True)
    s2.markdown(card("Resistance (3M high)", f"{c.iloc[-63:].max():,.2f}", color=RED), unsafe_allow_html=True)
    s3.markdown(card("Daily swing (ATR)", f"{r.ATR:,.2f}", f"{r.ATR / price:.1%} of price", AMBER), unsafe_allow_html=True)

with tab2:
    st.plotly_chart(fan_chart(c, paths, info.get("targetMeanPrice")), use_container_width=True)
    st.caption(f"10,000-path simulation. Assumed drift {mu:.1%}/yr, volatility {sigma:.1%}/yr. "
               "Shaded bands show the 50% and 80% likely ranges.")
    cols = st.columns(3)
    for col, (h, v) in zip(cols, table.items()):
        col.metric(f"{h} · median", f"{v['base']:,.2f}", f"{v['base'] / price - 1:+.1%}")
        col.progress(v["gain"], text=f"Chance of gain {v['gain']:.0%}")
        col.caption(f"Range {v['bear']:,.2f} – {v['bull']:,.2f}  |  >+20%: {v['up20']:.0%}  |  <-20%: {v['dn20']:.0%}")
    st.plotly_chart(scenario_chart(table, price), use_container_width=True)

with tab3:
    d = df.iloc[-252:]
    st.plotly_chart(rsi_chart(d), use_container_width=True)
    st.plotly_chart(macd_chart(d), use_container_width=True)

with tab4:
    a, b = st.columns(2)
    with a:
        st.subheader("Fundamentals")
        if f_sig:
            st.markdown("".join(chip(v, n) for v, n in f_sig), unsafe_allow_html=True)
        else:
            st.write("Fundamental data unavailable for this stock.")
    with b:
        st.subheader("Analyst view")
        if a_sig:
            st.markdown("".join(chip(v, n) for v, n in a_sig), unsafe_allow_html=True)
        else:
            st.write("Analyst data unavailable for this stock.")
    if info.get("longBusinessSummary"):
        with st.expander("About the company"):
            st.write(info["longBusinessSummary"])

with tab5:
    r1, r2, r3 = st.columns(3)
    lvl = "Low" if vol < 0.20 else "Moderate" if vol < 0.35 else "High"
    r1.markdown(card("Volatility (1Y)", f"{vol:.1%}", f"{lvl} risk", AMBER), unsafe_allow_html=True)
    r2.markdown(card("Sharpe ratio (1Y)", num(sharpe), "return per unit of risk", BLUE), unsafe_allow_html=True)
    r3.markdown(card("Max drawdown (1Y)", f"{mdd:.1%}", "worst peak-to-trough fall", RED), unsafe_allow_html=True)
    dd = (c.iloc[-756:] / c.iloc[-756:].cummax() - 1) * 100
    fig = go.Figure(go.Scatter(x=dd.index, y=dd.values, fill="tozeroy", line=dict(color=RED),
                               fillcolor="rgba(255,84,112,0.30)", name="Drawdown"))
    fig.update_layout(title="Drawdown from peak, last 3 years (%)")
    st.plotly_chart(style(fig, 320), use_container_width=True)
    rets = c.pct_change().dropna().iloc[-252:] * 100
    hist = go.Figure(go.Histogram(x=rets, nbinsx=40, marker_color=BLUE))
    hist.update_layout(title="Distribution of daily returns, last year (%)")
    st.plotly_chart(style(hist, 300), use_container_width=True)

st.caption("Statistical and educational output based on past data. Not financial advice.")