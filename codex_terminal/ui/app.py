from __future__ import annotations

from typing import Dict

import pandas as pd
import streamlit as st

from codex_terminal.analytics.brief import (
    CORE_BRIEF_TICKERS,
    correlation_snapshot,
    cross_asset_signal_table,
    leadership_table,
    rolling_corr_series,
    what_changed_table,
)
from codex_terminal.analytics.exposures import classify_holdings, compare_exposure_summary, summarize_exposures
from codex_terminal.analytics.factors import compute_factor_attribution
from codex_terminal.analytics.house import HOUSE_BENCHMARK_MODES, build_market_beating_portfolio
from codex_terminal.analytics.macro import classify_regime, macro_snapshots, regime_implications
from codex_terminal.analytics.metrics import rolling_total_return, summary_stats
from codex_terminal.analytics.portfolio import (
    compute_stress_table,
    equal_weight_portfolio,
    inverse_vol_weights,
    leverage_to_match_spy_vol,
    random_search_optimize,
)
from codex_terminal.analytics.screener import compute_screener_scores
from codex_terminal.config.universe import (
    UNIVERSE,
    VANGUARD_TARGET_RETIREMENT_LABEL,
    tickers,
    universe_by_ticker,
)
from codex_terminal.data.funds import fetch_fund_profiles
from codex_terminal.data.fred import DEFAULT_FRED_SERIES, fetch_fred_bundle, infer_fred_status
from codex_terminal.data.market_data import compute_returns, fetch_price_history, infer_status, latest_available
from codex_terminal.data.vanguard import fetch_vanguard_benchmark_history, infer_vanguard_target_fund
from codex_terminal.portfolio.benchmarks import degraded_vanguard_state
from codex_terminal.portfolio.compare import compare_stats, normalize_portfolio_frame, portfolio_returns


PAGES = ["Welcome", "Morning Brief", "Terminal", "Screener", "Portfolio Lab", "Compare", "Morningstar", "Macro", "Learn"]
DEFAULT_FUND_TICKERS = ["VTSAX", "VFIAX", "VWELX", "FBALX"]


def _format_pct(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value:.2%}"


def _format_float(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value:.2f}"


def _format_billions(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    return f"${value / 1_000_000_000:.1f}B"


def _inject_terminal_theme() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top right, rgba(11, 88, 122, 0.28), transparent 26%),
                radial-gradient(circle at bottom left, rgba(38, 68, 98, 0.20), transparent 24%),
                linear-gradient(180deg, #060b12 0%, #0a1018 35%, #081019 100%);
            color: #d7e2ea;
        }
        [data-testid="stSidebar"] {
            background:
                linear-gradient(180deg, rgba(6, 12, 18, 0.98) 0%, rgba(10, 18, 28, 0.98) 100%);
            border-right: 1px solid rgba(138, 164, 178, 0.12);
        }
        h1, h2, h3, h4, .stMarkdown, label, .stCaption {
            color: #d7e2ea !important;
        }
        .block-container {
            max-width: 1600px;
            padding-top: 1.1rem;
            padding-bottom: 2rem;
        }
        [data-testid="stMetric"] {
            background: linear-gradient(180deg, rgba(10, 18, 28, 0.95), rgba(9, 14, 22, 0.94));
            border: 1px solid rgba(85, 122, 141, 0.22);
            border-radius: 10px;
            padding: 0.6rem 0.8rem;
        }
        [data-testid="stDataFrame"], .stAlert, .stTextInput, .stSelectbox, .stMultiSelect {
            border-radius: 10px;
        }
        .terminal-band {
            background: linear-gradient(90deg, rgba(8, 63, 84, 0.32), rgba(13, 20, 28, 0.42));
            border: 1px solid rgba(88, 126, 145, 0.16);
            border-radius: 10px;
            padding: 0.7rem 0.9rem;
            margin: 0.15rem 0 0.9rem 0;
        }
        .terminal-kicker {
            color: #7db6cf;
            font-size: 0.74rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }
        .desk-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.7rem;
            margin: 0.3rem 0 1rem 0;
        }
        .desk-card {
            background: linear-gradient(180deg, rgba(10, 18, 28, 0.95), rgba(8, 13, 20, 0.94));
            border: 1px solid rgba(82, 119, 139, 0.2);
            border-radius: 10px;
            padding: 0.7rem 0.85rem;
        }
        .desk-label {
            color: #6ca6c0;
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin-bottom: 0.15rem;
        }
        .desk-value {
            color: #f2f7fb;
            font-size: 1.2rem;
            font-weight: 600;
        }
        .desk-sub {
            color: #93a9b5;
            font-size: 0.76rem;
            margin-top: 0.2rem;
        }
        .section-title {
            margin: 0.45rem 0 0.55rem 0;
            color: #ebf4fb;
            font-weight: 600;
            letter-spacing: 0.04em;
        }
        .hero-card {
            background:
                radial-gradient(circle at top right, rgba(59, 130, 246, 0.18), transparent 28%),
                linear-gradient(135deg, rgba(12, 21, 32, 0.98), rgba(8, 16, 24, 0.98));
            border: 1px solid rgba(106, 149, 173, 0.22);
            border-radius: 16px;
            padding: 1.1rem 1.2rem;
            margin: 0.35rem 0 1rem 0;
            box-shadow: 0 14px 30px rgba(2, 8, 14, 0.28);
        }
        .hero-eyebrow {
            color: #84b6cf;
            font-size: 0.72rem;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            margin-bottom: 0.4rem;
        }
        .hero-title {
            color: #f2f7fb;
            font-size: 1.75rem;
            line-height: 1.15;
            font-weight: 700;
            margin-bottom: 0.55rem;
        }
        .hero-copy {
            color: #aec2cf;
            max-width: 56rem;
            font-size: 0.98rem;
            line-height: 1.6;
        }
        .info-panel {
            background: linear-gradient(180deg, rgba(10, 18, 28, 0.92), rgba(8, 13, 20, 0.92));
            border: 1px solid rgba(82, 119, 139, 0.18);
            border-radius: 12px;
            padding: 0.9rem 1rem;
            margin: 0.2rem 0 0.9rem 0;
        }
        .info-panel-title {
            color: #dceaf3;
            font-size: 0.92rem;
            font-weight: 600;
            margin-bottom: 0.35rem;
        }
        .info-panel-copy {
            color: #9fb3c0;
            font-size: 0.88rem;
            line-height: 1.55;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _format_stats_frame(stats: pd.DataFrame):
    display = stats.T.copy()
    ordered = [metric for metric in ["CAGR", "Volatility", "Sharpe", "Sortino", "Max Drawdown"] if metric in display.columns]
    display = display[ordered]
    return display.style.format(
        {
            "CAGR": _format_pct,
            "Volatility": _format_pct,
            "Max Drawdown": _format_pct,
            "Sharpe": _format_float,
            "Sortino": _format_float,
        }
    )


def _house_holdings_frame(context: Dict[str, object]) -> pd.DataFrame:
    model = context["house_model"]
    weights = model.holdings.copy()
    known = universe_by_ticker()
    weights["proxy"] = weights["ticker"].map(lambda x: known[x].proxy_description if x in known else "unresolved")
    weights["sleeve"] = weights["ticker"].map(lambda x: known[x].sleeve if x in known else "Unresolved")
    return weights[["ticker", "proxy", "sleeve", "strategic_weight", "stance", "composite_percentile", "tilt_vs_strategic", "weight"]]


def _house_benchmark_series_and_stats(context: Dict[str, object]) -> tuple[pd.Series, dict]:
    model = context["house_model"]
    if not model.target_vol_series.empty and model.target_vol_stats:
        return model.target_vol_series, model.target_vol_stats
    return model.series, model.stats


def _render_desk_grid(items: list[tuple[str, str, str]]) -> None:
    blocks = []
    for label, value, sub in items:
        blocks.append(
            f'<div class="desk-card"><div class="desk-label">{label}</div>'
            f'<div class="desk-value">{value}</div><div class="desk-sub">{sub}</div></div>'
        )
    st.markdown(f'<div class="desk-grid">{"".join(blocks)}</div>', unsafe_allow_html=True)


def _render_section_title(title: str) -> None:
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


@st.dialog("Configure Vanguard Target Retirement Benchmark")
def _configure_vanguard_dialog() -> None:
    with st.form("vanguard_dialog_form"):
        age = st.number_input("Your age in years", min_value=18, max_value=100, value=35, step=1)
        start_date = st.date_input("Benchmark start date", value=pd.Timestamp("2018-01-01"))
        end_date = st.date_input("Benchmark end date", value=pd.Timestamp.today().date())
        submitted = st.form_submit_button("Load benchmark")
        if submitted:
            fund = infer_vanguard_target_fund(int(age))
            st.session_state["vanguard_benchmark_config"] = {
                "age": int(age),
                "start": pd.Timestamp(start_date).date().isoformat(),
                "end": pd.Timestamp(end_date).date().isoformat(),
                "ticker": fund.ticker,
                "label": fund.label,
            }
            st.rerun()


def _render_factor_block(name: str, attribution) -> None:
    _render_section_title(name)
    if attribution is None:
        st.info("Not enough data to estimate factor attribution yet.")
        return
    cols = st.columns(2)
    cols[0].metric("Annualized Alpha", _format_pct(attribution.annualized_alpha))
    cols[1].metric("R-Squared", _format_float(attribution.r_squared))
    st.dataframe(
        attribution.exposures.style.format({"Exposure": _format_float}),
        use_container_width=True,
    )


def _load_market_context() -> Dict[str, object]:
    mode = st.session_state.get("house_benchmark_mode", "Strategic + Tactical")
    prices = fetch_price_history(tickers())
    returns = compute_returns(prices)
    status = infer_status(prices, tickers())
    fred_bundle = fetch_fred_bundle(DEFAULT_FRED_SERIES)
    fred_status = infer_fred_status(fred_bundle, DEFAULT_FRED_SERIES)
    regime = classify_regime(fred_bundle)
    macro_snapshot = macro_snapshots(fred_bundle)
    screener = compute_screener_scores(prices, regime=regime.regime)
    spy_returns = returns.get("SPY", pd.Series(dtype=float))
    house_model = build_market_beating_portfolio(returns, screener, spy_returns, mode=mode)
    return {
        "prices": prices,
        "returns": returns,
        "status": status,
        "fred_bundle": fred_bundle,
        "fred_status": fred_status,
        "regime": regime,
        "macro_snapshot": macro_snapshot,
        "screener": screener,
        "house_model": house_model,
    }


def _render_sidebar() -> str:
    st.sidebar.title("codex_terminal")
    st.sidebar.caption("Institutional-style cross-asset research terminal")
    st.sidebar.markdown("**House Benchmark**")
    st.sidebar.caption("Strategic core plus tactical overlay, then volatility-targeted to SPY.")
    current_mode = st.session_state.get("house_benchmark_mode", "Strategic + Tactical")
    st.session_state["house_benchmark_mode"] = st.sidebar.selectbox(
        "Benchmark Mode",
        HOUSE_BENCHMARK_MODES,
        index=HOUSE_BENCHMARK_MODES.index(current_mode) if current_mode in HOUSE_BENCHMARK_MODES else 0,
    )
    default_page = "Welcome" if not st.session_state.get("onboarding_seen", False) else "Morning Brief"
    pending_page = st.session_state.pop("pending_nav_page", None)
    current_page = pending_page if pending_page in PAGES else st.session_state.get("nav_page", default_page)
    if current_page not in PAGES:
        current_page = default_page
    if "module_radio" not in st.session_state:
        st.session_state["module_radio"] = current_page
    elif st.session_state["module_radio"] not in PAGES:
        st.session_state["module_radio"] = default_page
    elif pending_page in PAGES:
        st.session_state["module_radio"] = pending_page

    page = st.sidebar.radio("Module", PAGES, key="module_radio")
    st.session_state["nav_page"] = page

    with st.sidebar.expander("First-Time Guide", expanded=False):
        st.caption("Best order for a new user")
        st.write("1. Morning Brief")
        st.write("2. Terminal")
        st.write("3. Compare")
        st.write("4. Morningstar")
        st.write("5. Portfolio Lab")
        if st.button("Show Welcome Again", use_container_width=True):
            st.session_state["onboarding_seen"] = False
            st.session_state["pending_nav_page"] = "Welcome"
            st.rerun()

    return page


def _render_header(context: Dict[str, object]) -> None:
    regime = context["regime"]
    status = context["status"]
    fred_status = context["fred_status"]
    st.title("codex_terminal")
    st.caption("Live cross-asset research terminal with market, macro, benchmark, and portfolio diagnostics.")
    cols = st.columns([2, 2, 2, 3])
    cols[0].metric("Macro Regime", regime.regime)
    cols[1].metric("Regime Confidence", regime.confidence)
    cols[2].metric("Macro Data", "Live" if fred_status.ok else "Degraded")
    if status.ok and fred_status.ok:
        cols[3].success(f"{status.message} {fred_status.message}")
    elif status.ok and not fred_status.ok:
        cols[3].warning(f"{status.message} {fred_status.message}")
    elif not status.ok and fred_status.ok:
        cols[3].warning(f"{status.message} {fred_status.message}")
    else:
        cols[3].error(f"{status.message} {fred_status.message}")
    house_model = context["house_model"]
    _, house_stats = _house_benchmark_series_and_stats(context)
    _render_desk_grid(
        [
            ("Macro Regime", regime.regime, f"confidence {regime.confidence.lower()}"),
            ("House Sharpe", _format_float(house_stats.get("Sharpe")), "vol-targeted"),
            ("Vol-Target Leverage", _format_float(house_model.vol_target_leverage), "match SPY risk"),
            ("Data Coverage", "Live" if status.ok else "Degraded", status.message),
        ]
    )
    st.markdown(
        '<div class="terminal-band"><div class="terminal-kicker">Desk Summary</div>'
        "Cross-asset signals, benchmark diagnostics, regime context, and a dynamic house benchmark driven by strategic weights and tactical overlays."
        "</div>",
        unsafe_allow_html=True,
    )


def _render_terminal(context: Dict[str, object]) -> None:
    st.subheader("Terminal")
    screener = context["screener"]
    if screener.empty:
        st.warning("No market data available for the terminal summary.")
        return

    top = screener.head(5).copy()
    bottom = screener.tail(5).copy()
    for frame in [top, bottom]:
        frame.insert(0, "Ticker", frame.index)
        frame.insert(1, "Proxy", [universe_by_ticker().get(ticker).proxy_description for ticker in frame.index])
        frame.insert(2, "Sleeve", [universe_by_ticker().get(ticker).sleeve for ticker in frame.index])
    top = top[["Ticker", "Proxy", "Sleeve", "1M", "3M", "6M", "Composite Score", "Stance"]]
    bottom = bottom[["Ticker", "Proxy", "Sleeve", "1M", "3M", "6M", "Composite Score", "Stance"]]
    latest = latest_available(context["prices"])
    house_model = context["house_model"]
    house_series, house_stats = _house_benchmark_series_and_stats(context)

    col1, col2 = st.columns(2)
    with col1:
        _render_section_title("Top Leadership")
        st.dataframe(top.style.format({"1M": _format_pct, "3M": _format_pct, "6M": _format_pct, "Composite Score": _format_float}))
    with col2:
        _render_section_title("Weakest Sleeves")
        st.dataframe(bottom.style.format({"1M": _format_pct, "3M": _format_pct, "6M": _format_pct, "Composite Score": _format_float}))

    _render_section_title("Latest Prices Snapshot")
    latest_frame = latest.rename("Latest").to_frame()
    latest_frame["Ticker"] = latest_frame.index
    latest_frame["Proxy"] = [universe_by_ticker().get(ticker).proxy_description for ticker in latest_frame.index]
    latest_frame["Sleeve"] = [universe_by_ticker().get(ticker).sleeve for ticker in latest_frame.index]
    st.dataframe(latest_frame[["Ticker", "Proxy", "Sleeve", "Latest"]])

    if house_stats:
        _render_section_title("Market Beating Portfolio Snapshot")
        _render_desk_grid(
            [
                ("CAGR", _format_pct(house_stats.get("CAGR")), "vol-targeted"),
                ("Volatility", _format_pct(house_stats.get("Volatility")), "targeted benchmark"),
                ("Sharpe", _format_float(house_stats.get("Sharpe")), "risk-adjusted"),
                ("Max Drawdown", _format_pct(house_stats.get("Max Drawdown")), "historical"),
            ]
        )
        _render_section_title("Market Beating Portfolio Holdings")
        st.dataframe(
            _house_holdings_frame(context).style.format(
                {
                    "strategic_weight": _format_pct,
                    "composite_percentile": _format_pct,
                    "tilt_vs_strategic": _format_pct,
                    "weight": _format_pct,
                }
            ),
            use_container_width=True,
        )

    spy_returns = context["returns"].get("SPY", pd.Series(dtype=float))
    if not house_series.empty and not spy_returns.empty:
        _render_section_title("Market Beating Portfolio vs SPY")
        vs_cols = st.columns([1.15, 0.85])
        with vs_cols[0]:
            comparison = pd.concat(
                [
                    (1 + house_series).cumprod().rename("Market Beating Portfolio"),
                    (1 + spy_returns).cumprod().rename("SPY"),
                ],
                axis=1,
            ).dropna()
            if not comparison.empty:
                st.line_chart(comparison)
        with vs_cols[1]:
            diag = compare_stats(house_series, spy_returns)
            st.dataframe(diag.style.format({"Value": "{:.2f}"}), use_container_width=True)

        rel_cols = st.columns(2)
        with rel_cols[0]:
            rolling_excess = (
                rolling_total_return(house_series, 63).rename("House 3M")
                - rolling_total_return(spy_returns, 63).rename("SPY 3M")
            ).dropna()
            _render_section_title("Rolling 3M Excess Return")
            if not rolling_excess.empty:
                st.line_chart(rolling_excess.to_frame(name="Excess vs SPY"))
        with rel_cols[1]:
            house_exposure = classify_holdings(house_model.holdings[["ticker", "weight"]])
            spy_exposure = classify_holdings(pd.DataFrame({"ticker": ["SPY"], "weight": [1.0]}))
            diff = compare_exposure_summary(house_exposure, spy_exposure, "Asset Class")
            _render_section_title("Asset-Class Difference vs SPY")
            st.dataframe(
                diff.style.format({"Left": _format_pct, "Right": _format_pct, "Difference": _format_pct}),
                use_container_width=True,
            )

    regime = context["regime"]
    _render_section_title("What Changed / What Matters / What To Watch")
    st.info(
        f"{regime.summary} Tactical leadership currently favors the top-ranked sleeves in the screener. "
        "The house benchmark now starts from a diversified strategic core, then adjusts tactical-eligible sleeves using the screener percentile, stance, and regime fit before risk-targeting against SPY. The vol-targeted house portfolio is the default benchmark throughout the app."
    )


def _render_morning_brief(context: Dict[str, object]) -> None:
    st.subheader("Morning Brief")
    regime = context["regime"]
    house_model = context["house_model"]
    _, house_stats = _house_benchmark_series_and_stats(context)
    pulse = cross_asset_signal_table(context["prices"])
    leaders = leadership_table(context["screener"])
    changes = what_changed_table(context["screener"])
    corr_current, corr_delta = correlation_snapshot(context["returns"], CORE_BRIEF_TICKERS)

    _render_desk_grid(
        [
            ("Regime", regime.regime, regime.confidence.lower()),
            ("Benchmark Mode", house_model.mode, "active"),
            ("House Sharpe", _format_float(house_stats.get("Sharpe")), "vol-targeted"),
            ("House Leverage", _format_float(house_model.vol_target_leverage), "vs SPY"),
        ]
    )

    tabs = st.tabs(["Market Pulse", "What Changed", "What's Hot / Not", "Correlation", "Implications"])

    with tabs[0]:
        _render_section_title("Market Pulse")
        if pulse.empty:
            st.info("Not enough market history yet.")
        else:
            st.dataframe(
                pulse.style.format({"1M": _format_pct, "Prev 1M": _format_pct, "1M Delta": _format_pct}),
                use_container_width=True,
            )

    with tabs[1]:
        _render_section_title("What Changed")
        if changes.empty:
            st.info("Not enough signal history yet.")
        else:
            st.dataframe(
                changes.head(12).style.format(
                    {"1W Proxy": _format_pct, "1M": _format_pct, "3M": _format_pct, "Momentum Delta": _format_pct, "Trend": "{:.2f}"}
                ),
                use_container_width=True,
            )

    with tabs[2]:
        _render_section_title("What's Hot")
        if leaders.empty:
            st.info("Leadership unavailable.")
        else:
            cols = st.columns(2)
            cols[0].dataframe(
                leaders.head(8).style.format(
                    {"1W Proxy": _format_pct, "1M": _format_pct, "3M": _format_pct, "6M": _format_pct, "Trend": "{:.2f}", "Composite Score": "{:.2f}"}
                ),
                use_container_width=True,
            )
            cols[1].markdown("**What's Not**")
            cols[1].dataframe(
                leaders.tail(8).sort_values("Composite Score").style.format(
                    {"1W Proxy": _format_pct, "1M": _format_pct, "3M": _format_pct, "6M": _format_pct, "Trend": "{:.2f}", "Composite Score": "{:.2f}"}
                ),
                use_container_width=True,
            )

    with tabs[3]:
        _render_section_title("Diversification Health")
        ccols = st.columns(2)
        with ccols[0]:
            if not corr_current.empty:
                st.markdown("**Current Correlation**")
                st.dataframe(corr_current.style.format("{:.2f}"), use_container_width=True)
        with ccols[1]:
            if not corr_delta.empty:
                st.markdown("**Correlation Change vs Prior Window**")
                st.dataframe(corr_delta.style.format("{:.2f}"), use_container_width=True)
        roll = pd.concat(
            [
                rolling_corr_series(context["returns"], "SPY", "TLT").rename("SPY vs TLT"),
                rolling_corr_series(context["returns"], "SPY", "PDBC").rename("SPY vs PDBC"),
                rolling_corr_series(context["returns"], "SPY", "GLDM").rename("SPY vs GLDM"),
            ],
            axis=1,
        ).dropna(how="all")
        if not roll.empty:
            st.markdown("**Rolling Correlation**")
            st.line_chart(roll)

    with tabs[4]:
        _render_section_title("Implications")
        notes = []
        if not leaders.empty:
            notes.append(f"Top leadership: {', '.join(leaders.head(3).index.tolist())}.")
            notes.append(f"Weakest sleeves: {', '.join(leaders.tail(3).index.tolist())}.")
        notes.extend(house_model.diagnostics)
        notes.append("If cross-asset correlations are rising, diversification is giving you less protection exactly when you want it most.")
        for note in notes:
            st.write(f"- {note}")


def _render_welcome(context: Dict[str, object]) -> None:
    st.subheader("Welcome")
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-eyebrow">Cross-Asset Decision Terminal</div>
            <div class="hero-title">One place to understand the market, evaluate a portfolio, and build conviction.</div>
            <div class="hero-copy">
                codex_terminal is built to answer three practical questions:
                what is happening in markets right now, what that means for a diversified long-term portfolio,
                and how your portfolio compares to disciplined benchmarks like SPY, a Vanguard target-date fund,
                and the Market Beating Portfolio.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _render_desk_grid(
        [
            ("Best First Tab", "Morning Brief", "start here"),
            ("Daily Use", "Terminal", "market dashboard"),
            ("Fund Research", "Morningstar", "mutual fund work"),
            ("Portfolio Work", "Compare", "upload a portfolio"),
        ]
    )

    intro_cols = st.columns([1.1, 0.9])
    with intro_cols[0]:
        st.markdown(
            """
            <div class="info-panel">
                <div class="info-panel-title">What You Should Expect In The First Five Minutes</div>
                <div class="info-panel-copy">
                    Read the brief, check whether leadership is broad or narrow, then use Compare if you already
                    have a portfolio. If you do not, the Portfolio Lab and house benchmark will show you how this
                    app thinks about building one.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        _render_section_title("What This App Is For")
        st.write(
            "- understanding current market leadership across asset classes"
        )
        st.write(
            "- seeing how diversification is working or failing in real time"
        )
        st.write(
            "- comparing a personal portfolio to SPY, a Vanguard target-date benchmark, and the Market Beating Portfolio"
        )
        st.write(
            "- learning how macro regime, factor tilts, and cross-asset correlations affect portfolio construction"
        )

        _render_section_title("How To Use It")
        st.write("1. Start on `Morning Brief` to understand what changed in markets.")
        st.write("2. Use `Terminal` for the broader dashboard view.")
        st.write("3. Use `Morningstar` if you want to research mutual funds, compare them, or build a fund portfolio.")
        st.write("4. Use `Compare` if you want to evaluate a real portfolio.")
        st.write("5. Use `Portfolio Lab` if you want to test or build allocations.")

    with intro_cols[1]:
        _render_section_title("Who This Is Built For")
        st.write(
            "This app is designed for an investor who wants professional-grade market context without needing full buy-side training."
        )
        st.write(
            "It is especially useful if you are comfortable with markets but want help thinking in terms of diversification, regimes, factors, and benchmark-relative risk."
        )

        st.markdown(
            """
            <div class="info-panel">
                <div class="info-panel-title">A Good First Question To Ask</div>
                <div class="info-panel-copy">
                    Is the current market rewarding broad risk-taking, narrow leadership, or diversification?
                    If you can answer that, the rest of the app becomes much easier to interpret.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        _render_section_title("What To Expect")
        st.write(
            "You will see a lot of information. That is intentional. The goal is not to simplify markets into a single signal, but to help you build conviction faster."
        )
        st.write(
            "The app is most useful when read top-down: start with the brief, then move into the diagnostics that matter for your portfolio."
        )

    _render_section_title("What Each Tab Does")
    tab_frame = pd.DataFrame(
        [
            {"Tab": "Morning Brief", "Purpose": "One-tab market read: what changed, what is leading, what matters now."},
            {"Tab": "Terminal", "Purpose": "The full command-center view across markets, benchmark positioning, and cross-asset context."},
            {"Tab": "Screener", "Purpose": "Ranks sleeves by trend, momentum, structural context, and macro fit."},
            {"Tab": "Portfolio Lab", "Purpose": "Build and test portfolio ideas and compare them to SPY and the house benchmark."},
            {"Tab": "Compare", "Purpose": "Upload a portfolio and compare it to key benchmarks and exposures."},
            {"Tab": "Morningstar", "Purpose": "Research mutual funds, compare them side by side, and build a model fund portfolio."},
            {"Tab": "Macro", "Purpose": "Understand the current regime and how it tends to affect assets and factors."},
            {"Tab": "Learn", "Purpose": "Explain the framework, what may be failing, and what to pay attention to next."},
        ]
    )
    st.dataframe(tab_frame, use_container_width=True)

    cta_cols = st.columns([0.55, 0.45])
    with cta_cols[0]:
        if st.button("Start With Morning Brief", use_container_width=True):
            st.session_state["onboarding_seen"] = True
            st.session_state["pending_nav_page"] = "Morning Brief"
            st.rerun()
    with cta_cols[1]:
        if st.button("Go To Compare", use_container_width=True):
            st.session_state["onboarding_seen"] = True
            st.session_state["pending_nav_page"] = "Compare"
            st.rerun()


def _parse_fund_tickers(raw_value: str) -> list[str]:
    tokens = [token.strip().upper() for token in raw_value.replace("\n", ",").split(",")]
    return [token for token in dict.fromkeys(tokens) if token]


def _render_morningstar(context: Dict[str, object]) -> None:
    st.subheader("Morningstar")
    st.write("Mutual fund research, side-by-side comparison, and model fund portfolio building.")

    with st.container():
        st.markdown(
            """
            <div class="info-panel">
                <div class="info-panel-title">How To Use This Page</div>
                <div class="info-panel-copy">
                    Start with a short list of mutual funds you actually want to compare. The app will pull live history,
                    surface basic fund metadata, compare performance and diversification, and let you build a simple model
                    portfolio against SPY and the Market Beating Portfolio.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    default_raw = ", ".join(st.session_state.get("morningstar_tickers", DEFAULT_FUND_TICKERS))
    raw_tickers = st.text_input(
        "Mutual fund tickers",
        value=default_raw,
        help="Enter Yahoo Finance mutual fund tickers, separated by commas.",
    )
    fund_tickers = _parse_fund_tickers(raw_tickers)
    st.session_state["morningstar_tickers"] = fund_tickers
    if not fund_tickers:
        st.info("Enter at least one mutual fund ticker to begin.")
        return

    horizon = st.selectbox(
        "History window",
        ["3Y", "5Y", "10Y", "Max"],
        index=1,
        help="Used for mutual fund comparison and portfolio-building history.",
    )
    horizon_map = {
        "3Y": "2023-01-01",
        "5Y": "2021-01-01",
        "10Y": "2016-01-01",
        "Max": "2010-01-01",
    }
    benchmark_options = fund_tickers + ["SPY", "Market Beating Portfolio"]
    selected_benchmark = st.selectbox("Comparison benchmark", benchmark_options, index=min(len(benchmark_options) - 1, benchmark_options.index("SPY") if "SPY" in benchmark_options else 0))

    fund_prices = fetch_price_history(fund_tickers + ["SPY"], start=horizon_map[horizon])
    fund_returns = compute_returns(fund_prices)
    profiles = fetch_fund_profiles(fund_tickers)
    available_funds = [ticker for ticker in fund_tickers if ticker in fund_returns.columns]
    if not available_funds:
        st.warning("No overlapping price history returned for the selected mutual funds.")
        return

    latest_prices = latest_available(fund_prices[available_funds]) if not fund_prices.empty else pd.Series(dtype=float)
    loaded = len(available_funds)
    _render_desk_grid(
        [
            ("Funds Loaded", str(loaded), f"of {len(fund_tickers)} requested"),
            ("Primary Benchmark", selected_benchmark, "comparison base"),
            ("History Window", horizon, f"since {horizon_map[horizon]}"),
            ("House Mode", context["house_model"].mode, "current benchmark engine"),
        ]
    )

    research_tabs = st.tabs(["Research", "Compare", "Builder"])

    with research_tabs[0]:
        _render_section_title("Fund Profiles")
        if profiles.empty:
            st.info("Fund metadata is unavailable right now.")
        else:
            display = profiles.copy()
            if not latest_prices.empty:
                display["latest_price"] = display["ticker"].map(latest_prices.to_dict())
            summary_display = display[
                [
                    "ticker",
                    "name",
                    "category",
                    "family",
                    "expense_ratio",
                    "yield_pct",
                    "total_assets",
                    "quote_type",
                    "latest_price",
                ]
            ].rename(
                columns={
                    "ticker": "Ticker",
                    "name": "Fund",
                    "category": "Category",
                    "family": "Family",
                    "expense_ratio": "Expense Ratio",
                    "yield_pct": "Yield",
                    "total_assets": "Total Assets",
                    "quote_type": "Quote Type",
                    "latest_price": "Latest Price",
                }
            )
            st.dataframe(
                summary_display.style.format(
                    {
                        "Expense Ratio": _format_pct,
                        "Yield": _format_pct,
                        "Total Assets": _format_billions,
                        "Latest Price": "{:.2f}",
                    }
                ),
                use_container_width=True,
            )

            selected_profile = st.selectbox("Fund profile detail", available_funds, index=0, key="fund_profile_detail")
            row = profiles.set_index("ticker").loc[selected_profile]
            st.markdown(f"**{row['name']}**")
            st.caption(f"{row['family']} | {row['category']} | {row['quote_type']}")
            st.write(row["summary"])

        _render_section_title("Historical Summary Stats")
        stats_frame = pd.DataFrame({ticker: summary_stats(fund_returns[ticker].dropna()) for ticker in available_funds})
        if "SPY" in fund_returns:
            stats_frame["SPY"] = summary_stats(fund_returns["SPY"].dropna())
        st.dataframe(_format_stats_frame(stats_frame), use_container_width=True)

    with research_tabs[1]:
        _render_section_title("Normalized Growth")
        growth = (1 + fund_returns[available_funds + (["SPY"] if "SPY" in fund_returns else [])]).cumprod()
        st.line_chart(growth)

        _render_section_title("Return Comparison")
        compare_table = pd.DataFrame(index=available_funds)
        compare_table["1M"] = [fund_returns[ticker].tail(21).add(1).prod() - 1 for ticker in available_funds]
        compare_table["3M"] = [fund_returns[ticker].tail(63).add(1).prod() - 1 for ticker in available_funds]
        compare_table["1Y"] = [fund_returns[ticker].tail(252).add(1).prod() - 1 for ticker in available_funds]
        compare_table["Sharpe"] = [summary_stats(fund_returns[ticker]).get("Sharpe") for ticker in available_funds]
        compare_table["Max Drawdown"] = [summary_stats(fund_returns[ticker]).get("Max Drawdown") for ticker in available_funds]
        st.dataframe(
            compare_table.style.format(
                {"1M": _format_pct, "3M": _format_pct, "1Y": _format_pct, "Sharpe": _format_float, "Max Drawdown": _format_pct}
            ),
            use_container_width=True,
        )

        corr = fund_returns[available_funds].corr() if len(available_funds) > 1 else pd.DataFrame()
        compare_cols = st.columns([1.1, 0.9])
        with compare_cols[0]:
            _render_section_title("Correlation")
            if corr.empty:
                st.info("Add at least two funds to compare diversification.")
            else:
                st.dataframe(corr.style.format("{:.2f}"), use_container_width=True)
        with compare_cols[1]:
            benchmark_series = (
                context["house_model"].target_vol_series
                if selected_benchmark == "Market Beating Portfolio"
                else fund_returns.get(selected_benchmark, pd.Series(dtype=float))
            )
            if selected_benchmark != "Market Beating Portfolio" and selected_benchmark not in fund_returns:
                benchmark_series = pd.Series(dtype=float)
            _render_section_title(f"Diagnostics vs {selected_benchmark}")
            if benchmark_series.empty:
                st.info("Not enough overlapping history for benchmark diagnostics.")
            else:
                diag_frames = []
                for ticker in available_funds:
                    diag = compare_stats(fund_returns[ticker], benchmark_series)
                    if diag.empty:
                        continue
                    metric_map = dict(zip(diag["Metric"], diag["Value"]))
                    metric_map["Ticker"] = ticker
                    diag_frames.append(metric_map)
                if diag_frames:
                    diag_frame = pd.DataFrame(diag_frames).set_index("Ticker")
                    st.dataframe(
                        diag_frame.style.format(
                            {
                                "Correlation": "{:.2f}",
                                "Beta": "{:.2f}",
                                "Tracking Error": _format_pct,
                                "Cumulative Return Spread": _format_pct,
                            }
                        ),
                        use_container_width=True,
                    )
                else:
                    st.info("Not enough overlapping history for diagnostics.")

    with research_tabs[2]:
        _render_section_title("Model Mutual Fund Portfolio")
        builder_default = available_funds[: min(4, len(available_funds))]
        selected_builder = st.multiselect(
            "Funds to include",
            available_funds,
            default=builder_default,
            key="morningstar_builder_funds",
        )
        if not selected_builder:
            st.info("Select at least one fund to build a model portfolio.")
        else:
            weighting_mode = st.selectbox(
                "Portfolio construction",
                ["Equal Weight", "Inverse Vol", "Custom Weights"],
                key="morningstar_weight_mode",
            )
            if weighting_mode == "Equal Weight":
                weights = equal_weight_portfolio(selected_builder)
            elif weighting_mode == "Inverse Vol":
                weights = inverse_vol_weights(fund_returns, selected_builder)
            else:
                custom = {}
                cols = st.columns(min(3, len(selected_builder)) or 1)
                default_weight = round(1 / len(selected_builder), 2)
                for idx, ticker in enumerate(selected_builder):
                    label = f"{ticker} weight"
                    custom[ticker] = cols[idx % len(cols)].number_input(
                        label,
                        min_value=0.0,
                        max_value=1.0,
                        value=default_weight,
                        step=0.01,
                        key=f"morningstar_weight_{ticker}",
                    )
                weights = normalize_portfolio_frame(pd.DataFrame({"ticker": list(custom.keys()), "weight": list(custom.values())}))

            portfolio_series = portfolio_returns(weights, fund_returns)
            spy = fund_returns.get("SPY", pd.Series(dtype=float))
            house_series, house_stats = _house_benchmark_series_and_stats(context)
            house_slice = house_series.loc[portfolio_series.index.min(): portfolio_series.index.max()] if not portfolio_series.empty else pd.Series(dtype=float)

            if portfolio_series.empty:
                st.warning("No overlapping return history for the selected fund portfolio.")
            else:
                builder_stats = pd.DataFrame(
                    {
                        "Fund Portfolio": summary_stats(portfolio_series),
                        "SPY": summary_stats(spy),
                        "Market Beating Portfolio": summary_stats(house_slice) if not house_slice.empty else house_stats,
                    }
                )
                st.dataframe(_format_stats_frame(builder_stats), use_container_width=True)

                _render_section_title("Portfolio Growth")
                growth = pd.concat(
                    [
                        (1 + portfolio_series).cumprod().rename("Fund Portfolio"),
                        (1 + spy).cumprod().rename("SPY"),
                        (1 + house_slice).cumprod().rename("Market Beating Portfolio"),
                    ],
                    axis=1,
                ).dropna(how="all")
                st.line_chart(growth)

                lower_cols = st.columns([1.0, 1.0])
                with lower_cols[0]:
                    _render_section_title("Fund Portfolio Weights")
                    st.dataframe(weights.style.format({"weight": _format_pct}), use_container_width=True)
                with lower_cols[1]:
                    _render_section_title("Stress Windows vs SPY")
                    stress = compute_stress_table(portfolio_series, spy)
                    if stress.empty:
                        st.info("Not enough overlapping history for stress windows.")
                    else:
                        st.dataframe(
                            stress.style.format({"Portfolio Return": _format_pct, "SPY Return": _format_pct}),
                            use_container_width=True,
                        )


def _render_screener(context: Dict[str, object]) -> None:
    st.subheader("Screener")
    screener = context["screener"]
    if screener.empty:
        st.warning("No screener output available.")
        return
    display = screener.copy()
    display["Ticker"] = display.index
    display["Proxy"] = [universe_by_ticker().get(ticker).proxy_description for ticker in display.index]
    display["Sleeve"] = [universe_by_ticker().get(ticker).sleeve for ticker in display.index]
    display["Asset Class"] = [universe_by_ticker().get(ticker).asset_class for ticker in display.index]
    ordered = [
        "Ticker",
        "Proxy",
        "Sleeve",
        "Asset Class",
        "1M",
        "3M",
        "6M",
        "12M",
        "Absolute Momentum",
        "Relative Momentum",
        "Trend",
        "Vol-Adjusted Strength",
        "Tactical Score",
        "Structural Score",
        "Macro Score",
        "Composite Score",
        "Composite Percentile",
        "Stance",
    ]
    st.dataframe(
        display[ordered].style.format(
            {
                "1M": _format_pct,
                "3M": _format_pct,
                "6M": _format_pct,
                "12M": _format_pct,
                "Absolute Momentum": _format_pct,
                "Relative Momentum": "{:.2f}",
                "Trend": "{:.2f}",
                "Vol-Adjusted Strength": "{:.2f}",
                "Tactical Score": "{:.2f}",
                "Structural Score": "{:.2f}",
                "Macro Score": "{:.2f}",
                "Composite Score": "{:.2f}",
                "Composite Percentile": "{:.0%}",
            }
        ),
        use_container_width=True,
        height=560,
    )
    st.caption("Composite ranks combine tactical, structural, and macro blocks. Tactical inputs still carry the largest weight in this build.")
    _render_section_title("Cross-Section Dispersion")
    dispersion = pd.DataFrame(
        {
            "Metric": ["1M Return Dispersion", "3M Return Dispersion", "Composite Score Dispersion"],
            "Value": [display["1M"].std(), display["3M"].std(), display["Composite Score"].std()],
        }
    )
    st.dataframe(dispersion.style.format({"Value": _format_pct}), use_container_width=True)


def _render_portfolio_lab(context: Dict[str, object]) -> None:
    st.subheader("Portfolio Lab")
    st.write("Stress-aware portfolio construction and benchmark design relative to SPY risk.")
    available_tickers = tickers()
    selected = st.multiselect(
        "Select sleeves",
        available_tickers,
        default=["SPY", "TLT", "PDBC", "GLDM", "VWO"],
        format_func=lambda ticker: f"{ticker} - {universe_by_ticker()[ticker].proxy_description}",
    )
    if not selected:
        st.info("Select at least one ticker to build a test portfolio.")
        return

    weighting_mode = st.selectbox("Construction mode", ["Equal Weight", "Inverse Vol", "Stress-Aware Search", "Custom Weights"])
    if weighting_mode == "Equal Weight":
        weights = equal_weight_portfolio(selected)
    elif weighting_mode == "Inverse Vol":
        weights = inverse_vol_weights(context["returns"], selected)
    elif weighting_mode == "Stress-Aware Search":
        weights = random_search_optimize(context["returns"], selected)
    else:
        custom = {}
        cols = st.columns(min(4, len(selected)) or 1)
        for idx, ticker in enumerate(selected):
            label = f"{ticker} ({universe_by_ticker()[ticker].proxy_description}) weight"
            custom[ticker] = cols[idx % len(cols)].number_input(label, min_value=0.0, max_value=1.0, value=round(1 / len(selected), 2), step=0.01, key=f"weight_{ticker}")
        weight_frame = pd.DataFrame({"ticker": list(custom.keys()), "weight": list(custom.values())})
        weights = normalize_portfolio_frame(weight_frame)

    portfolio_series = portfolio_returns(weights, context["returns"])
    spy_series = context["returns"].get("SPY", pd.Series(dtype=float))
    house_model = context["house_model"]
    house_series, house_stats = _house_benchmark_series_and_stats(context)
    if portfolio_series.empty:
        st.warning("Selected sleeves do not have enough return history yet.")
        return

    _render_desk_grid(
        [
            ("Selected Sleeves", str(len(selected)), "portfolio sandbox"),
            ("House Leverage", _format_float(house_model.vol_target_leverage), "SPY vol target"),
            ("House Sharpe", _format_float(house_stats.get("Sharpe")), "vol-targeted benchmark"),
            ("House CAGR", _format_pct(house_stats.get("CAGR")), "vol-targeted benchmark"),
        ]
    )

    col1, col2 = st.columns([1.3, 1.0])
    with col1:
        stats = pd.DataFrame({"Portfolio": summary_stats(portfolio_series), "SPY": summary_stats(spy_series)})
        _render_section_title("Summary Stats")
        st.dataframe(_format_stats_frame(stats), use_container_width=True)
    with col2:
        realized_vol = summary_stats(portfolio_series)["Volatility"]
        spy_vol = summary_stats(spy_series)["Volatility"]
        leverage = leverage_to_match_spy_vol(portfolio_series, spy_series)
        _render_section_title("Risk Targeting Snapshot")
        st.metric("Portfolio Vol", _format_pct(realized_vol))
        st.metric("SPY Vol", _format_pct(spy_vol))
        st.metric("Vol Match Leverage", _format_float(leverage))
        st.caption("Leverage is shown as the multiplier required to match trailing SPY volatility.")

    _render_section_title("Rolling 3-Month Return")
    roll = rolling_total_return(portfolio_series, 63).rename("Portfolio").to_frame()
    if not roll.empty:
        st.line_chart(roll)

    lab_cols = st.columns([1.15, 0.85])
    with lab_cols[0]:
        _render_section_title("Stress Windows")
        stress = compute_stress_table(portfolio_series, spy_series)
        st.dataframe(stress.style.format({"Portfolio Return": _format_pct, "SPY Return": _format_pct}), use_container_width=True)
    with lab_cols[1]:
        _render_section_title("House Benchmark vs SPY")
        house_vs_spy = pd.DataFrame(
            {"Market Beating Portfolio": house_stats, "SPY": summary_stats(spy_series)}
        )
        st.dataframe(_format_stats_frame(house_vs_spy), use_container_width=True)

    factor_attr = compute_factor_attribution(portfolio_series, context["returns"])
    house_factor_attr = compute_factor_attribution(house_series, context["returns"])
    factor_cols = st.columns(2)
    with factor_cols[0]:
        _render_factor_block("Portfolio Factor Attribution", factor_attr)
    with factor_cols[1]:
        _render_factor_block("House Benchmark Factor Attribution", house_factor_attr)

    _render_section_title("Current Weights")
    weight_display = weights.copy()
    weight_display["Proxy"] = weight_display["ticker"].map(lambda x: universe_by_ticker()[x].proxy_description if x in universe_by_ticker() else "unresolved")
    st.dataframe(weight_display, use_container_width=True)

    corr_current, corr_delta = correlation_snapshot(context["returns"], selected)
    corr_cols = st.columns(2)
    with corr_cols[0]:
        _render_section_title("Current Correlation")
        if corr_current.empty:
            st.info("Need at least two sleeves for correlation.")
        else:
            st.dataframe(corr_current.style.format("{:.2f}"), use_container_width=True)
    with corr_cols[1]:
        _render_section_title("Correlation Change")
        if corr_delta.empty:
            st.info("Need more history for correlation change.")
        else:
            st.dataframe(corr_delta.style.format("{:.2f}"), use_container_width=True)

    if len(selected) >= 2:
        pair_series = rolling_corr_series(context["returns"], selected[0], selected[1])
        _render_section_title(f"Rolling Correlation: {selected[0]} vs {selected[1]}")
        if pair_series.empty:
            st.info("Not enough overlapping history.")
        else:
            st.line_chart(pair_series.to_frame(name=f"{selected[0]} / {selected[1]}"))

    _render_section_title("Market Beating Portfolio Holdings")
    st.dataframe(
        _house_holdings_frame(context).style.format(
            {
                "strategic_weight": _format_pct,
                "composite_percentile": _format_pct,
                "tilt_vs_strategic": _format_pct,
                "weight": _format_pct,
            }
        ),
        use_container_width=True,
    )


def _render_compare(context: Dict[str, object]) -> None:
    st.subheader("Compare")
    st.write("Upload a portfolio with `ticker` and `weight` columns.")
    template_csv = "ticker,weight\nSPY,0.60\nBND,0.40\n"
    st.download_button(
        "Get a Template",
        data=template_csv,
        file_name="portfolio_template.csv",
        mime="text/csv",
        help="Download a sample CSV with the required column headers.",
        use_container_width=False,
    )
    uploaded = st.file_uploader("Portfolio CSV", type=["csv"])

    if uploaded is None:
        st.info("Upload a CSV to compare it against SPY and the house framework.")
        return

    if "vanguard_benchmark_config" not in st.session_state:
        _configure_vanguard_dialog()
        return

    vg_config = st.session_state["vanguard_benchmark_config"]
    if st.button("Reconfigure Vanguard Benchmark", key="reconfigure_vanguard"):
        _configure_vanguard_dialog()
        return

    benchmark_prices = fetch_vanguard_benchmark_history(vg_config["ticker"], vg_config["start"], vg_config["end"])
    benchmark_returns = benchmark_prices.pct_change().dropna() if not benchmark_prices.empty else pd.Series(dtype=float)
    benchmark_state = (
        degraded_vanguard_state()
        if benchmark_prices.empty
        else degraded_vanguard_state(float(benchmark_prices.iloc[-1]))
    )
    status_note = (
        f"{vg_config['label']} ({vg_config['ticker']}) from {vg_config['start']} to {vg_config['end']}"
        if not benchmark_prices.empty
        else benchmark_state.note
    )
    st.caption(f"{VANGUARD_TARGET_RETIREMENT_LABEL}: {status_note}")

    frame = pd.read_csv(uploaded)
    try:
        weights = normalize_portfolio_frame(frame)
    except ValueError as exc:
        st.error(str(exc))
        return

    compare_start = vg_config["start"]
    compare_end = vg_config["end"]

    sliced_returns = context["returns"].loc[compare_start:compare_end]
    port = portfolio_returns(weights, sliced_returns)
    spy = sliced_returns.get("SPY", pd.Series(dtype=float))
    house_model = context["house_model"]
    full_house_series, full_house_stats = _house_benchmark_series_and_stats(context)
    house = full_house_series.loc[compare_start:compare_end]
    if port.empty:
        st.warning("No overlapping market data for the uploaded portfolio.")
        return

    _render_desk_grid(
        [
            ("Input Holdings", str(len(weights)), "uploaded"),
            ("House Holdings", str(len(house_model.holdings)), "benchmark"),
            ("Vanguard Fund", vg_config["ticker"], vg_config["label"]),
            ("Window", f"{compare_start} to {compare_end}", "comparison range"),
        ]
    )

    vanguard_stats = summary_stats(benchmark_returns) if not benchmark_returns.empty else {}
    stats = pd.DataFrame(
        {
            "Input Portfolio": summary_stats(port),
            "SPY": summary_stats(spy),
            "Market Beating Portfolio": summary_stats(house) if not house.empty else full_house_stats,
            vg_config["ticker"]: vanguard_stats,
        }
    )
    st.dataframe(_format_stats_frame(stats), use_container_width=True)

    exposure = classify_holdings(weights)
    house_exposure = classify_holdings(house_model.holdings[["ticker", "weight"]])
    spy_exposure = classify_holdings(pd.DataFrame({"ticker": ["SPY"], "weight": [1.0]}))
    compare_tabs = st.tabs(["Input Portfolio", "House vs SPY", "Benchmark Diagnostics"])
    with compare_tabs[0]:
        _render_section_title("Exposure Mapping")
        st.dataframe(exposure, use_container_width=True)

        summaries = summarize_exposures(exposure)
        exp_cols = st.columns(3)
        for idx, key in enumerate(["Asset Class", "Region", "Style"]):
            with exp_cols[idx]:
                st.markdown(f"**{key} Summary**")
                st.dataframe(summaries[key], use_container_width=True)

    with compare_tabs[1]:
        _render_section_title("Market Beating Portfolio Holdings")
        st.dataframe(
            _house_holdings_frame(context).style.format(
                {
                    "strategic_weight": _format_pct,
                    "composite_percentile": _format_pct,
                    "tilt_vs_strategic": _format_pct,
                    "weight": _format_pct,
                }
            ),
            use_container_width=True,
        )
        diff_cols = st.columns(2)
        for idx, key in enumerate(["Asset Class", "Style"]):
            with diff_cols[idx]:
                st.markdown(f"**Market Beating Portfolio minus SPY: {key}**")
                diff = compare_exposure_summary(house_exposure, spy_exposure, key)
                st.dataframe(diff.style.format({"Left": _format_pct, "Right": _format_pct, "Difference": _format_pct}), use_container_width=True)

        house_spy_diag = compare_stats(house, spy)
        st.markdown("**Market Beating Portfolio vs SPY Diagnostics**")
        st.dataframe(house_spy_diag.style.format({"Value": "{:.2f}"}), use_container_width=True)

        if not benchmark_returns.empty:
            vg_diag = compare_stats(house, benchmark_returns)
            st.markdown(f"**Market Beating Portfolio vs {vg_config['ticker']} Diagnostics**")
            st.dataframe(vg_diag.style.format({"Value": "{:.2f}"}), use_container_width=True)

    with compare_tabs[2]:
        _render_section_title("Benchmark-Relative Diagnostics")
        diag_spy = compare_stats(port, spy)
        diag_house = compare_stats(port, house)
        diag_vanguard = compare_stats(port, benchmark_returns) if not benchmark_returns.empty else pd.DataFrame()
        dcols = st.columns(3)
        with dcols[0]:
            st.markdown("`Input vs SPY`")
            if diag_spy.empty:
                st.info("Not enough overlapping history.")
            else:
                st.dataframe(diag_spy.style.format({"Value": "{:.2f}"}), use_container_width=True)
        with dcols[1]:
            st.markdown("`Input vs Market Beating Portfolio`")
            if diag_house.empty:
                st.info("Not enough overlapping history.")
            else:
                st.dataframe(diag_house.style.format({"Value": "{:.2f}"}), use_container_width=True)
        with dcols[2]:
            st.markdown(f"`Input vs {vg_config['ticker']}`")
            if diag_vanguard.empty:
                st.info("Not enough overlapping history.")
            else:
                st.dataframe(diag_vanguard.style.format({"Value": "{:.2f}"}), use_container_width=True)

        factor_cols = st.columns(3)
        with factor_cols[0]:
            _render_factor_block("Input Factor Attribution", compute_factor_attribution(port, sliced_returns))
        with factor_cols[1]:
            _render_factor_block("House Factor Attribution", compute_factor_attribution(house, sliced_returns))
        with factor_cols[2]:
            _render_factor_block("SPY Factor Attribution", compute_factor_attribution(spy, sliced_returns))

    regime = context["regime"]
    st.info(
        f"Current regime: {regime.regime}. The uploaded portfolio is being compared to SPY. "
        "Unresolved holdings are surfaced explicitly so the comparison does not overstate classification confidence."
    )


def _render_macro(context: Dict[str, object]) -> None:
    st.subheader("Macro")
    regime = context["regime"]
    fred_status = context["fred_status"]
    house_series, house_stats = _house_benchmark_series_and_stats(context)
    implications = regime_implications(regime.regime)
    snapshots = context["macro_snapshot"]
    _render_desk_grid(
        [
            ("Regime", regime.regime, "current"),
            ("Confidence", regime.confidence, "classification"),
            ("Macro Feed", "Live" if fred_status.ok else "Degraded", fred_status.source),
            ("House Leverage", _format_float(context["house_model"].vol_target_leverage), "SPY-targeted"),
            ("House CAGR", _format_pct(house_stats.get("CAGR")), "vol-targeted benchmark"),
        ]
    )
    if fred_status.ok:
        st.caption(fred_status.message)
    else:
        st.warning(fred_status.message)
    st.write(regime.summary)

    snapshot_cols = st.columns(3)
    snapshot_items = list(snapshots.items())
    for idx, (_, snap) in enumerate(snapshot_items[:6]):
        with snapshot_cols[idx % 3]:
            st.metric(
                snap.label,
                _format_float(snap.latest),
                delta=_format_float(snap.trailing_change),
            )
            st.caption(f"Direction: {snap.direction}")

    macro_tabs = st.tabs(["Regime", "Indicators", "Portfolio Impact"])
    with macro_tabs[0]:
        _render_section_title("Regime Narrative")
        for line in implications["narrative"]:
            st.write(f"- {line}")
        favored = ", ".join(implications["favored"]) if implications["favored"] else "none"
        challenged = ", ".join(implications["challenged"]) if implications["challenged"] else "none"
        st.info(f"Favored sleeves: {favored}. Challenged sleeves: {challenged}.")

    with macro_tabs[1]:
        _render_section_title("Macro Indicators")
        fred_bundle = context["fred_bundle"]
        for series in DEFAULT_FRED_SERIES:
            values = fred_bundle.get(series.series_id, pd.Series(dtype=float))
            with st.expander(series.label, expanded=False):
                if values.empty:
                    st.warning("Series unavailable.")
                else:
                    st.line_chart(values.to_frame(name=series.label))

    with macro_tabs[2]:
        _render_section_title("Portfolio Impact")
        impact = pd.DataFrame(
            {
                "Market Beating Portfolio": house_stats,
                "SPY": summary_stats(context["returns"].get("SPY", pd.Series(dtype=float))),
            }
        )
        st.dataframe(_format_stats_frame(impact), use_container_width=True)
        attr = compute_factor_attribution(house_series, context["returns"])
        _render_factor_block("House Benchmark Factor Attribution", attr)
        if context["house_model"].diagnostics:
            _render_section_title("What This Means For The House Benchmark")
            for note in context["house_model"].diagnostics:
                st.write(f"- {note}")


def _render_learn(context: Dict[str, object]) -> None:
    st.subheader("Learn")
    regime = context["regime"]
    house_series, house_stats = _house_benchmark_series_and_stats(context)
    spy_stats = summary_stats(context["returns"].get("SPY", pd.Series(dtype=float)))
    implications = regime_implications(regime.regime)

    learn_tabs = st.tabs(["Framework", "Why It May Be Failing", "What To Learn Next"])
    with learn_tabs[0]:
        _render_section_title("Framework")
        st.write(
            "The thesis is not 'diversification alone wins.' The thesis is that diversified sleeves with decent standalone Sharpe, "
            "weak enough correlation, and positive expected return can be levered to equity-like risk and outperform broad equity beta over long horizons."
        )
        st.write(
            "That means the key object is the vol-targeted benchmark, not the unlevered mix. If the unlevered mix does not earn a better Sharpe than SPY, "
            "adding leverage just scales disappointment."
        )
        st.dataframe(
            pd.DataFrame({"Market Beating Portfolio": house_stats, "SPY": spy_stats}),
            use_container_width=True,
        )
        st.write(f"The active house benchmark mode is `{context['house_model'].mode}`.")

    with learn_tabs[1]:
        _render_section_title("Why It May Be Failing Right Now")
        reasons = [
            "The strategic core may still be too equity-adjacent, so diversification is weaker than it appears.",
            "The current structural inputs are still rough proxies; value, carry, and expected-return signals are not yet research-grade.",
            "The tactical overlay may not be adding enough to justify leverage.",
            "Managed futures, commodities, and long-duration sleeves can lag for long stretches even if they help in specific regimes.",
            "A high-beta equity regime can make SPY look dominant over a given sample even when the long-run thesis is still alive.",
        ]
        for item in reasons:
            st.write(f"- {item}")
        st.info(
            f"Current regime is `{regime.regime}`. Favored sleeves here: {', '.join(implications['favored']) if implications['favored'] else 'none'}."
        )
        for note in context["house_model"].diagnostics:
            st.write(f"- {note}")

    with learn_tabs[2]:
        _render_section_title("What To Learn Next")
        lessons = [
            "Does the house benchmark beat SPY on Sharpe before leverage, not after?",
            "Which sleeves actually diversify in selloffs rather than only in normal markets?",
            "Are factor tilts adding expected return, or just complexity?",
            "Does regime-aware overlay improve drawdown resilience enough to matter?",
            "Would a more disciplined strategic core plus smaller tactical deviations outperform the current design?",
        ]
        for lesson in lessons:
            st.write(f"- {lesson}")
        st.write(
            "The right interpretation of a weak current chart is not 'the idea is dead.' It is: the current implementation has not yet earned the right to be scaled."
        )


def main() -> None:
    st.set_page_config(page_title="codex_terminal", layout="wide")
    _inject_terminal_theme()
    context = _load_market_context()
    page = _render_sidebar()
    _render_header(context)

    if page == "Welcome":
        _render_welcome(context)
    elif page == "Morning Brief":
        _render_morning_brief(context)
    elif page == "Terminal":
        _render_terminal(context)
    elif page == "Screener":
        _render_screener(context)
    elif page == "Portfolio Lab":
        _render_portfolio_lab(context)
    elif page == "Compare":
        _render_compare(context)
    elif page == "Morningstar":
        _render_morningstar(context)
    elif page == "Macro":
        _render_macro(context)
    else:
        _render_learn(context)
