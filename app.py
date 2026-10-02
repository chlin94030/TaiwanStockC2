"""
Taiwan Alpha Radar V13.2 - mobile-first plain-language Streamlit research dashboard.
Run: streamlit run app.py
"""
from __future__ import annotations

from pathlib import Path
import gc
import html
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import radar_service as service
from market_data import DailyPriceStore, _taipei_timestamp, provider_runtime_status
from trading_calendar import calendar_reference, daily_freshness, entry_review_allowed
from presentation import investment_view
from policy_engine import HORIZONS
from return_first_model import holding_review

ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("ALPHA_RADAR_DATA_DIR", str(ROOT / "data")))

VIEW_LABELS = ["⭐ 極選", "⚡ 短線", "📈 中線", "🧭 長波段", "🔎 個股診斷"]
HORIZON_LABELS = {"short": "短線｜約2週", "mid": "中線｜約2個月", "long": "長波段｜約半年"}
FAMILY_LABELS = {
    "完整模式｜價格＋公司表現＋法人": "full",
    "價格模式｜只看走勢與歷史案例": "price_only",
    "公司模式｜再看營收／EPS／毛利率": "business_confirmed",
    "資金模式｜再看近月三大法人": "flow_confirmed",
}
HOLD_LABELS = {
    "DATA_UNVERIFIED": "價格資料未核對，暫不作留／賣判斷。",
    "ORIGINAL_STRUCTURE_INVALIDATED": "⚠️ 已跌破原始防守價，原交易結構失效。",
    "ORIGINAL_THESIS_INVALIDATED": "⚠️ 原買進理由已失效，應重新檢視部位。",
    "PROTECTION_TRIGGER_REVIEW_EXECUTION": "🔔 已觸及移動保護價，請依原定紀律檢視執行。",
    "ORIGINAL_THESIS_UNKNOWN_MANUAL_REVIEW": "缺少原始停損或保護價，需手動評估。",
    "ORIGINAL_RULES_NOT_BREACHED_NOT_A_RETURN_GUARANTEE": "✅ 尚未破壞原設定的防守結構；不代表未來報酬保證。",
}
REGIME_LABELS = {"BULL": "🟢 大盤偏多", "NEUTRAL": "🟡 大盤震盪", "BEAR": "🔴 大盤偏弱", "UNKNOWN": "⚪ 大盤待確認"}

CSS = """
<style>
:root { --ink:#0f172a; --muted:#64748b; --line:#e2e8f0; --soft:#f8fafc; --blue:#1463ff; --green:#0f9f6e; --red:#e23b3b; --amber:#e07a00; }
.stApp { background:#f7f9fc; color:var(--ink); font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; font-size:16px; }
.block-container { max-width:1040px; padding-top:1rem; padding-bottom:4rem; }
.hero { padding:18px 22px; border-radius:20px; background:#111827; color:white; margin-bottom:14px; box-shadow:0 10px 24px rgba(15,23,42,.12); }
.hero .eyebrow { letter-spacing:.12em; font-size:.78rem; font-weight:850; color:#93c5fd; }
.hero h1 { font-size:1.9rem; font-weight:900; margin:.25rem 0 0; color:white; letter-spacing:-.02em; }
.statusline { display:block; background:white; border:1px solid var(--line); border-radius:12px; padding:10px 13px; margin:4px 0 12px; color:#334155; font-weight:750; line-height:1.45; }
.card { background:white; border:1px solid var(--line); border-radius:20px; padding:22px; margin:16px 0 8px; box-shadow:0 7px 20px rgba(15,23,42,.04); }
.card-head { display:flex; justify-content:space-between; gap:14px; align-items:flex-start; }
.card-title { font-size:1.58rem; font-weight:900; letter-spacing:-.02em; line-height:1.2; }
.card-code { color:var(--muted); margin-top:4px; font-size:.94rem; font-weight:650; }
.card-price { font-size:1.48rem; font-weight:900; text-align:right; white-space:nowrap; }
.badge { display:inline-block; font-size:.88rem; font-weight:800; padding:6px 10px; border-radius:10px; background:#f1f5f9; color:#475569; margin:9px 5px 5px 0; }
.badge-blue { background:#e8f0ff; color:#174ea6; }
.industry-tag { display:inline-block; font-size:.9rem; font-weight:850; padding:7px 11px; border-radius:999px; background:#eef2ff; color:#4338ca; margin:8px 5px 4px 0; }
.verdict { margin:12px 0; padding:14px 16px; border-radius:15px; background:#f8fafc; border:1px solid #e2e8f0; }
.verdict-title { font-size:1.18rem; font-weight:900; color:#0f172a; margin-bottom:6px; }
.reason-label { color:#475569; font-size:.84rem; font-weight:850; margin-top:6px; }
.reason-list { margin:5px 0 0 0; padding-left:1.18rem; color:#334155; line-height:1.55; }
.risk-note { margin-top:8px; padding:8px 10px; border-radius:9px; background:#fff7ed; color:#9a3412; font-size:.9rem; line-height:1.45; }
.return-box { margin:11px 0; padding:14px 17px; border-radius:15px; background:#eff6ff; border:1px solid #bfdbfe; }
.return-k { color:#1e40af; font-weight:850; font-size:.92rem; }
.return-v { font-size:2.15rem; line-height:1.08; font-weight:900; color:#0b63ce; margin:3px 0; }
.return-desc { color:#475569; font-weight:650; font-size:.88rem; line-height:1.4; }
.stats { display:grid; grid-template-columns:repeat(2,1fr); gap:9px; margin:10px 0; }
.stat { background:#fafafa; border:1px solid #edf1f5; border-radius:13px; padding:11px 13px; }
.stat .k { color:var(--muted); font-size:.83rem; font-weight:750; line-height:1.3; }
.stat .v { color:var(--ink); font-size:1.16rem; font-weight:900; margin-top:4px; }
.research-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:9px; margin:12px 0 3px; }
.research { background:#fff; border:1px solid var(--line); border-radius:13px; padding:11px 12px; }
.research .k { color:#64748b; font-size:.82rem; font-weight:800; }
.research .v { color:#0f172a; font-size:1.06rem; font-weight:900; margin-top:4px; line-height:1.3; }
.research .s { color:#64748b; font-size:.77rem; margin-top:3px; line-height:1.3; }
.decision { padding:12px 14px; border-radius:13px; background:#fff7ed; border:1px solid #fed7aa; color:#9a3412; font-weight:850; margin-top:11px; }
.decision-ok { background:#ecfdf5; border-color:#a7f3d0; color:#065f46; }
.levels { display:grid; grid-template-columns:repeat(4,1fr); gap:9px; margin:10px 0 12px; }
.level { background:#f8fafc; border:1px solid var(--line); border-radius:11px; padding:11px; }
.level .k { color:#64748b; font-size:.81rem; font-weight:750; }
.level .v { color:#0f172a; font-size:1.04rem; font-weight:900; margin-top:3px; }
.small-note { color:#64748b; font-size:.84rem; line-height:1.45; }
[data-testid="stExpander"] details summary p { font-weight:750; }
@media(max-width:760px){
 .stApp{font-size:17px}.block-container{padding:.65rem .58rem 3.2rem}.hero{padding:14px 14px;border-radius:15px;margin-bottom:10px}
 .hero .eyebrow{font-size:.72rem}.hero h1{font-size:1.52rem}.statusline{font-size:.94rem;padding:10px 11px}
 .card{padding:16px 12px;border-radius:16px;margin:12px 0 7px}.card-title{font-size:1.5rem}.card-code{font-size:.98rem}.card-price{font-size:1.34rem}
 .industry-tag,.badge{font-size:.92rem}.verdict{padding:13px 12px}.verdict-title{font-size:1.22rem}.reason-list{font-size:1rem;line-height:1.5}
 .return-v{font-size:2rem}.return-desc{font-size:.9rem}.stat .k,.research .k{font-size:.9rem}.stat .v,.research .v{font-size:1.16rem}.research .s{font-size:.84rem}
 .research-grid{grid-template-columns:repeat(2,1fr)}.levels{grid-template-columns:repeat(2,1fr)}.decision{font-size:1rem;line-height:1.4}
 .stButton>button{min-height:46px;font-size:1.03rem}.stRadio label{font-size:1rem!important}
}
</style>
"""


def esc(v):
    return html.escape(str(v if v is not None else ""))


def finite(v):
    try:
        x = float(v)
        return x if np.isfinite(x) else None
    except Exception:
        return None


def pct_decimal(v, signed=True):
    x = finite(v)
    if x is None:
        return "—"
    return f"{x*100:+.2f}%" if signed else f"{x*100:.2f}%"


def pct_point(v, signed=True):
    x = finite(v)
    if x is None:
        return "—"
    return f"{x:+.1f}%" if signed else f"{x:.1f}%"


def num(v, digits=2):
    x = finite(v)
    return "—" if x is None else f"{x:,.{digits}f}".rstrip("0").rstrip(".")


def signed_lots(v):
    x = finite(v)
    return "—" if x is None else f"{x:+,.0f} 張"


def _latest_research(obj):
    r = obj.get("research") or {}
    rev = r.get("monthly_revenue") or {}
    fin = r.get("financials") or {}
    val = r.get("valuation") or {}
    inst = r.get("institutional_flow") or {}
    main = r.get("main_force_proxy") or {}
    return r, rev, fin, val, inst, main


def render_chart(chart, plan, key):
    if not chart or "ohlcv" not in chart:
        st.caption("K 線資料未取得。")
        return

    raw = pd.DataFrame(chart["ohlcv"], columns=["Open", "High", "Low", "Close", "Volume"])
    raw["Date"] = pd.to_datetime(chart.get("dates", []), errors="coerce")
    raw = raw.dropna(subset=["Date"]).reset_index(drop=True)
    if raw.empty:
        return

    # Indicators are calculated on the full series first, then the visible
    # window is sliced. This avoids distorted moving averages at the left edge.
    close, high, low = raw["Close"], raw["High"], raw["Low"]
    for n in [5, 20, 60, 120]:
        raw[f"MA{n}"] = close.rolling(n).mean()
    std20 = close.rolling(20).std()
    raw["BBU"] = raw["MA20"] + 2 * std20
    raw["BBL"] = raw["MA20"] - 2 * std20
    low9, high9 = low.rolling(9).min(), high.rolling(9).max()
    rsv = (close - low9) / (high9 - low9).replace(0, np.nan) * 100
    raw["K"] = rsv.ewm(com=2, adjust=False).mean()
    raw["D"] = raw["K"].ewm(com=2, adjust=False).mean()
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    raw["RSI"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    ema12, ema26 = close.ewm(span=12, adjust=False).mean(), close.ewm(span=26, adjust=False).mean()
    raw["MACD"] = ema12 - ema26
    raw["SIGNAL"] = raw["MACD"].ewm(span=9, adjust=False).mean()
    raw["HIST"] = raw["MACD"] - raw["SIGNAL"]

    range_label = st.radio(
        "K線區間", ["3個月", "6個月", "1年"], index=0, horizontal=True,
        label_visibility="collapsed", key=f"{key}_range",
    )
    sessions = {"3個月": 66, "6個月": 132, "1年": 252}[range_label]
    more_lines = st.checkbox("顯示更多線", value=False, key=f"{key}_more")

    df = raw.tail(sessions).copy().reset_index(drop=True)
    x = df["Date"].dt.strftime("%Y-%m-%d").tolist()

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.035,
        row_heights=[0.78, 0.22],
    )
    fig.add_trace(go.Candlestick(
        x=x, open=df.Open, high=df.High, low=df.Low, close=df.Close,
        increasing_line_color="#e53935", increasing_fillcolor="#e53935",
        decreasing_line_color="#00a36c", decreasing_fillcolor="#00a36c",
        whiskerwidth=0.45, name="K線", showlegend=False,
    ), row=1, col=1)

    # Keep the default legend intentionally small: two lines only.
    fig.add_trace(go.Scatter(
        x=x, y=df.MA20, mode="lines", line=dict(color="#0066ff", width=2.3),
        name="20日線", connectgaps=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=x, y=df.MA60, mode="lines", line=dict(color="#ff8a00", width=2.3),
        name="60日線", connectgaps=False,
    ), row=1, col=1)

    if more_lines:
        fig.add_trace(go.Scatter(x=x, y=df.MA5, mode="lines", line=dict(color="#7c3aed", width=1.5), name="5日線", showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=df.MA120, mode="lines", line=dict(color="#111827", width=1.5, dash="dot"), name="120日線", showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=df.BBU, mode="lines", line=dict(color="#94a3b8", width=1, dash="dot"), showlegend=False, hoverinfo="skip"), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=df.BBL, mode="lines", line=dict(color="#94a3b8", width=1, dash="dot"), showlegend=False, hoverinfo="skip"), row=1, col=1)

    if plan:
        zl, zh = finite(plan.get("zone_low")), finite(plan.get("zone_high"))
        trigger, stop = finite(plan.get("trigger")), finite(plan.get("invalidation"))
        if zl is not None and zh is not None:
            fig.add_hrect(y0=zl, y1=zh, fillcolor="rgba(0,102,255,.09)", line_width=0, row=1, col=1)
        if trigger is not None:
            fig.add_hline(y=trigger, line_color="#d97706", line_width=1.7, line_dash="dot", row=1, col=1,
                          annotation_text="轉強", annotation_position="top left", annotation_font_color="#b45309")
        if stop is not None:
            fig.add_hline(y=stop, line_color="#dc2626", line_width=1.7, line_dash="dash", row=1, col=1,
                          annotation_text="防守", annotation_position="bottom left", annotation_font_color="#b91c1c")

    vol_colors = np.where(df.Close >= df.Open, "rgba(229,57,53,.72)", "rgba(0,163,108,.72)")
    fig.add_trace(go.Bar(x=x, y=df.Volume, marker_color=vol_colors, name="成交量", showlegend=False), row=2, col=1)

    # Four date labels are enough on a phone and avoid the compressed barcode look.
    if len(x) > 1:
        idxs = sorted(set(np.linspace(0, len(x) - 1, min(4, len(x)), dtype=int).tolist()))
        tickvals = [x[i] for i in idxs]
        ticktext = [pd.to_datetime(x[i]).strftime("%m/%d") for i in idxs]
    else:
        tickvals, ticktext = x, x

    # Give price extra vertical breathing room so a narrow mobile screen does not
    # visually exaggerate every slope.
    y_values = pd.concat([df["Low"], df["High"], df["MA20"], df["MA60"]]).dropna()
    if plan:
        extras = [finite(plan.get(k)) for k in ["zone_low", "zone_high", "trigger", "invalidation"]]
        extras = [v for v in extras if v is not None]
        if extras:
            y_values = pd.concat([y_values, pd.Series(extras)], ignore_index=True)
    if not y_values.empty:
        ymin, ymax = float(y_values.min()), float(y_values.max())
        span = max(ymax - ymin, max(1.0, float(df.Close.iloc[-1]) * 0.04))
        pad = span * 0.20
        fig.update_yaxes(range=[ymin - pad, ymax + pad], row=1, col=1)

    fig.update_layout(
        height=430,
        margin=dict(l=6, r=6, t=38, b=4),
        template="plotly_white",
        dragmode=False,
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0,
            font=dict(size=12), bgcolor="rgba(255,255,255,.78)",
        ),
        bargap=0.15,
    )
    fig.update_xaxes(type="category", fixedrange=True, showgrid=False, showticklabels=False, row=1, col=1)
    fig.update_xaxes(
        type="category", fixedrange=True, showgrid=False,
        tickmode="array", tickvals=tickvals, ticktext=ticktext,
        tickfont=dict(size=11, color="#64748b"), row=2, col=1,
    )
    fig.update_yaxes(fixedrange=True, gridcolor="#edf2f7", zeroline=False, tickfont=dict(size=11), row=1, col=1)
    fig.update_yaxes(fixedrange=True, showgrid=False, showticklabels=False, zeroline=False, row=2, col=1)
    st.plotly_chart(
        fig, use_container_width=True, key=key,
        config={"displayModeBar": False, "displaylogo": False, "scrollZoom": False, "responsive": True},
    )

    with st.expander("其他走勢指標", expanded=False):
        last = raw.iloc[-1]
        kd_text = "偏強" if finite(last.get("K")) is not None and finite(last.get("D")) is not None and last["K"] >= last["D"] else "偏弱"
        macd_text = "偏多" if finite(last.get("HIST")) is not None and last["HIST"] >= 0 else "偏空"
        rsi = finite(last.get("RSI"))
        rsi_text = "—" if rsi is None else f"{rsi:.0f}"
        c1, c2, c3 = st.columns(3)
        c1.metric("短線強弱", kd_text)
        c2.metric("趨勢動能", macd_text)
        c3.metric("強弱位置", rsi_text)


def _research_html(obj):
    _, rev, fin, val, inst, main = _latest_research(obj)
    latest_rev = rev.get("latest") or {}
    rev_v = "—"
    rev_s = "近一年月營收未取得"
    if latest_rev:
        rev_v = f"{num(latest_rev.get('revenue_billion'))} 億"
        rev_s = f"{latest_rev.get('month','')}｜MoM {pct_point(latest_rev.get('mom_pct'))}｜YoY {pct_point(latest_rev.get('yoy_pct'))}"
    eps_v = f"{num(fin.get('eps_4q_sum'))} 元" if fin.get("eps_4q_sum") is not None else "—"
    eps_s = "近4季單季EPS加總（未校正配股/分割）"
    gm_v = pct_point(fin.get("gross_margin_latest_pct"), signed=False) if fin.get("gross_margin_latest_pct") is not None else "—"
    gm_s = f"近4季平均 {pct_point(fin.get('gross_margin_4q_avg_pct'), signed=False)}" if fin.get("gross_margin_4q_avg_pct") is not None else "毛利率未取得／不適用"
    pe_v = num(val.get("pe")) + " 倍" if val.get("pe") is not None else "—"
    pe_s = f"資料日 {val.get('date','—')}"
    inst_v = signed_lots(inst.get("total_net_lots")) if inst.get("available") else "—"
    inst_s = "近月三大法人合計" if inst.get("available") else "近月法人資料未取得"
    main_v = signed_lots(main.get("proxy_net_lots")) if main.get("available") else "未啟用"
    main_s = (f"近{main.get('sessions',20)}日券商分點代理" if main.get("available") else "需額外券商分點資料權限")
    return f"""
    <div class="research-grid">
      <div class="research"><div class="k">最新月營收</div><div class="v">{esc(rev_v)}</div><div class="s">{esc(rev_s)}</div></div>
      <div class="research"><div class="k">近4季 EPS 合計</div><div class="v">{esc(eps_v)}</div><div class="s">{esc(eps_s)}</div></div>
      <div class="research"><div class="k">毛利率</div><div class="v">{esc(gm_v)}</div><div class="s">{esc(gm_s)}</div></div>
      <div class="research"><div class="k">目前本益比</div><div class="v">{esc(pe_v)}</div><div class="s">{esc(pe_s)}</div></div>
      <div class="research"><div class="k">三大法人近1月</div><div class="v">{esc(inst_v)}</div><div class="s">{esc(inst_s)}</div></div>
      <div class="research"><div class="k">主力近1月（代理）</div><div class="v">{esc(main_v)}</div><div class="s">{esc(main_s)}</div></div>
    </div>"""


def _render_research_details(obj, key):
    _, rev, fin, val, inst, main = _latest_research(obj)
    with st.expander("公司與資金明細", expanded=False):
        if rev.get("rows"):
            st.markdown("**近 12 個月營收**")
            rdf = pd.DataFrame(rev["rows"]).rename(columns={
                "month":"月份", "revenue_billion":"營收(億元)", "mom_pct":"MoM(%)", "yoy_pct":"YoY(%)"
            })
            st.dataframe(rdf, hide_index=True, use_container_width=True)
        else:
            st.caption("未取得近一年月營收資料。")

        if fin.get("quarters"):
            st.markdown("**近一年各季 EPS／累積 EPS／毛利率**")
            qdf = pd.DataFrame(fin["quarters"]).rename(columns={
                "quarter":"季別", "eps":"單季EPS", "eps_ytd_sum_unadjusted":"當年度累積EPS*", "gross_margin_pct":"毛利率(%)"
            })
            cols = [c for c in ["季別", "單季EPS", "當年度累積EPS*", "毛利率(%)"] if c in qdf.columns]
            st.dataframe(qdf[cols], hide_index=True, use_container_width=True)
            st.caption("* FinMind 的 EPS 為單季值；此處累積為單季直接加總。若期間有配股或股票分割，可能與公司財報列示之累積 EPS 不一致。")
        else:
            st.caption("未取得近一年季財報資料，或該業別毛利率不適用。")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**近一月三大法人**")
            if inst.get("available"):
                idf = pd.DataFrame([
                    ["外資", inst.get("foreign_net_lots")],
                    ["投信", inst.get("trust_net_lots")],
                    ["自營商", inst.get("dealer_net_lots")],
                    ["合計", inst.get("total_net_lots")],
                ], columns=["類別", "買賣超(張)"])
                st.dataframe(idf, hide_index=True, use_container_width=True)
                st.caption(inst.get("summary", ""))
            else:
                st.caption("未取得三大法人資料。")
        with c2:
            st.markdown("**近一月主力買賣超代理**")
            if main.get("available"):
                st.metric("券商分點代理淨額", signed_lots(main.get("proxy_net_lots")))
                st.caption(main.get("summary", ""))
                if main.get("top_buy_branches"):
                    bdf = pd.DataFrame(main.get("top_buy_branches") + main.get("top_sell_branches", []))
                    st.dataframe(bdf.rename(columns={"name":"券商分點", "net_lots":"淨買賣超(張)"}), hide_index=True, use_container_width=True)
            else:
                st.caption(main.get("summary", "主力代理未啟用。"))
                st.caption("主力沒有交易所統一正式分類；本系統只在取得券商分點資料時做近似觀察，不會用三大法人代替主力。")

        st.markdown("**評分拆解**")
        # Find the first horizon with the current card key isn't available here; show all three compactly.
        score_rows = []
        for h in HORIZONS:
            b = obj.get("horizons", {}).get(h, {})
            comp = b.get("score_components") or {}
            score_rows.append({
                "週期": HORIZON_LABELS[h],
                "整體": b.get("ranking_score"),
                "走勢": comp.get("technical"),
                "歷史案例": comp.get("empirical"),
                "公司數據": comp.get("fundamental"),
                "法人資金": comp.get("flow"),
            })
        st.dataframe(pd.DataFrame(score_rows), hide_index=True, use_container_width=True)


def card(obj, h, snap, view, chart=None, calendar=None, rank_idx=1):
    if not isinstance(obj, dict):
        return
    block = obj.get("horizons", {}).get(h, {})
    f = block.get("forecast") or {}
    plan = block.get("plan")
    strategy = f.get("strategy") or {}
    condition = block.get("entry_state", "NO_RETURN_ESTIMATE")
    calendar = calendar or calendar_reference(DATA_DIR, now=_taipei_timestamp())
    fresh = daily_freshness(obj.get("price_date", ""), calendar)
    review_ok = entry_review_allowed(obj.get("price_date", ""), plan, calendar)

    empirical_med = strategy.get("median") if f.get("estimate_available") else None
    pos_rate = strategy.get("historical_positive_rate") if f.get("estimate_available") else None
    sample_n = int(f.get("local_effective_n") or 0)
    fine = obj.get("fine_industry") or obj.get("industry") or "產業分類待補"
    view_info = investment_view(obj, h, snap.get("market", {}) if isinstance(snap, dict) else {})

    reason_items = "".join(f"<li>{esc(x)}</li>" for x in view_info.get("reasons", []))
    caution_items = " ".join(esc(x) for x in view_info.get("cautions", []))
    risk_html = f'<div class="risk-note"><b>注意｜</b>{caution_items}</div>' if caution_items else ""

    action_text = {
        "CONDITIONS_MET_NOT_FILLED": "守穩目前區間再考慮。",
        "WAIT_ENTRY_ZONE": "等拉回，不追價。",
        "WAIT_BREAKOUT": "等價格真正轉強。",
        "WAIT_CONFIRMATION": "等價格與買盤一起轉強。",
        "DO_NOT_CHASE": "不要追高，等回到合理位置。",
        "INVALIDATED": "先不進場，等走勢重新站穩。",
        "DATA_UNVERIFIED": "先更新最新交易日資料。",
        "NO_RETURN_ESTIMATE": "歷史案例不足，先觀察。",
    }.get(condition, "先觀察。")
    if not fresh.get("current_daily"):
        action_text = f"資料停在 {obj.get('price_date','—')}，先更新再判斷。"

    horizon_result = {"short": "約2週", "mid": "約2個月", "long": "約半年"}.get(h, "之後")
    pos_text = pct_decimal(pos_rate, signed=False) if pos_rate is not None else "—"
    med_text = pct_decimal(empirical_med) if empirical_med is not None else "案例不足"

    st.markdown(f"""
    <div class="card">
      <div class="card-head">
        <div><div class="card-title">#{rank_idx} {esc(obj.get('name', obj.get('ticker','')))}</div>
        <div class="card-code">{esc(obj.get('ticker',''))} · {esc(obj.get('industry',''))}</div></div>
        <div class="card-price">{num(obj.get('price'))} 元<div class="card-code">{esc(obj.get('price_date',''))} 收盤</div></div>
      </div>
      <span class="industry-tag">{esc(fine)}</span>
      <span class="badge badge-blue">{esc(HORIZON_LABELS.get(h,h))}</span>

      <div class="verdict">
        <div class="verdict-title">總評｜{esc(view_info.get('title','觀察'))}</div>
        <div class="reason-label">推薦理由</div>
        <ul class="reason-list">{reason_items}</ul>
        {risk_html}
      </div>

      <div class="return-box">
        <div class="return-k">過去相似走勢｜{esc(horizon_result)}</div>
        <div class="return-v">{med_text}</div>
        <div class="return-desc">中位數 · 上漲比例 {pos_text} · {sample_n:,} 個案例</div>
      </div>
      <div class="stats">
        <div class="stat"><div class="k">近20日相對大盤</div><div class="v">{pct_decimal(f.get('alpha_mean'))}</div></div>
        <div class="stat"><div class="k">較差情況平均</div><div class="v">{pct_decimal(strategy.get('expected_shortfall10_loss')) if strategy else '—'}</div></div>
      </div>
      {_research_html(obj)}
      <div class="decision {'decision-ok' if review_ok and condition=='CONDITIONS_MET_NOT_FILLED' else ''}"><b>操作｜</b>{esc(action_text)}</div>
    </div>
    """, unsafe_allow_html=True)

    _render_research_details(obj, f"research_{view}_{h}_{obj.get('ticker')}")

    with st.expander("K線與價格區間", expanded=False):
        if plan:
            st.markdown(f"""<div class="levels">
              <div class="level"><div class="k">較理想區間</div><div class="v">{num(plan.get('zone_low'))}–{num(plan.get('zone_high'))}</div></div>
              <div class="level"><div class="k">轉強確認</div><div class="v">{num(plan.get('trigger'))}</div></div>
              <div class="level"><div class="k">不追高超過</div><div class="v">{num(plan.get('chase_limit'))}</div></div>
              <div class="level"><div class="k">跌破需重看</div><div class="v">{num(plan.get('invalidation'))}</div></div>
            </div>""", unsafe_allow_html=True)
        snap_id = snap.get("snapshot_id", "default") if isinstance(snap, dict) else "default"
        chart = chart or (snap.get("charts", {}).get(obj.get("ticker")) if isinstance(snap, dict) else None)
        if chart is None and isinstance(snap, dict):
            try:
                chart = service.chart_on_demand(snap, obj.get("ticker", ""), DATA_DIR, allow_fetch=False)
            except Exception:
                chart = None
        render_chart(chart, plan, f"chart_{view}_{h}_{obj.get('ticker')}_{snap_id}")


def render_horizon(snap, h, calendar=None):
    st.subheader(HORIZON_LABELS[h])
    if not snap or not isinstance(snap, dict):
        st.info("尚無研究快照，請先更新市場資料。")
        return
    picked = service.select_market_best(snap, h, n=5)
    if not picked:
        st.info("目前沒有可排序標的。")
        return
    for idx, obj in enumerate(picked, 1):
        card(obj, h, snap, h, calendar=calendar, rank_idx=idx)


def main():
    st.set_page_config(page_title="Alpha Radar V13.2", page_icon="📈", layout="centered", initial_sidebar_state="collapsed")
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown("""<div class="hero"><div class="eyebrow">TAIWAN ALPHA RADAR · V13.2</div><h1>Alpha Radar</h1></div>""", unsafe_allow_html=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if st.session_state.get("alpha_version") != service.OPERATIONS_VERSION:
        for key in ("alpha_snapshot", "alpha_doctor", "alpha_error"):
            st.session_state.pop(key, None)
        st.session_state["alpha_version"] = service.OPERATIONS_VERSION
        previous = service.load_dashboard(DATA_DIR / "dashboard_snapshot.json", include_features=False)
        if previous and previous.get("model_version") == service.OPERATIONS_VERSION:
            st.session_state["alpha_snapshot"] = service.compact_session_dashboard(previous)

    with st.sidebar:
        st.markdown("### 掃描設定")
        family_label = st.selectbox("資料範圍", list(FAMILY_LABELS), index=0, key="family_v132")
        period = st.selectbox("歷史比較長度", ["5y", "8y", "10y", "3y"], index=0, key="period_v132")
        candidate_size = st.selectbox("深入比較股票數", [400, 700, 1000], index=1, key="candidate_v132")
        research_pool = st.selectbox("補充公司資料的候選數", [5, 10, 15], index=1, key="research_pool_v132")
        status = provider_runtime_status()
        st.caption(f"FinMind：{'已設定 Token' if status['finmind_token_configured'] else '未設定 Token'}")
        st.caption(f"券商分點：{'已啟用' if status['branch_flow_enabled'] else '未啟用'}")
        with st.expander("快取維護", expanded=False):
            if st.button("清除行情與舊快照", key="clear_v132"):
                DailyPriceStore(DATA_DIR / "daily_prices.sqlite").clear()
                service.remove_saved_dashboard(DATA_DIR / "dashboard_snapshot.json")
                st.session_state.pop("alpha_snapshot", None)
                st.success("已清除。")

    settings = service.RunSettings(
        reference_size=600,
        candidate_size=int(candidate_size),
        research_pool_per_horizon=int(research_pool),
        history_period=period,
        model_family=FAMILY_LABELS[family_label],
    )

    if st.button("⚡ 更新最新交易日並重新選股", type="primary", use_container_width=True, key="scan_v132"):
        progress = st.progress(0, text="更新資料")
        try:
            def update(stage, value):
                progress.progress(min(1.0, max(0.0, float(value))), text=stage)
            snap = service.run_scan(DATA_DIR, settings, progress=update)
            st.session_state["alpha_snapshot"] = service.compact_session_dashboard(snap)
            st.session_state.pop("alpha_doctor", None)
            st.session_state.pop("alpha_error", None)
            del snap
            gc.collect()
        except Exception as exc:
            st.session_state["alpha_error"] = f"{type(exc).__name__}: {exc}"
            st.error(f"更新失敗：{type(exc).__name__}: {exc}")
        finally:
            progress.empty()

    snap = st.session_state.get("alpha_snapshot")
    calendar = calendar_reference(DATA_DIR, now=_taipei_timestamp())
    if snap and isinstance(snap, dict):
        m = snap.get("market", {})
        cov = snap.get("coverage", {})
        off = cov.get("official_eod") or {}
        target = str(cov.get("expected_date") or calendar.get("expected_date") or "")
        twse_date = str(off.get("twse_date") or "—")
        tpex_date = str(off.get("tpex_date") or "—")
        latest = str(snap.get("price_date") or "—")
        st.markdown(
            f"<div class='statusline'><b>最新完整交易日 {esc(latest)}</b> · {REGIME_LABELS.get(m.get('regime'),'⚪ 大盤待確認')}<br>上市 {esc(twse_date)} · 上櫃 {esc(tpex_date)}</div>",
            unsafe_allow_html=True,
        )

        stale = int(cov.get("stale_excluded") or 0)
        if target and latest != "—" and latest < target:
            st.error(f"系統預期最新交易日為 {target}，目前只取得 {latest}；本次不把舊資料當成最新推薦。")
        elif stale > 0:
            st.warning(f"有 {stale:,} 檔仍停在較舊交易日，已從本輪推薦排除。")

        source_problem = bool(
            (target and twse_date not in {"—", ""} and twse_date < target)
            or (target and tpex_date not in {"—", ""} and tpex_date < target)
            or (twse_date == "—" or tpex_date == "—")
        )
        if source_problem:
            with st.expander("資料來源狀態", expanded=True):
                st.write(f"上市：{off.get('twse_source') or '未取得'}｜{twse_date}")
                st.write(f"上櫃：{off.get('tpex_source') or '未取得'}｜{tpex_date}")
                st.write(f"大盤：{off.get('index_source') or '未取得'}｜{off.get('index_date') or '—'}")
                if cov.get("errors"):
                    st.code("\n".join(str(x) for x in cov.get("errors", [])[:10]))

    err = st.session_state.get("alpha_error")
    if err:
        st.warning(err)

    view = st.radio("功能", VIEW_LABELS, horizontal=True, label_visibility="collapsed", key="view_v132")
    if view == VIEW_LABELS[0]:
        st.subheader("⭐ 短、中、長線代表")
        if not snap:
            st.info("尚無資料，請先更新。")
        else:
            used = set()
            for h in HORIZONS:
                picks = service.select_market_best(snap, h, n=5)
                obj = next((x for x in picks if x.get("ticker") not in used), picks[0] if picks else None)
                if obj:
                    used.add(obj.get("ticker"))
                    rank = next((i for i, x in enumerate(picks, 1) if x.get("ticker") == obj.get("ticker")), 1)
                    card(obj, h, snap, "prime", calendar=calendar, rank_idx=rank)
    elif view == VIEW_LABELS[4]:
        st.subheader("🔎 個股診斷")
        with st.form("doctor_v132"):
            code = st.text_input("股票代碼", value="2330", help="如 2330 或 6187")
            horizon_text = st.selectbox("評估週期", list(HORIZON_LABELS.values()), index=1)
            own = st.checkbox("已有持股")
            c1, c2 = st.columns(2)
            with c1:
                cost = st.number_input("買進成本（元/股）", min_value=0.0, value=0.0)
                invalid = st.number_input("原始停損價（元/股）", min_value=0.0, value=0.0)
            with c2:
                shares = st.number_input("持股股數", min_value=0, value=0, step=1000)
                trail = st.number_input("獲利保護價（元/股）", min_value=0.0, value=0.0)
            thesis = st.selectbox("原買進理由是否仍成立？", ["尚未確認", "仍成立", "已不成立"])
            submitted = st.form_submit_button("診斷", type="primary", use_container_width=True)
        if submitted:
            dr = service.diagnose(code, snap, DATA_DIR)
            dr["h"] = next(k for k, v in HORIZON_LABELS.items() if v == horizon_text)
            st.session_state["alpha_doctor"] = service.compact_doctor_result(dr)
        dr = st.session_state.get("alpha_doctor")
        if dr:
            if dr.get("error"):
                st.error(f"資料不足：{dr.get('error')}")
            elif dr.get("stock"):
                card(dr["stock"], dr.get("h", "mid"), snap or {}, "doctor", calendar=calendar)
                if own:
                    result = holding_review(
                        dr["stock"]["price"],
                        original_invalidation=invalid if invalid > 0 else None,
                        trailing_protection=trail if trail > 0 else None,
                        thesis_broken=(thesis == "已不成立"),
                    )
                    pnl = (dr["stock"]["price"] / cost - 1.0) * 100.0 if cost > 0 else None
                    st.markdown("#### 持股檢查")
                    st.info(HOLD_LABELS.get(result, result) + (f"｜目前損益約 {pnl:+.2f}%" if pnl is not None else ""))
    else:
        render_horizon(snap, {VIEW_LABELS[1]:"short", VIEW_LABELS[2]:"mid", VIEW_LABELS[3]:"long"}[view], calendar=calendar)

    st.caption("歷史案例只供比較，不代表未來結果。")


if __name__ == "__main__":
    main()
