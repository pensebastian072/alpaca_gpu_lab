"""Options liquidity gate for the ETF universe expansion (data-only, read-only).

For each candidate ETF, pull the CURRENT Alpaca option-chain snapshot restricted to the
30-60 DTE, near-ATM band the desk actually trades, and measure whether its options are
liquid enough to sell: median per-leg relative bid/ask spread, open interest, volume.
Pass/fail on the same thresholds as the live desk (`options_etf_universe_v2` liquidity).

This is a NOW liquidity snapshot (can we trade options here today), the honest structural
gate for the universe expansion. It does NOT price history or PnL. Data-only Alpaca market
client -- never the trading client.

    .venv\\Scripts\\python.exe -m src.research.option_liquidity
"""
from __future__ import annotations

import json
import os
import ssl
import statistics
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from .. import config

# Live-desk liquidity thresholds (mirror options_etf_universe_v2 entry_policy.liquidity)
MAX_REL_SPREAD = 0.15
MIN_VOL = 200              # cumulative day option volume across the near-ATM band (RTH, timing-robust)
DTE_LO, DTE_HI = 30, 60
STRIKE_BAND = 0.04         # +/-4% of spot: NEAR-ATM only. Wider bands are dominated by cheap
                          # far-OTM wings whose penny spreads read as huge relative spreads.
MIN_MID = 0.20            # ignore sub-$0.20 contracts (cheap-wing relative-spread inflation)

OUT = config.REPO / "journal" / "option_liquidity.json"

# Candidate symbols: the frozen 31 (known liquid) + the decorrelation shortlist from
# options_desk. GLD trades as GLD (Alpaca ticker), qlib's GOLD alias is internal only.
BASE_31 = ["SPY", "QQQ", "IWM", "DIA", "TLT", "IEF", "AGG", "HYG", "LQD", "GLD", "SLV",
           "XLF", "XLE", "XLK", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC",
           "SMH", "KRE", "GDX", "XBI", "EEM", "EFA", "FXI", "EWZ", "VNQ"]
DECORRELATORS = ["UNG", "XOP", "IWO", "EMB", "DBC", "INDA", "SHY", "XHB", "VLUE", "GDXJ",
                 "XRT", "IYT", "RSP", "EWY", "ITA", "IWN", "MUB", "BNDX", "IWF", "IWD",
                 "MTUM", "QUAL", "USMV", "SPLV", "MDY", "TIP", "EWJ", "EWG", "EWU", "EWC"]


def _trust_and_keys() -> tuple[str, str]:
    try:
        import truststore
        truststore.inject_into_ssl()
    except ImportError:
        pass
    import certifi
    bundle = config.DATA / "_windows_ca_bundle.pem"
    if not bundle.exists():
        bundle.parent.mkdir(parents=True, exist_ok=True)
        parts = [Path(certifi.where()).read_text(encoding="utf-8")]
        for store in ("ROOT", "CA"):
            for der, enc, _ in ssl.enum_certificates(store):
                if enc == "x509_asn":
                    parts.append(ssl.DER_cert_to_PEM_cert(der))
        bundle.write_text("\n".join(parts), encoding="utf-8")
    for v in ("CURL_CA_BUNDLE", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"):
        os.environ[v] = str(bundle)
    envf = config.REPO / ".env"
    for line in envf.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, val = line.split("=", 1)
            os.environ.setdefault(k.strip(), val.strip().strip('"').strip("'"))
    return os.environ["APCA_API_KEY_ID"], os.environ["APCA_API_SECRET_KEY"]


def _spot(stock_client, symbol: str) -> float | None:
    from alpaca.data.requests import StockLatestTradeRequest
    try:
        t = stock_client.get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols=symbol))
        return float(t[symbol].price)
    except Exception:  # noqa: BLE001
        return None


def screen_symbol(opt_client, stock_client, symbol: str, now: datetime) -> dict:
    from alpaca.data.requests import OptionChainRequest
    spot = _spot(stock_client, symbol)
    if not spot or spot <= 0:
        return {"symbol": symbol, "ok": False, "reason": "no spot"}
    lo = (now + timedelta(days=DTE_LO)).date()
    hi = (now + timedelta(days=DTE_HI)).date()
    req = OptionChainRequest(
        underlying_symbol=symbol,
        expiration_date_gte=lo, expiration_date_lte=hi,
        strike_price_gte=str(round(spot * (1 - STRIKE_BAND), 2)),
        strike_price_lte=str(round(spot * (1 + STRIKE_BAND), 2)),
    )
    try:
        chain = opt_client.get_option_chain(req)
    except Exception as e:  # noqa: BLE001
        return {"symbol": symbol, "ok": False, "reason": f"chain error: {str(e)[:80]}"}
    spreads, vols, n_quoted = [], [], 0
    for _sym, snap in chain.items():
        q = getattr(snap, "latest_quote", None)
        if q is None:
            continue
        bid, ask = float(q.bid_price or 0), float(q.ask_price or 0)
        if bid <= 0 or ask <= 0 or ask < bid:
            continue
        mid = (bid + ask) / 2
        if mid < MIN_MID:          # skip cheap wings that inflate relative spread
            continue
        n_quoted += 1
        spreads.append((ask - bid) / mid)
        tv = getattr(getattr(snap, "latest_trade", None), "size", None)
        if tv is not None:
            vols.append(float(tv))
    if n_quoted < 3:
        return {"symbol": symbol, "ok": False, "reason": f"only {n_quoted} near-ATM two-sided quotes >= ${MIN_MID}",
                "spot": round(spot, 2)}
    med_spread = round(statistics.median(spreads), 4)
    tot_vol = round(sum(vols), 0) if vols else 0
    liquid = (med_spread <= MAX_REL_SPREAD) and (tot_vol >= MIN_VOL)
    return {"symbol": symbol, "ok": True, "liquid": bool(liquid), "spot": round(spot, 2),
            "n_quoted": n_quoted, "median_rel_spread": med_spread,
            "atm_volume": tot_vol,
            "reason": "" if liquid else f"spread {med_spread:.0%} / atm_vol {tot_vol}"}


def run(symbols: list[str] | None = None) -> dict:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.historical.option import OptionHistoricalDataClient
    key, sec = _trust_and_keys()
    opt = OptionHistoricalDataClient(key, sec)
    stock = StockHistoricalDataClient(key, sec)
    now = datetime.now(timezone.utc)
    syms = symbols or (BASE_31 + DECORRELATORS)
    results = []
    for s in syms:
        r = screen_symbol(opt, stock, s, now)
        results.append(r)
        tag = ("LIQUID" if r.get("liquid") else "thin" if r.get("ok") else "ERR")
        print(f"  {s:5s} {tag:6s} spread={r.get('median_rel_spread')} "
              f"OI={r.get('median_open_interest')} vol={r.get('total_volume')} {r.get('reason','')}")
    passed = [r["symbol"] for r in results if r.get("liquid")]
    return {"generated_at": now.isoformat(), "dte_band": [DTE_LO, DTE_HI],
            "feed_caveat": ("Alpaca option quotes here are the non-OPRA 'indicative' feed; "
                            "absolute spreads are inflated and unreliable. Trust the VOLUME "
                            "(real trades) and the relative ranking; confirm spreads via OPRA "
                            "or a broker RTH quote (Robinhood) before adopting."),
            "thresholds": {"max_rel_spread": MAX_REL_SPREAD, "min_vol": MIN_VOL, "min_mid": MIN_MID},
            "base_31": BASE_31, "decorrelator_shortlist": DECORRELATORS,
            "results": results,
            "liquid_symbols": passed,
            "new_liquid_decorrelators": [s for s in passed if s in DECORRELATORS]}


def main() -> None:
    report = run()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\noption liquidity -> {OUT}")
    print(f"new liquid decorrelators ({len(report['new_liquid_decorrelators'])}): "
          f"{report['new_liquid_decorrelators']}")


if __name__ == "__main__":
    main()
