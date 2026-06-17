"""Оценка справедливой стоимости инструментов: DCF ОФЗ по кривой ZCYC,
цена акций (безмодельная и факторная), пересчёт FX, ребалансировка портфеля.
"""

import numpy as np
import pandas as pd

# конвенции
RANDOM_SEED = 42
YEAR_BASIS = 365          # act/365 для ОФЗ
NOMINAL = 1000.0          # запасной номинал ОФЗ (руб.), если не выводится из cashflows

TENORS = [0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0]

OFZ_ISIN_TO_ID = {
    "SU26219RMFS4": "OFZ_26219",
    "SU26212RMFS9": "OFZ_26212",
    "SU26221RMFS0": "OFZ_26221",
    "SU26218RMFS6": "OFZ_26218",
    "SU26230RMFS1": "OFZ_26230",
}
OFZ_IDS = list(OFZ_ISIN_TO_ID.values())
STOCK_TICKERS = ["SBER", "GAZP", "LKOH", "GMKN", "ROSN",
                 "NVTK", "TATN", "MGNT", "MTSS", "CHMF"]
FX_CURRENCIES = ["USD", "EUR"]


# кривая ставок
def _row_to_curve(row: pd.Series) -> dict:
    return {t: float(row[f"ZCYC_{t}y"]) for t in TENORS}


def get_rate_curve(date, market_data: pd.DataFrame) -> dict:
    """Кривая ZCYC: соответствие сроков и ставок на date из market_data."""
    return _row_to_curve(market_data.loc[pd.Timestamp(date)])


def interp_curve(curve: dict, taus, method: str = "linear") -> np.ndarray:
    """Ставка в процентах годовых на сроки taus (метод 'linear' или 'pchip')."""
    xs = np.asarray(TENORS, dtype=float)
    ys = np.asarray([curve[t] for t in TENORS], dtype=float)
    taus = np.atleast_1d(np.asarray(taus, dtype=float))
    if method == "linear":
        return np.interp(taus, xs, ys)
    if method == "pchip":
        from scipy.interpolate import PchipInterpolator
        return PchipInterpolator(xs, ys)(taus)
    raise ValueError(f"unknown interpolation method: {method!r}")


def discount_factors(curve: dict, taus, method: str = "linear") -> np.ndarray:
    """Дисконтные множители для сроков taus (сложный процент годовых)."""
    z = interp_curve(curve, taus, method) / 100.0
    taus = np.atleast_1d(np.asarray(taus, dtype=float))
    return (1.0 + z) ** (-taus)


# облигации
def _nominal(ofz_id: str, cashflows: pd.DataFrame) -> float:
    """Номинал выпуска (руб.) из платежа погашения (type='amortization'); если такого
    нет — запасная константа NOMINAL. Номинал не хардкодим: у отдельных выпусков он
    может отличаться (П.9 ревью)."""
    sub = cashflows[(cashflows["ofz_id"] == ofz_id)
                    & (cashflows["type"] == "amortization")]
    if not sub.empty:
        return float(sub["value"].sum())
    return float(NOMINAL)


def _bond_flows(ofz_id: str, asof, cashflows: pd.DataFrame):
    """Будущие платежи облигации после asof: список пар (срок в годах, сумма в рублях).

    Если выпуск уже погашён (нет платежей строго после asof), возвращает пустой список;
    тогда price_bond возвращает (0, 0, 0)."""
    asof = pd.Timestamp(asof)
    sub = cashflows[(cashflows["ofz_id"] == ofz_id) & (cashflows["date"] > asof)]
    return [((d - asof).days / YEAR_BASIS, float(v))
            for d, v in zip(sub["date"], sub["value"])]


def _nkd_fraction(ofz_id: str, asof, cashflows: pd.DataFrame) -> float:
    """НКД в долях номинала на asof."""
    asof = pd.Timestamp(asof)
    coupons = cashflows[(cashflows["ofz_id"] == ofz_id)
                        & (cashflows["type"] == "coupon")].sort_values("date")
    prev = coupons[coupons["date"] <= asof]
    nxt = coupons[coupons["date"] > asof]
    if prev.empty or nxt.empty:
        return 0.0
    days_elapsed = (asof - prev["date"].iloc[-1]).days
    days_total = (nxt["date"].iloc[0] - prev["date"].iloc[-1]).days
    coupon = float(nxt["value"].iloc[0])
    return (days_elapsed / days_total) * (coupon / _nominal(ofz_id, cashflows))


def price_bond(ofz_id: str, asof, curve: dict, cashflows: pd.DataFrame,
               *, method: str = "linear"):
    """DCF ОФЗ по кривой ZCYC; возвращает (clean, dirty, nkd) в процентах от номинала.

    Номинал берётся из cashflows (см. _nominal). Если на asof не осталось будущих
    платежей — выпуск погашён — возвращает (0, 0, 0)."""
    flows = _bond_flows(ofz_id, asof, cashflows)
    if not flows:
        return 0.0, 0.0, 0.0
    nominal = _nominal(ofz_id, cashflows)
    taus = np.array([f[0] for f in flows])
    values = np.array([f[1] for f in flows])
    df = discount_factors(curve, taus, method)
    dirty_frac = float(np.sum((values / nominal) * df))
    nkd_frac = _nkd_fraction(ofz_id, asof, cashflows)
    clean_frac = dirty_frac - nkd_frac
    return clean_frac * 100.0, dirty_frac * 100.0, nkd_frac * 100.0


# акции и валюта
def price_equity(base_price: float, cum_logret: float) -> float:
    """Цена как база, наращенная на накопленную лог-доходность."""
    return float(base_price) * float(np.exp(float(cum_logret)))


def price_fx(base_rate: float, cum_logret: float) -> float:
    """Курс как база, наращенная на накопленную лог-доходность."""
    return float(base_rate) * float(np.exp(float(cum_logret)))


def equity_factor_fit(stock_pca: dict, risk_factors: pd.DataFrame,
                      tickers=STOCK_TICKERS) -> dict:
    """OLS доходностей акций на главные компоненты; по тикеру возвращает alpha, betas, r2, resid_std."""
    pcs = stock_pca["pcs"]
    common = risk_factors.index.intersection(pcs.index)
    P = pcs.loc[common, ["STOCK_PC1", "STOCK_PC2", "STOCK_PC3"]].values
    X = np.column_stack([np.ones(len(P)), P])
    out = {}
    for tk in tickers:
        r = risk_factors.loc[common, f"r_{tk}"].values
        beta, *_ = np.linalg.lstsq(X, r, rcond=None)
        rec = X @ beta
        ss_res = float(np.sum((r - rec) ** 2))
        ss_tot = float(np.sum((r - r.mean()) ** 2))
        out[tk] = {"alpha": float(beta[0]),
                   "betas": beta[1:].astype(float),
                   "r2": 1.0 - ss_res / ss_tot if ss_tot else 0.0,
                   "resid_std": float(np.sqrt(ss_res / len(r)))}
    return out


def equity_factor_returns(ticker: str, stock_pca: dict, factor_fit: dict) -> pd.Series:
    """Ряд факторных лог-доходностей по тикеру."""
    pcs = stock_pca["pcs"]
    P = pcs[["STOCK_PC1", "STOCK_PC2", "STOCK_PC3"]].values
    f = factor_fit[ticker]
    return pd.Series(f["alpha"] + P @ f["betas"], index=pcs.index)


def price_equity_factor_model(base_price: float, cum_factor_logret: float) -> float:
    """Цена через факторную модель: база, наращенная на накопленную факторную доходность."""
    return float(base_price) * float(np.exp(float(cum_factor_logret)))


def price_instrument(instrument: dict, risk_factors: dict, prev_prices: dict = None):
    """Стоимость одной единицы инструмента (руб.) по его типу.

    prev_prices входит в требуемый заданием контракт (цены t-1) и зарезервирован для
    будущих типов инструментов — например, опционов и облигаций со встроенным опционом
    (блок 6), где может потребоваться цена t-1. Для базовых классов (bond/equity/fx)
    параметр не используется."""
    itype = instrument["type"]
    if itype == "bond":
        clean_pct, _, _ = price_bond(
            instrument["id"], risk_factors["date"],
            risk_factors["curve"], risk_factors["cashflows"])
        return clean_pct / 100.0 * _nominal(instrument["id"], risk_factors["cashflows"])
    if itype == "equity":
        return price_equity(risk_factors["base_prices"]["equity"][instrument["id"]],
                            risk_factors["equity_cumret"][instrument["id"]])
    if itype == "fx":
        return price_fx(risk_factors["base_prices"]["fx"][instrument["id"]],
                        risk_factors["fx_cumret"][instrument["id"]])
    raise ValueError(f"unknown instrument type: {itype!r}")


# портфель и ребалансировка
def build_holdings(portfolio_spec: dict, prev_day_prices: dict, capital: float) -> dict:
    """Количество единиц каждого инструмента по ценам t-1."""
    holdings = {}
    for ofz_id, spec in portfolio_spec["bonds"].items():
        holdings[ofz_id] = spec["weight"] * capital / prev_day_prices[ofz_id]
    for tk, spec in portfolio_spec["equities"].items():
        holdings[tk] = spec["weight"] * capital / prev_day_prices[tk]
    for ccy, spec in portfolio_spec["fx"].items():
        holdings[ccy] = spec["rub_notional"] / prev_day_prices[ccy]
    return holdings


def price_portfolio(portfolio_spec: dict, risk_factors: dict,
                    prev_day_prices: dict, prev_capital: float) -> dict:
    """Переоценка портфеля по ценам дня t; количества фиксируются по t-1."""
    holdings = build_holdings(portfolio_spec, prev_day_prices, prev_capital)
    prices = risk_factors["prices"]
    by_class = {"bonds": 0.0, "equities": 0.0, "fx": 0.0}
    for ofz_id in portfolio_spec["bonds"]:
        by_class["bonds"] += holdings[ofz_id] * prices[ofz_id]
    for tk in portfolio_spec["equities"]:
        by_class["equities"] += holdings[tk] * prices[tk]
    for ccy in portfolio_spec["fx"]:
        by_class["fx"] += holdings[ccy] * prices[ccy]
    value = float(sum(by_class.values()))
    pnl = value - prev_capital
    ret = pnl / prev_capital if prev_capital else 0.0
    return {"value": value, "by_class": by_class, "pnl": pnl,
            "ret": ret, "holdings": holdings}


def historical_prices(market_data: pd.DataFrame, date) -> dict:
    """Абсолютные цены (руб. за единицу) на дату из market_data.

    ОФЗ-котировка в data_clean дана в % номинала; перевод в рубли идёт по стандартному
    номиналу 1000 (все ОФЗ-ПД). DCF-оценка (price_bond) берёт номинал из cashflows
    (см. _nominal); здесь же это рыночное соглашение о котировке, а не оценка модели."""
    row = market_data.loc[pd.Timestamp(date)]
    prices = {ofz_id: float(row[ofz_id]) / 100.0 * NOMINAL for ofz_id in OFZ_IDS}
    prices.update({tk: float(row[f"STK_{tk}"]) for tk in STOCK_TICKERS})
    prices.update({ccy: float(row[f"FX_{ccy}"]) for ccy in FX_CURRENCIES})
    return prices


def backtest_portfolio(portfolio_spec: dict, market_data: pd.DataFrame,
                       start: str = "2025-01-01", end: str = "2025-12-30") -> pd.DataFrame:
    """Ежедневная ребалансировка на [start, end]; возвращает ряд стоимостей, P&L и доходностей.

    Ряд начинается со стартовой точки на dates[0] — портфель, сконструированный под
    начальный капитал (value = capital, pnl = 0, ret = 0), — а со следующего дня идёт
    переоценка по ценам t при количестве единиц, зафиксированном по t-1.

    Упрощение: купонные выплаты и погашения внутри горизонта не моделируются —
    полученные купоны не добавляются к капиталу и не реинвестируются. В реальности они
    увеличивают капитал; в рамках упрощённой модели daily-rebalanced P&L это допустимо."""
    dates = market_data.loc[start:end].index
    if len(dates) == 0:
        raise ValueError(f"нет торговых дней в market_data на [{start}, {end}]")
    rows = []
    capital = float(portfolio_spec["total_rub_capital"])
    prev_prices = historical_prices(market_data, dates[0])
    # стартовая точка: переоценка сконструированного портфеля по ценам dates[0] даёт
    # value == capital (P&L = 0 по определению); by_class — начальное распределение
    res0 = price_portfolio(portfolio_spec, {"date": dates[0], "prices": prev_prices},
                           prev_prices, capital)
    rows.append({"DATE": dates[0], **res0["by_class"], "value": res0["value"],
                 "pnl": 0.0, "ret": 0.0})
    for d in dates[1:]:
        rf = {"date": d, "prices": historical_prices(market_data, d)}
        res = price_portfolio(portfolio_spec, rf, prev_prices, capital)
        rows.append({"DATE": d, **res["by_class"], "value": res["value"],
                     "pnl": res["pnl"], "ret": res["ret"]})
        capital = res["value"]
        prev_prices = rf["prices"]
    return pd.DataFrame(rows).set_index("DATE")


# валидация
def validate_bonds(market_data: pd.DataFrame, cashflows: pd.DataFrame,
                   start: str = "2025-01-01", end: str = "2025-12-30",
                   method: str = "linear"):
    """Сравнение чистой цены DCF с рыночной ценой ОФЗ в процентах номинала; возвращает long df и summary."""
    dates = market_data.loc[start:end].index
    recs = []
    for d in dates:
        curve = _row_to_curve(market_data.loc[d])
        for ofz_id in OFZ_IDS:
            clean_model, _, _ = price_bond(ofz_id, d, curve, cashflows, method=method)
            recs.append({"DATE": d, "ofz": ofz_id,
                         "model": clean_model, "market": float(market_data.loc[d, ofz_id])})
    df = pd.DataFrame(recs)
    df["err"] = df["model"] - df["market"]
    summary = df.groupby("ofz")["err"].agg(
        bias="mean", std="std",
        MAE=lambda s: float(np.mean(np.abs(s))),
        RMSE=lambda s: float(np.sqrt(np.mean(s ** 2))))
    return df, summary


def validate_cumret(market_data: pd.DataFrame, risk_factors: pd.DataFrame,
                    start: str = "2025-01-01", end: str = "2025-12-30"):
    """Сравнение безмодельной цены с рынком для акций и валют; возвращает df_eq и df_fx."""
    common = market_data.index.intersection(risk_factors.index)
    prev_date = market_data.index[market_data.index < common[0]][-1]

    eq_rows, fx_rows = [], []
    for tk in STOCK_TICKERS:
        base = float(market_data.loc[prev_date, f"STK_{tk}"])
        cum = risk_factors.loc[common, f"r_{tk}"].cumsum()
        model = base * np.exp(cum)
        market = market_data.loc[common, f"STK_{tk}"]
        eq_rows.append(pd.DataFrame({"DATE": common, "id": tk,
                                     "model": model.values, "market": market.values}))
    for ccy in FX_CURRENCIES:
        base = float(market_data.loc[prev_date, f"FX_{ccy}"])
        cum = risk_factors.loc[common, f"r_{ccy}"].cumsum()
        model = base * np.exp(cum)
        market = market_data.loc[common, f"FX_{ccy}"]
        fx_rows.append(pd.DataFrame({"DATE": common, "id": ccy,
                                     "model": model.values, "market": market.values}))
    eq = pd.concat(eq_rows, ignore_index=True).set_index("DATE").sort_index()
    fx = pd.concat(fx_rows, ignore_index=True).set_index("DATE").sort_index()
    # индекс не уникален (несколько id на дату) — фильтруем по значениям
    eq = eq[(eq.index >= pd.Timestamp(start)) & (eq.index <= pd.Timestamp(end))]
    fx = fx[(fx.index >= pd.Timestamp(start)) & (fx.index <= pd.Timestamp(end))]
    eq["err"] = eq["model"] - eq["market"]
    fx["err"] = fx["model"] - fx["market"]
    return eq, fx


def validate_equity_factor_model(stock_pca: dict, risk_factors: pd.DataFrame,
                                 market_data: pd.DataFrame,
                                 tickers=STOCK_TICKERS,
                                 start: str = "2025-01-01", end: str = "2025-12-30"):
    """Сравнение факторной цены с рынком; по тикеру возвращает r2 и mae_pct."""
    fit = equity_factor_fit(stock_pca, risk_factors, tickers)
    pcs = stock_pca["pcs"]
    common = market_data.index.intersection(risk_factors.index).intersection(pcs.index)
    prev_date = market_data.index[market_data.index < common[0]][-1]
    win = (common >= pd.Timestamp(start)) & (common <= pd.Timestamp(end))
    common_win = common[win]
    out = {}
    for tk in tickers:
        base = float(market_data.loc[prev_date, f"STK_{tk}"])
        cum_fac = equity_factor_returns(tk, stock_pca, fit).loc[common].cumsum()
        model = pd.Series(base * np.exp(cum_fac.values), index=common)
        market = market_data.loc[common, f"STK_{tk}"]
        err = model.loc[common_win] - market.loc[common_win]
        out[tk] = {"r2": fit[tk]["r2"],
                   "mae_pct": float(np.mean(np.abs(err))
                                    / float(np.mean(market.loc[common_win])) * 100)}
    return out


DEFAULT_PORTFOLIO = {
    "bonds": {ofz_id: {"weight": 0.08} for ofz_id in OFZ_IDS},            # 40%, 5 ОФЗ
    "equities": {tk: {"weight": 0.03} for tk in STOCK_TICKERS},           # 30%, 10 акций
    "fx": {"USD": {"rub_notional": 100_000_000.0},                        # ≈30%
           "EUR": {"rub_notional": 100_000_000.0}},
    "total_rub_capital": 666_666_667.0,
    "as_of": "2025-12-30",
    "rebalance": "daily",
}
