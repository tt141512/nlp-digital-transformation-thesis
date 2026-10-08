# -*- coding: utf-8 -*-
"""
utils.py —— 公共工具
统一路径、随机种子、以及简化版面板双向固定效应(TWFE)估计。

说明：为了保证代码"零重依赖即可运行"，这里的面板回归在 numpy/pandas 上
自行实现了 within-transformation(去个体均值 + 去时间均值)的 CLS 估计，
并给出聚类稳健标准误；若本机安装了 statsmodels/linearmodels，也不影响本实现。
"""
from __future__ import annotations

import math
import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------- 统计函数
# 说明：为把依赖压到最小，这里用纯 Python 实现 Student-t 双侧 p 值与
# Spearman / Pearson 相关（结果与 scipy 一致，精度 ~1e-10），
# 因此本仓库不强制依赖 scipy。


def _betacf(a: float, b: float, x: float) -> float:
    """连分式展开，用于不完全贝塔函数（Numerical Recipes, 6.4）。"""
    MAXIT, EPS, FPMIN = 300, 3.0e-16, 1.0e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < EPS:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """正则化不完全贝塔函数 I_x(a, b)。"""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                  + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def t_two_sided_p(t: float, df: float) -> float:
    """Student-t 分布双侧 p 值。"""
    t = abs(float(t))
    if not np.isfinite(t) or df <= 0:
        return float("nan")
    return float(_betai(df / 2.0, 0.5, df / (df + t * t)))


def pearsonr(x, y) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = x.size
    if n < 3:
        return float("nan"), float("nan")
    xc, yc = x - x.mean(), y - y.mean()
    denom = math.sqrt(float(xc @ xc) * float(yc @ yc))
    r = float(xc @ yc) / denom if denom > 0 else float("nan")
    r = max(min(r, 1.0), -1.0)
    if abs(r) >= 1.0:
        return r, 0.0
    t = r * math.sqrt((n - 2) / (1 - r * r))
    return r, t_two_sided_p(t, n - 2)


def spearmanr(x, y) -> tuple[float, float]:
    """Spearman 秩相关（对秩计算 Pearson）。"""
    xs = pd.Series(np.asarray(x, dtype=float)).rank()
    ys = pd.Series(np.asarray(y, dtype=float)).rank()
    return pearsonr(xs.values, ys.values)


# ---------------------------------------------------------------- 路径
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P_DATA_RAW = os.path.join(ROOT, "data", "raw")
P_DATA_PROC = os.path.join(ROOT, "data", "processed")
P_LEXICON = os.path.join(ROOT, "data", "lexicon")
P_RESULTS = os.path.join(ROOT, "results")
P_TABLES = os.path.join(P_RESULTS, "tables")
P_FIGS = os.path.join(P_RESULTS, "figures")

for _p in (P_DATA_RAW, P_DATA_PROC, P_LEXICON, P_TABLES, P_FIGS):
    os.makedirs(_p, exist_ok=True)

SEED = 20261007


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)


# ---------------------------------------------------------------- 面板估计
def _demean(df: pd.DataFrame, cols, entity: str, time: str) -> pd.DataFrame:
    """双向去均值(迭代式 within transformation)，返回去均值后的列副本。"""
    out = df[cols].astype(float).copy()
    # 先去掉个体均值，再去掉时间均值，交替 3 轮以逼近双向 within
    for _ in range(3):
        out = out - out.groupby(df[entity]).transform("mean")
        out = out - out.groupby(df[time]).transform("mean")
    return out


def twfe(df: pd.DataFrame, y: str, xs, entity: str = "stock_code",
         time: str = "year", cluster: str | None = None) -> pd.DataFrame:
    """
    双向固定效应估计。
    返回 DataFrame: 变量名 / 系数 / 标准误 / t值 / p值 / 显著性 / 样本量 / 组内R2
    """
    cols = [y] + list(xs)
    # 聚类列可能既不是 entity/time 也不在模型列中，需在子集中保留它；
    # 但若 cluster 与 entity/time 相同，则已存在，不可重复加入（否则选中会变成 DataFrame）
    cols_with_cluster = list(dict.fromkeys(
        cols + ([cluster] if cluster and cluster not in cols
                and cluster not in (entity, time) else [])))
    d = df[[entity, time] + cols_with_cluster].dropna().copy()
    dm = _demean(d, cols, entity, time)

    # 剔除被固定效应吸收的（近）时不变变量：去均值后方差接近 0，
    # 否则会出现 0/0（系数与标准误均为 NaN）
    xs = list(xs)
    _std = dm[xs].std()
    xs = [c for c in xs if float(_std[c]) > 1e-10]

    Y = dm[y].values.reshape(-1, 1)
    X = dm[xs].values
    n, k = X.shape

    XtX = X.T @ X
    XtX_inv = np.linalg.pinv(XtX)
    beta = XtX_inv @ (X.T @ Y)
    resid = (Y - X @ beta).ravel()

    # 自由度修正：扣除个体与时间虚拟变量个数
    g_ent = d[entity].nunique()
    g_time = d[time].nunique()
    df_resid = max(n - k - g_ent - g_time + 1, 1)

    if cluster is not None:
        # 聚类稳健(按 cluster 分组求和)
        s = pd.Series(resid, index=d.index)
        Xd = pd.DataFrame(X, index=d.index, columns=list(xs))
        meat = np.zeros((k, k))
        for _, idx in s.groupby(d[cluster]).groups.items():
            Xi = Xd.loc[idx].values
            ui = s.loc[idx].values.reshape(-1, 1)
            g = Xi.T @ ui
            meat += g @ g.T
        G = d[cluster].nunique()
        c = (G / (G - 1)) * ((n - 1) / df_resid)
        V = XtX_inv @ meat @ XtX_inv * c
    else:
        sigma2 = float(resid @ resid) / df_resid
        V = XtX_inv * sigma2

    se = np.sqrt(np.diag(V))
    tval = beta.ravel() / se
    pval = np.array([t_two_sided_p(t, df_resid) for t in tval])

    def star(p):
        return "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.1 else ""))

    ss_tot = float(((Y - Y.mean()) ** 2).sum())
    ss_res = float(resid @ resid)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return pd.DataFrame({
        "变量": list(xs),
        "系数": beta.ravel(),
        "标准误": se,
        "t值": tval,
        "p值": pval,
        "显著性": [star(p) for p in pval],
        "N": n,
        "组内R2": r2,
    })


def fmt(df: pd.DataFrame, cols, nd=4) -> pd.DataFrame:
    """结果表美化：把系数与标准误拼成 `b***\n(se)` 风格前的准备。"""
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].map(lambda v: round(float(v), nd) if pd.notna(v) else v)
    return out


def twfe_iv(df: pd.DataFrame, y: str, endog, instr, exog=(),
            entity: str = "stock_code", time: str = "year",
            cluster: str | None = None) -> pd.DataFrame:
    """
    面板双向固定效应两阶段最小二乘 (2SLS)。

    先对 y、内生变量、工具变量、外生控制变量统一做 within 变换，
    再用 Z 对 X 拟合得到 X̂，最后对 X̂ 做 OLS。
    标准误按标准 2SLS 公式计算（可选聚类稳健）。
    返回：变量 / 系数 / 标准误 / t值 / p值 / 显著性 / N
    """
    endog = list(endog)
    instr = list(instr)
    exog = list(exog)
    Zcols = instr + exog
    Xcols = endog + exog
    cols = [y] + Xcols + instr
    cols_with_cluster = list(dict.fromkeys(
        cols + ([cluster] if cluster and cluster not in cols
                and cluster not in (entity, time) else [])))
    d = df[[entity, time] + cols_with_cluster].dropna().copy()
    dm = _demean(d, list(dict.fromkeys(cols)), entity, time)

    Y = dm[y].values.reshape(-1, 1)
    X = dm[Xcols].values
    Z = dm[Zcols].values
    n, k = X.shape

    ZtZ_inv = np.linalg.pinv(Z.T @ Z)
    PzX = Z @ (ZtZ_inv @ (Z.T @ X))            # X̂ = Z(Z'Z)^-1 Z'X
    XtX = PzX.T @ PzX
    XtX_inv = np.linalg.pinv(XtX)
    beta = XtX_inv @ (PzX.T @ Y)
    resid = (Y - X @ beta).ravel()

    g_ent, g_time = d[entity].nunique(), d[time].nunique()
    df_resid = max(n - k - g_ent - g_time + 1, 1)

    if cluster is not None:
        s = pd.Series(resid, index=d.index)
        Xd = pd.DataFrame(PzX, index=d.index, columns=Xcols)
        meat = np.zeros((k, k))
        for _, idx in s.groupby(d[cluster]).groups.items():
            g = Xd.loc[idx].values.T @ s.loc[idx].values.reshape(-1, 1)
            meat += g @ g.T
        G = d[cluster].nunique()
        V = XtX_inv @ meat @ XtX_inv * ((G / (G - 1)) * ((n - 1) / df_resid))
    else:
        sigma2 = float(resid @ resid) / df_resid
        V = XtX_inv * sigma2

    se = np.sqrt(np.diag(V))
    tval = beta.ravel() / se
    pval = np.array([t_two_sided_p(t, df_resid) for t in tval])

    def star(p):
        return "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.1 else ""))

    return pd.DataFrame({
        "变量": Xcols, "系数": beta.ravel(), "标准误": se, "t值": tval,
        "p值": pval, "显著性": [star(p) for p in pval], "N": n,
    })


def logit_fit(X, y, max_iter: int = 100, tol: float = 1e-9):
    """
    二元 Logit 模型的 IRLS（迭代加权最小二乘）极大似然估计。
    纯 numpy 实现，避免依赖 statsmodels / scikit-learn。
    X 需自行加入常数项列。返回 (beta, p_hat)。
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, k = X.shape
    beta = np.zeros(k)
    for _ in range(max_iter):
        eta = np.clip(X @ beta, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-eta))
        w = np.maximum(p * (1.0 - p), 1e-10)
        z = eta + (y - p) / w
        sw = np.sqrt(w)
        Xw = X * sw[:, None]
        zw = z * sw
        XtWX = Xw.T @ Xw
        try:
            beta_new = np.linalg.solve(XtWX, Xw.T @ zw)
        except np.linalg.LinAlgError:
            beta_new = np.linalg.pinv(XtWX) @ (Xw.T @ zw)
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new
    eta = np.clip(X @ beta, -30.0, 30.0)
    return beta, 1.0 / (1.0 + np.exp(-eta))


def winsorize(df: pd.DataFrame, cols, lower=0.01, upper=0.99) -> pd.DataFrame:
    """按分位数缩尾（默认上下 1%）。"""
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue
        lo, hi = out[c].quantile(lower), out[c].quantile(upper)
        out[c] = out[c].clip(lo, hi)
    return out


def save_table(df: pd.DataFrame, name: str, note: str = "") -> None:
    """同时输出 CSV 与 Markdown 两种格式的结果表。"""
    csv_path = os.path.join(P_TABLES, f"{name}.csv")
    md_path = os.path.join(P_TABLES, f"{name}.md")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    try:
        body = df.to_markdown(index=False)
    except Exception:
        body = "```\n" + df.to_string(index=False) + "\n```"
    with open(md_path, "w", encoding="utf-8") as f:
        if note:
            f.write(f"> {note}\n\n")
        f.write(body + "\n")
    print(f"      -> results/tables/{name}.csv / .md")
