# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 11:21:07 2025

@author: EDSON.DOSSANTOS
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd

# ------------------------------
# Tipos literais para parâmetros
# ------------------------------
CompareBy   = Literal["year", "month"]
AggFreq     = Literal["monthly", "daily", "hourly"]
AggStat     = Literal["mean", "min", "max"]
CompareMode = Literal["curve", "value"]


@dataclass
class SimilarityResult:
    """Resultado padronizado da função de similaridade.

    Attributes
    ----------
    ranking_long : pd.DataFrame
        Tabela "longa": uma linha por (candidato, variável), contendo métricas,
        scores por métrica, score da variável e o score final do candidato.
    ranking_summary : pd.DataFrame
        Resumo por candidato (ano ou ano-mês), com score_final, cobertura média
        ponderada e a posição do rank.
    targets : Dict[str, pd.Series]
        Dicionário com as curvas-alvo por variável (útil para inspeção/plots).
    """
    ranking_long: pd.DataFrame
    ranking_summary: pd.DataFrame
    targets: Dict[str, pd.Series]


def rank_similarity(
    data: Union[pd.DataFrame, Dict[str, Union[pd.Series, pd.DataFrame]]],
    *,
    compare_by: CompareBy,
    agg_freq: AggFreq,
    agg_stat: AggStat = "mean",
    compare_mode: CompareMode = "curve",
    target: Optional[Union[pd.Series, pd.DataFrame, Dict[str, Union[pd.Series, pd.DataFrame]]]] = None,
    target_month: Optional[int] = None,
    variables: Optional[List[str]] = None,
    metrics: Optional[List[str]] = None,
    metric_weights: Optional[Dict[str, float]] = None,
    var_weights: Optional[Dict[str, float]] = None,
    normalize: Optional[Literal["none", "zscore", "minmax"]] = "none",
    dtw_radius: Optional[int] = None,
    top_k: int = 5,
    min_coverage: float = 0.7,
    align_how: Literal["inner", "left_target"] = "inner",
    # --- Outliers ---
    outlier_method: Optional[Literal["iqr", "zscore", "mad"]] = None,
    outlier_scope: Literal["none", "pre_agg", "post_agg", "both"] = "none",
    outlier_min_points: int = 5,
    z_thresh: float = 3.0,
    iqr_factor: float = 1.5,
    mad_thresh: float = 3.5,
) -> SimilarityResult:
    """Ranqueia os candidatos (anos ou ano-mês) mais similares a um alvo.

    A função aceita uma série temporal (ou múltiplas variáveis) e compara
    candidatos (agrupados por ano, ou por ano dentro de um mês específico)
    contra uma curva/valor **alvo** (fornecido ou construído automaticamente).

    O usuário escolhe: frequência/estatística de agregação, métricas de
    similaridade (corr, rmse, mae, dtw), pesos por métrica em [0..1], pesos por
    variável, normalização, remoção de outliers e regra de cobertura mínima.

    Parâmetros
    ----------
    data : DataFrame | dict
        Série(s) temporal(is) com índice `DatetimeIndex`.
        - DataFrame: colunas numéricas são variáveis; use `variables` para filtrar.
        - Dict: {nome_var: Series/DataFrame(com 1 coluna)}.
    compare_by : {"year", "month"}
        "year": candidatos são anos.  "month": candidatos são ano-mês
        (necessita `target_month`).
    agg_freq : {"monthly", "daily", "hourly"}
        Frequência de agregação. "monthly" só é válido quando compare_by="year".
    agg_stat : {"mean", "min", "max"}, default "mean"
        Estatística aplicada ao reamostrar as séries.
    compare_mode : {"curve", "value"}, default "curve"
        "curve": compara curvas ponto a ponto.
        "value": compara um escalar por grupo (via `agg_stat`).
    target : Series | DataFrame | dict | None
        Alvo opcional. Se não informado, a função constrói a curva-alvo a partir
        da própria base (climatologia mensal para monthly, DOY para daily, etc.).
        Pode ser um dict {var: série/df} para múltiplas variáveis.
    target_month : int | None
        Necessário quando compare_by="month" (1..12). Define o mês alvo.
    variables : list[str] | None
        Quais colunas usar quando `data` é DataFrame. Default: todas numéricas.
    metrics : list[str] | None
        Quais métricas usar. Opções: ["corr","rmse","mae","dtw"].
        Default: todas as quatro, nessa ordem.
    metric_weights : dict[str,float] | None
        **Distribuição de pesos** (cada w ∈ [0,1]) para as métricas em `metrics`;
        a soma é **normalizada para 1** internamente. Pelo menos um peso > 0.
        Ex.: {"corr":0.5,"rmse":0.2,"mae":0.2,"dtw":0.1}.
    var_weights : dict[str,float] | None
        Pesos por variável (normalizados para somar 1). Se None: iguais.
    normalize : {"none","zscore","minmax"}
        Normalização **após** a agregação. "zscore" e "minmax" ajudam a focar
        na forma da curva (tirando o efeito de nível/escala).
    dtw_radius : int | None
        Janela Sakoe-Chiba para DTW (limita a deformação temporal). None = sem
        restrição (mais custoso).
    top_k : int, default 5
        Quantidade de candidatos no topo do ranking a retornar.
    min_coverage : float, default 0.7
        Fração mínima (0..1) de pontos **válidos do alvo** que devem ser
        comparados (pares não-NaN) para que o candidato seja aceito.
    align_how : {"inner","left_target"}
        "inner": interseção de índices.  "left_target": reindex no alvo e
        descarta pares com NaN em qualquer lado.
    outlier_method : {None,"iqr","zscore","mad"}
        Método para marcação de outliers (pontos extremos são trocados por NaN).
    outlier_scope : {"none","pre_agg","post_agg","both"}
        Onde aplicar: antes da agregação, depois, ou ambos.
    outlier_min_points : int
        Mínimo de pontos na janela para aplicar o filtro de outliers.
    z_thresh : float
        Limiar de |z| para método Z-score.
    iqr_factor : float
        Fator k do IQR (mantém valores em [Q1-k·IQR, Q3+k·IQR]).
    mad_thresh : float
        Limiar de |z_robusto| no método MAD (baseado no desvio absoluto mediano).

    Retorno
    -------
    SimilarityResult
        ranking_summary: ranking final por candidato.
        ranking_long: detalhamento por variável com métricas e scores.
        targets: curvas-alvo por variável.
    """
    # ------------------------ Helpers internos ------------------------
    def _as_series_dict(
        data_obj: Union[pd.DataFrame, Dict[str, Union[pd.Series, pd.DataFrame]]],
        variables: Optional[List[str]],
    ) -> Dict[str, pd.Series]:
        """Converte a entrada em dict {variável: Series float com DatetimeIndex}."""
        if isinstance(data_obj, dict):
            out: Dict[str, pd.Series] = {}
            for k, v in data_obj.items():
                if isinstance(v, pd.DataFrame):
                    if v.shape[1] != 1:
                        raise ValueError(f"DataFrame para var '{k}' deve ter 1 coluna.")
                    s = v.iloc[:, 0]
                else:
                    s = v
                if not isinstance(s.index, pd.DatetimeIndex):
                    raise ValueError(f"Índice de '{k}' deve ser DatetimeIndex.")
                out[k] = s.sort_index().astype(float)
            return out
        else:
            df = data_obj.copy()
            if not isinstance(df.index, pd.DatetimeIndex):
                raise ValueError("Índice do DataFrame deve ser DatetimeIndex.")
            if variables is None:
                variables = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            return {c: df[c].dropna().astype(float).sort_index() for c in variables}

    # ------- Outlier utilities -------
    def _mask_outliers(s: pd.Series, method: Optional[str]) -> pd.Series:
        """Retorna máscara booleana True=manter, False=outlier, conforme método."""
        if method is None:
            return pd.Series(True, index=s.index)
        x = s.astype(float).values
        if len(x) < outlier_min_points:
            return pd.Series(True, index=s.index)

        if method == "zscore":
            mu = np.nanmean(x)
            sd = np.nanstd(x)
            if not np.isfinite(sd) or sd == 0:
                return pd.Series(True, index=s.index)
            z = (x - mu) / sd
            keep = np.abs(z) <= z_thresh
        elif method == "iqr":
            q1 = np.nanpercentile(x, 25)
            q3 = np.nanpercentile(x, 75)
            iqr = q3 - q1
            low = q1 - iqr_factor * iqr
            high = q3 + iqr_factor * iqr
            keep = (x >= low) & (x <= high)
        elif method == "mad":
            med = np.nanmedian(x)
            mad = np.nanmedian(np.abs(x - med)) + 1e-12
            robust_z = 0.6745 * (x - med) / mad
            keep = np.abs(robust_z) <= mad_thresh
        else:
            raise ValueError("outlier_method inválido.")
        # Mantém NaNs como NaNs (não os marca como outliers)
        keep = keep | ~np.isfinite(x)
        return pd.Series(keep, index=s.index)

    def _remove_outliers_in_group(g: pd.Series, method: Optional[str]) -> pd.Series:
        if method is None or g.dropna().shape[0] < outlier_min_points:
            return g
        keep = _mask_outliers(g, method)
        return g.where(keep)

    def _aggregate_series(s: pd.Series, freq: AggFreq, stat: AggStat) -> pd.Series:
        """Reamostra a série + aplica filtros de outlier pre/post conforme config."""
        if freq == "monthly":
            res = s.resample("MS")  # etiqueta no início do mês
        elif freq == "daily":
            res = s.resample("D")
        elif freq == "hourly":
            res = s.resample("H")
        else:
            raise ValueError("agg_freq inválido.")

        # Pre-aggregation: filtra outliers dentro da janela de reamostragem
        if outlier_scope in ("pre_agg", "both") and outlier_method is not None:
            def _agg_with_pre(g):
                g2 = _remove_outliers_in_group(g, outlier_method)
                if stat == "mean":
                    return g2.mean()
                elif stat == "min":
                    return g2.min()
                elif stat == "max":
                    return g2.max()
            out = res.apply(_agg_with_pre)
        else:
            if stat == "mean":
                out = res.mean()
            elif stat == "min":
                out = res.min()
            elif stat == "max":
                out = res.max()

        # Post-aggregation: filtra outliers da curva agregada
        if outlier_scope in ("post_agg", "both") and outlier_method is not None:
            out = _remove_outliers_in_group(out, outlier_method)

        return out.astype(float)

    def _normalize_series(s: pd.Series, how: Optional[str]) -> pd.Series:
        if how is None or how == "none":
            return s
        if how == "zscore":
            std = s.std(skipna=True)
            if std == 0 or np.isnan(std):
                return s * 0.0
            return (s - s.mean(skipna=True)) / std
        if how == "minmax":
            mn, mx = s.min(skipna=True), s.max(skipna=True)
            if mx == mn:
                return pd.Series(0.5, index=s.index)
            return (s - mn) / (mx - mn)
        raise ValueError("normalize inválido.")

    # ---- Métricas e utilitários de alinhamento ----
    def _dtw_distance(a: np.ndarray, b: np.ndarray, radius: Optional[int] = None) -> float:
        """DTW (L1) com janela Sakoe-Chiba opcional; custo O(n*m) limitado pela janela."""
        n, m = len(a), len(b)
        if n == 0 or m == 0:
            return np.inf
        inf = float("inf")
        if radius is None:
            radius = max(n, m)
        radius = int(radius)
        prev = np.full(m + 1, inf)
        curr = np.full(m + 1, inf)
        prev[0] = 0.0
        for i in range(1, n + 1):
            j_start = max(1, i - radius)
            j_end = min(m, i + radius)
            curr.fill(inf)
            for j in range(j_start, j_end + 1):
                cost = abs(a[i - 1] - b[j - 1])
                curr[j] = cost + min(curr[j - 1], prev[j], prev[j - 1])
            prev, curr = curr, prev
        return prev[m]

    def _align_and_drop_nan(target_s: pd.Series, cand_s: pd.Series, how: str) -> Tuple[np.ndarray, np.ndarray, int]:
        """Alinha pelo método escolhido e remove pares com NaN.

        Retorna:
            x, y : arrays alinhados sem NaN
            denom: número de pontos válidos no alvo (base p/ coverage)
        """
        if how == "inner":
            idx = target_s.index.intersection(cand_s.index)
            xt = target_s.reindex(idx)
            yt = cand_s.reindex(idx)
        elif how == "left_target":
            xt = target_s.copy()
            yt = cand_s.reindex(target_s.index)
        else:
            raise ValueError("align_how inválido.")
        denom = int(xt.notna().sum())
        mask = xt.notna() & yt.notna()
        x = xt[mask].to_numpy(dtype=float)
        y = yt[mask].to_numpy(dtype=float)
        return x, y, denom

    def _corr(x: np.ndarray, y: np.ndarray) -> float:
        if len(x) < 2:
            return np.nan
        sx, sy = np.std(x), np.std(y)
        if sx == 0 or sy == 0:
            return np.nan
        return float(np.corrcoef(x, y)[0, 1])

    def _rmse(x: np.ndarray, y: np.ndarray) -> float:
        if len(x) == 0:
            return np.nan
        return float(np.sqrt(np.mean((x - y) ** 2)))

    def _mae(x: np.ndarray, y: np.ndarray) -> float:
        if len(x) == 0:
            return np.nan
        return float(np.mean(np.abs(x - y)))

    def _scalar_metrics(xv: float, yv: float) -> Dict[str, float]:
        # para compare_mode="value": diferenças em escalares
        if pd.isna(xv) or pd.isna(yv):
            return {"abs_diff": np.nan, "pct_diff": np.nan}
        ad = abs(xv - yv)
        pdiff = ad / (abs(yv) + 1e-12)
        return {"abs_diff": float(ad), "pct_diff": float(pdiff)}

    # ------------------------ Validações ------------------------
    if compare_by not in ("year", "month"):
        raise ValueError("compare_by deve ser 'year' ou 'month'.")
    if agg_freq == "monthly" and compare_by != "year":
        raise ValueError("Agregação mensal só é permitida quando compare_by='year'.")
    if compare_by == "month":
        if target_month is None or not (1 <= int(target_month) <= 12):
            raise ValueError("Para compare_by='month', informe target_month 1..12.")
        target_month = int(target_month)

    # métricas e pesos das métricas
    if metrics is None:
        metrics = ["corr", "rmse", "mae", "dtw"]
    else:
        metrics = list(metrics)
    for m in metrics:
        if m not in {"corr", "rmse", "mae", "dtw"}:
            raise ValueError(f"Métrica não suportada: {m}")

    if metric_weights is None:
        metric_weights = {m: 1.0 / len(metrics) for m in metrics}
    else:
        # garante existência e intervalo [0,1]
        metric_weights = {m: float(metric_weights.get(m, 0.0)) for m in metrics}
        for m, w in metric_weights.items():
            if not (0.0 <= w <= 1.0):
                raise ValueError(f"metric_weights['{m}'] deve estar em [0,1], veio {w}.")
        total = sum(metric_weights.values())
        if total <= 0:
            raise ValueError("Pelo menos um peso de métrica deve ser > 0.")
        metric_weights = {m: (w / total) for m, w in metric_weights.items()}

    # ------------------------ Preparação das variáveis ------------------------
    series_dict = _as_series_dict(data, variables)

    # Agrega (+ outliers) + normaliza
    for k in list(series_dict.keys()):
        s = _aggregate_series(series_dict[k].dropna(), agg_freq, agg_stat)
        s = _normalize_series(s, normalize)
        series_dict[k] = s

    # ---- Construção do alvo ----
    def _build_target_for_var(var: str) -> pd.Series:
        # alvo fornecido
        if target is not None:
            if isinstance(target, dict):
                t = target[var]
            else:
                if isinstance(target, pd.DataFrame):
                    if var in target.columns:
                        t = target[var]
                    else:
                        if target.shape[1] != 1:
                            raise ValueError("target DataFrame deve ter a coluna da variável ou 1 coluna.")
                        t = target.iloc[:, 0]
                else:
                    t = target
            t = _aggregate_series(t.dropna(), agg_freq, agg_stat)
            t = _normalize_series(t, normalize)
            return t

        # alvo automático (climatologias)
        s = series_dict[var]
        if compare_by == "year":
            if agg_freq == "monthly":
                df = s.to_frame("v"); df["month"] = df.index.month
                tgt = df.groupby("month")["v"].agg(agg_stat).sort_index()
                tgt.index = pd.Index(tgt.index, name="month")
                return tgt
            elif agg_freq == "daily":
                df = s.to_frame("v"); df["doy"] = df.index.dayofyear
                tgt = df.groupby("doy")["v"].agg(agg_stat).sort_index()
                tgt.index = pd.Index(tgt.index, name="doy")
                return tgt
            elif agg_freq == "hourly":
                df = s.to_frame("v"); df["doy"] = df.index.dayofyear; df["hour"] = df.index.hour
                df["key"] = (df["doy"] - 1) * 24 + df["hour"]
                tgt = df.groupby("key")["v"].agg(agg_stat).sort_index()
                tgt.index = pd.Index(tgt.index, name="key")
                return tgt
        else:  # month
            s_month = s[s.index.month == target_month]
            if agg_freq == "daily":
                df = s_month.to_frame("v"); df["dom"] = df.index.day
                tgt = df.groupby("dom")["v"].agg(agg_stat).sort_index()
                tgt.index = pd.Index(tgt.index, name="dom")
                return tgt
            elif agg_freq == "hourly":
                df = s_month.to_frame("v"); df["dom"] = df.index.day; df["hour"] = df.index.hour
                df["key"] = (df["dom"] - 1) * 24 + df["hour"]
                tgt = df.groupby("key")["v"].agg(agg_stat).sort_index()
                tgt.index = pd.Index(tgt.index, name="key")
                return tgt
        raise RuntimeError("Combinação (compare_by, agg_freq) não suportada ao construir alvo.")

    targets: Dict[str, pd.Series] = {v: _build_target_for_var(v) for v in series_dict.keys()}

    # ---- Enumerar candidatos ----
    def _enumerate_candidates(s: pd.Series) -> Dict[str, pd.Series]:
        if compare_by == "year":
            if agg_freq == "monthly":
                df = s.to_frame("v"); df["year"], df["month"] = df.index.year, df.index.month
                out = {}
                for y, g in df.groupby("year"):
                    ser = g.set_index("month")["v"].sort_index(); ser.index = pd.Index(ser.index, name="month")
                    out[str(y)] = ser
                return out
            elif agg_freq == "daily":
                df = s.to_frame("v"); df["year"], df["doy"] = df.index.year, df.index.dayofyear
                out = {}
                for y, g in df.groupby("year"):
                    ser = g.set_index("doy")["v"].sort_index(); ser.index = pd.Index(ser.index, name="doy")
                    out[str(y)] = ser
                return out
            elif agg_freq == "hourly":
                df = s.to_frame("v"); df["year"], df["doy"], df["hour"] = df.index.year, df.index.dayofyear, df.index.hour
                df["key"] = (df["doy"] - 1) * 24 + df["hour"]
                out = {}
                for y, g in df.groupby("year"):
                    ser = g.set_index("key")["v"].sort_index(); ser.index = pd.Index(ser.index, name="key")
                    out[str(y)] = ser
                return out
        else:  # month
            s_month = s[s.index.month == target_month]
            if agg_freq == "daily":
                df = s_month.to_frame("v"); df["year"], df["dom"] = df.index.year, df.index.day
                out = {}
                for y, g in df.groupby("year"):
                    ser = g.set_index("dom")["v"].sort_index(); ser.index = pd.Index(ser.index, name="dom")
                    out[f"{y}-{target_month:02d}"] = ser
                return out
            elif agg_freq == "hourly":
                df = s_month.to_frame("v"); df["year"], df["dom"], df["hour"] = df.index.year, df.index.day, df.index.hour
                df["key"] = (df["dom"] - 1) * 24 + df["hour"]
                out = {}
                for y, g in df.groupby("year"):
                    ser = g.set_index("key")["v"].sort_index(); ser.index = pd.Index(ser.index, name="key")
                    out[f"{y}-{target_month:02d}"] = ser
                return out
        raise RuntimeError("Combinação (compare_by, agg_freq) não suportada ao enumerar candidatos.")

    candidates_by_var: Dict[str, Dict[str, pd.Series]] = {v: _enumerate_candidates(series_dict[v]) for v in series_dict}

    # Interseção de rótulos (anos/ano-mês) entre todas as variáveis
    candidate_keys = set.intersection(*(set(d.keys()) for d in candidates_by_var.values()))
    if not candidate_keys:
        raise ValueError("Não há candidatos em comum entre as variáveis após agregação/filtragem.")
    candidate_keys = sorted(candidate_keys)

    # Pesos por variável
    if var_weights is None:
        var_weights = {v: 1.0 for v in series_dict.keys()}
    else:
        var_weights = {v: float(var_weights.get(v, 0.0)) for v in series_dict.keys()}
    vw_sum = sum(var_weights.values())
    if vw_sum <= 0:
        raise ValueError("Pelo menos um peso de variável deve ser > 0.")
    var_weights = {v: w / vw_sum for v, w in var_weights.items()}

    # ------------------------ Cálculo das métricas ------------------------
    rows = []
    better_is_high = {"corr": True, "rmse": False, "mae": False, "dtw": False}
    value_metrics = ["abs_diff", "pct_diff"]  # usadas no modo 'value'

    for key in candidate_keys:
        for var, cand_dict in candidates_by_var.items():
            target_s = targets[var]
            cand_s = cand_dict[key]

            if compare_mode == "curve":
                x, y, denom = _align_and_drop_nan(target_s, cand_s, align_how)
                coverage = 0.0 if denom == 0 else (len(x) / denom)
                if coverage < min_coverage or len(x) == 0:
                    mvals = {m: np.nan for m in metrics}
                else:
                    mvals: Dict[str, float] = {}
                    for m in metrics:
                        if m == "corr":
                            mvals[m] = _corr(x, y)
                        elif m == "rmse":
                            mvals[m] = _rmse(x, y)
                        elif m == "mae":
                            mvals[m] = _mae(x, y)
                        elif m == "dtw":
                            mvals[m] = _dtw_distance(x, y, radius=dtw_radius)
                        else:
                            mvals[m] = np.nan
                row = {"candidate": key, "variable": var, "coverage": coverage, **mvals}
                rows.append(row)

            else:  # compare_mode == 'value'
                x_val = getattr(pd.Series(target_s.values), agg_stat)()
                y_val = getattr(pd.Series(cand_s.values), agg_stat)()
                mvals = _scalar_metrics(x_val, y_val)
                row = {"candidate": key, "variable": var, "coverage": 1.0, **mvals}
                rows.append(row)

    df = pd.DataFrame(rows)

    # Filtra candidatos que não atingem cobertura mínima em TODAS as variáveis (apenas no modo curva)
    if compare_mode == "curve":
        def _cand_ok(g: pd.DataFrame) -> bool:
            return (g["coverage"] >= min_coverage).all()
        valid_keys = df.groupby("candidate").apply(_cand_ok)
        df = df[df["candidate"].isin(valid_keys[valid_keys].index)].copy()

    if df.empty:
        raise ValueError("Após filtros de cobertura, não restaram candidatos válidos.")

    # ------------------------ Scoring por variável e final ------------------------
    if compare_mode == "curve":
        parts = []
        for var, g in df.groupby("variable"):
            g = g.copy()
            # normaliza métricas por variável (min-max) respeitando direção (↑/↓)
            for m in metrics:
                vals = g[m].to_numpy()
                vmin = np.nanmin(vals); vmax = np.nanmax(vals)
                if not np.isfinite(vmin) or not np.isfinite(vmax) or math.isclose(vmin, vmax, rel_tol=1e-12, abs_tol=1e-12):
                    g[m + "_score"] = 1.0
                else:
                    if better_is_high[m]:
                        g[m + "_score"] = (vals - vmin) / (vmax - vmin)
                    else:
                        g[m + "_score"] = (vmax - vals) / (vmax - vmin)
            # score por variável = soma ponderada pelos pesos das métricas
            g["score_var"] = 0.0
            for m in metrics:
                g["score_var"] += metric_weights[m] * g[m + "_score"]
            parts.append(g)
        df_scored = pd.concat(parts, ignore_index=True)

    else:  # compare_mode == 'value'
        parts = []
        for var, g in df.groupby("variable"):
            g = g.copy()
            for m in value_metrics:
                vals = g[m].to_numpy()
                vmin = np.nanmin(vals); vmax = np.nanmax(vals)
                if not np.isfinite(vmin) or not np.isfinite(vmax) or math.isclose(vmin, vmax, rel_tol=1e-12, abs_tol=1e-12):
                    g[m + "_score"] = 1.0
                else:
                    g[m + "_score"] = (vmax - vals) / (vmax - vmin)  # menor é melhor
            # aqui mantemos pesos iguais para abs_diff/pct_diff
            g["score_var"] = 0.5 * g["abs_diff_score"] + 0.5 * g["pct_diff_score"]
            parts.append(g)
        df_scored = pd.concat(parts, ignore_index=True)

    # Aplica pesos de variável no score final (média ponderada por var)
    df_scored = df_scored.assign(weight=lambda d: d["variable"].map(var_weights))

    def _agg_weighted(group: pd.DataFrame) -> pd.Series:
        w = group["weight"].to_numpy()
        sv = group["score_var"].to_numpy()
        cov = group["coverage"].to_numpy()
        # média ponderada por variável (pesos do usuário)
        score_final = float(np.average(sv, weights=w))
        coverage_w   = float(np.average(cov, weights=w))
        return pd.Series({"score_final": score_final, "coverage": coverage_w})

    summary = (
        df_scored.groupby("candidate", as_index=False).apply(_agg_weighted).reset_index(drop=True)
    )

    summary = summary.sort_values("score_final", ascending=False).reset_index(drop=True)
    summary["rank"] = np.arange(1, len(summary) + 1)

    # top_k
    top_candidates = set(summary.head(top_k)["candidate"])
    df_long = df_scored[df_scored["candidate"].isin(top_candidates)].copy()
    df_long = df_long.merge(summary[["candidate", "rank", "score_final"]], on="candidate", how="left")
    df_long = df_long.sort_values(["rank", "candidate", "variable"]).reset_index(drop=True)

    return SimilarityResult(ranking_long=df_long, ranking_summary=summary.head(top_k).copy(), targets=targets)
