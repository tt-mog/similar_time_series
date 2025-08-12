# rank_similarity: Ranqueamento de similaridade de curvas/valores em séries temporais

`rank_similarity` ranqueia **anos** (ou **ano-mês**) que apresentem curvas/valores mais semelhantes a um **alvo** em séries temporais. Funciona com **uma ou várias variáveis**, suporta **agregação temporal** (mensal/diária/horária), **remoção de outliers**, **normalização**, escolha de **métricas** e **pesos por métrica** (distribuição em [0..1]), além de **pesos por variável**.

A saída traz:
- `ranking_summary`: top-*k* candidatos com `score_final`, `coverage` e `rank`.
- `ranking_long`: detalhamento por candidato×variável, com todas as métricas e scores parciais.
- `targets`: curvas-alvo por variável (útil para visualização/validação).

## Instalação

Requer Python 3.9+ e as libs: `pandas`, `numpy`.
```bash
pip install pandas numpy
```

## Assinatura
```python
SimilarityResult = rank_similarity(
    data,
    *,
    compare_by: {"year","month"},
    agg_freq: {"monthly","daily","hourly"},
    agg_stat: {"mean","min","max"} = "mean",
    compare_mode: {"curve","value"} = "curve",
    target: Series|DataFrame|dict|None = None,
    target_month: int|None = None,
    variables: list[str]|None = None,
    metrics: list[str]|None = None,
    metric_weights: dict[str,float]|None = None,   # distribuição [0..1], soma normalizada p/ 1
    var_weights: dict[str,float]|None = None,      # normalizada p/ soma 1
    normalize: {"none","zscore","minmax"} = "none",
    dtw_radius: int|None = None,
    top_k: int = 5,
    min_coverage: float = 0.7,
    align_how: {"inner","left_target"} = "inner",
    outlier_method: {None,"iqr","zscore","mad"} = None,
    outlier_scope: {"none","pre_agg","post_agg","both"} = "none",
    outlier_min_points: int = 5,
    z_thresh: float = 3.0,
    iqr_factor: float = 1.5,
    mad_thresh: float = 3.5,
)
```

## Parâmetros essenciais
- **compare_by**: `"year"` (anos) ou `"month"` (ano-mês, requer `target_month`).
- **agg_freq** / **agg_stat**: reamostragem + estatística (ex.: `"monthly"` + `"mean"` → médias mensais).
- **compare_mode**: `"curve"` (curvas ponto a ponto) ou `"value"` (um escalar por grupo via `agg_stat`).
- **metrics** (modo curva): `corr`, `rmse`, `mae`, `dtw`.
- **metric_weights**: distribuição em [0..1] por métrica (normalizada para somar 1).
- **var_weights**: pesos por variável (normalizados para somar 1).
- **normalize**: `"zscore"`/`"minmax"` para focar **forma** em vez de nível/escala.
- **min_coverage**: fração mínima do alvo (pontos válidos) que precisa virar par válido (~70% por padrão).
- **outlier_method/outlier_scope**: tratamento de outliers antes/depois da agregação.

## Construção do alvo (quando `target` é None)
- `compare_by="year"`
  - `agg_freq="monthly"` → **climatologia mensal** (12 pontos).
  - `agg_freq="daily"` → alvo por **dia-do-ano** (1..365/366).
  - `agg_freq="hourly"` → alvo por **(dia-do-ano, hora)**.
- `compare_by="month"` (requer `target_month`)
  - `daily`/`hourly` → alvo por **dia-do-mês** (e hora) agregando sobre os anos.

## Tratamento de NaNs e *coverage*
1. Alinhamento (`inner` ou `left_target`).  
2. Remoção de quaisquer linhas com NaN em alvo **ou** candidato.  
3. `coverage = pares_válidos / pontos_válidos_do_alvo`.  
4. Exigência de `coverage >= min_coverage` **em todas as variáveis** para manter o candidato.

## Remoção de Outliers
- **Métodos**: `iqr`, `zscore`, `mad`.
- **Escopos**: `pre_agg` (antes da agregação), `post_agg` (depois), `both` (ambos).  
Outliers viram `NaN` e influenciam `coverage`.

## Métricas (modo curva)
Dadas sequências alinhadas x (alvo) e y (candidato):
- **corr** (↑ melhor): correlação de Pearson.
- **rmse** (↓ melhor): raiz do erro quadrático médio.
- **mae** (↓ melhor): erro absoluto médio.
- **dtw** (↓ melhor): Dynamic Time Warping com janela Sakoe–Chiba opcional (`dtw_radius`).

## Cálculo do *score* e ranqueamento
Para cada variável:
1. **Min–max** das métricas respeitando direção (↑/↓).  
2. `score_var = Σ_m (w_m * score_m)` usando `metric_weights` (somam 1).  
Score final por candidato: `score_final = Σ_var (W_var * score_var)` com `var_weights` (somam 1).  
Ordena por `score_final` descrescente e retorna `top_k`.

## Exemplos
```python
# 1) Ano vs. alvo climatológico mensal (médias mensais), multivariável
res = rank_similarity(
    data=df,
    compare_by="year",
    agg_freq="monthly",
    agg_stat="mean",
    compare_mode="curve",
    variables=["pr","temp"],
    normalize="zscore",
    metrics=["corr","rmse","mae","dtw"],
    metric_weights={"corr":0.5,"rmse":0.2,"mae":0.2,"dtw":0.1},
    var_weights={"pr":0.6,"temp":0.4},
    outlier_method="iqr",
    outlier_scope="pre_agg",
    min_coverage=0.7,
    top_k=5,
)
print(res.ranking_summary)
print(res.ranking_long)

# 2) Janeiro por ano (curva diária), alvo automático do mês
res = rank_similarity(
    data=df,
    compare_by="month",
    target_month=1,
    agg_freq="daily",
    agg_stat="mean",
    compare_mode="curve",
    variables=["pr"],
    min_coverage=0.75,
)

# 3) Modo "value": comparar apenas o escalar agregado
res = rank_similarity(
    data=df,
    compare_by="year",
    agg_freq="monthly",
    agg_stat="mean",
    compare_mode="value",
    variables=["pr"],
)
```

## Notas e limitações
- DTW sem janela é O(n·m); use `dtw_radius` para limitar.  
- A DTW implementada usa distância L1 e não faz z-normalização interna.  
- Em dados muito esparsos, ajuste `min_coverage` conforme necessário.
