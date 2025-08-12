# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 11:21:07 2025

@author: EDSON.DOSSANTOS
"""
#%% faz o código rodar na pasta raiz do arquivo
import os 
script_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_directory)
#%%
import warnings

# Suprimir todos os warnings
warnings.filterwarnings('ignore')

#%%
from rank_similarity import rank_similarity
import pandas as pd
import numpy as np
#%%

df = pd.read_csv('vento_combinado_todos_meses.csv', sep=',')

df = df[(df['longitude'] == -49.50) & (df['latitude'] == -33.50)]
df['valid_time'] = pd.to_datetime(df['valid_time'])
df.set_index('valid_time', inplace=True)
df = df[df.index >= '1995-01-01']
df = df[['vento_u10', 'vento_v10']]
df.columns = ['u', 'v']
#Velocidade do Vento

df['vvel'] = np.sqrt(df['u']**2 + df['v']**2)

#%%


res = rank_similarity(
    data=df,
    compare_by="year", # ou "month" (se "month", passe target_month=1..12)
    agg_freq="monthly", # "monthly" | "daily" | "hourly"
    agg_stat="mean", # "mean" | "min" | "max"
    compare_mode="curve", # ou "value"
    variables=["u","v","vvel"], # 1+ variáveis
    normalize="none", # "none" | "zscore" | "minmax"
    metrics=["corr","rmse","mae","dtw"],
    metric_weights={"corr":0.35,"rmse":0.35,"mae":0.2,"dtw":0.1}, # distribuição [0..1]
    var_weights={"u":0.35,"v":0.35,"vvel":0.3}, # distribuição [0..1]
    dtw_radius=1,
    outlier_method="iqr", # None | "iqr" | "zscore" | "mad"
    outlier_scope="pre_agg", # "none" | "pre_agg" | "post_agg" | "both"
    iqr_factor=1.5,
    outlier_min_points=12,
    min_coverage=0.80, # 70% exigidos não nulos
    align_how="left_target",
    top_k=5, # Pegao top 5
)

#%%
res.ranking_summary   # ranking final por candidato
res.ranking_long      # por candidato x variável, com métricas, scores e rank 
