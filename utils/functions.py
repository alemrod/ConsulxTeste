#!/usr/bin/env python3
"""
calcula_resultados.py

Lê um ou mais arquivos JSON de balancete (hierárquico, contas sintéticas com children),
extrai folhas (contas analíticas) e calcula indicadores financeiros:
- Receita Bruta
- Receita Líquida (Receita Bruta - impostos sobre vendas)
- Lucro Bruto (Receita Líquida - custos)
- Lucro Líquido (Lucro Bruto - despesas)
- Disponibilidade de Caixa (caixa + bancos + aplicações)

"""

import math
from sklearn.metrics import mean_absolute_error, mean_squared_error
from prophet import Prophet
import numpy as np
from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime
import pandas as pd
import re



# Extrator de balancete para dataframe
def extract_accounts(node, hierarchy=None):
    if hierarchy is None:
        hierarchy = []

    current_hierarchy = hierarchy + [node["descricao"]]
    rows = []

    if "children" in node:
        for child in node["children"]:
            rows.extend(extract_accounts(child, current_hierarchy))
    else:
        row = {f"nivel_{i+1}": level for i,
               level in enumerate(current_hierarchy)}
        row["conta"] = node["conta"]
        row["descricao"] = node["descricao"]
        row["saldo_atual"] = node.get("saldo_atual", 0.0)
        rows.append(row)

    return rows


def extract_mes_from_periodo(periodo_str):
    """
    Recebe strings como '01/01/2023 - 31/01/2023' ou '31/01/2023'
    e retorna 'YYYY-MM' (ex: '2023-01').
    """
    if not periodo_str:
        return None
    # encontra todas as datas no formato dd/mm/yyyy
    dates = re.findall(r'\d{2}/\d{2}/\d{4}', periodo_str)
    if not dates:
        return None
    # preferir a segunda data (fim do período) se existir, senão a primeira
    date_str = dates[1] if len(dates) > 1 else dates[0]
    try:
        dt = datetime.strptime(date_str, "%d/%m/%Y")
        return f"{dt.year}-{dt.month:02d}"
    except Exception:
        return None


def processar_indicadores_financeiros(df: pd.DataFrame) -> pd.DataFrame:
    """
    Processa o DataFrame de dados financeiros e retorna uma tabela com indicadores
    financeiros calculados.
    
    Args:
        df (pd.DataFrame): DataFrame contendo os dados financeiros
        
    Returns:
        pd.DataFrame: Tabela com indicadores financeiros calculados
    """
    # Filtragem dos dados básicos
    receita_bruta = df[(df["nivel_1"] == "RECEITAS") &
                       (df["nivel_2"] == "Serviços Prestados a Prazo")]
    impostos_sobre_receita = df[(df["nivel_1"] == "RECEITAS") &
                                (df["nivel_2"] == "Simples Nacional sobre vendas e serviços")]
    custos_despesas = df[df["nivel_1"] == "CUSTOS E DESPESAS"]
    caixa = df[(df["nivel_2"] == "ATIVO CIRCULANTE") &
               (df["nivel_3"] == "DISPONIBILIDADES")]

    # Agregação mensal dos dados
    receita_bruta_mensal = receita_bruta.groupby("mes")["saldo_atual"].sum()
    impostos_mensal = impostos_sobre_receita.groupby("mes")[
        "saldo_atual"].sum()
    custos_mensal = custos_despesas.groupby("mes")["saldo_atual"].sum()
    caixa_mensal = caixa.groupby("mes")["saldo_atual"].sum()

    # Criação da tabela de indicadores
    df_indices = pd.DataFrame({
        "Receita_Bruta": receita_bruta_mensal,
        "Impostos_Receita": impostos_mensal,
        "Custo_Total": custos_mensal,
        "Disponibilidade_Caixa": caixa_mensal
    })

    # Cálculos de indicadores
    df_indices["Receita_Líquida"] = df_indices["Receita_Bruta"] + \
        df_indices["Impostos_Receita"]
    df_indices["Lucro_Bruto"] = df_indices["Receita_Líquida"] - \
        df_indices["Custo_Total"]
    df_indices["Lucro_Líquido"] = df_indices["Lucro_Bruto"] 

    # Processamento adicional para tabela pivot
    df_agrupado = df.groupby(['nivel_2', 'mes'], as_index=False)[
        'saldo_atual'].sum()
    filtro = df_agrupado.loc[
        df_agrupado['nivel_2'].isin(
            ['ATIVO CIRCULANTE', 'PASSIVO CIRCULANTE',
             'ATIVO NÃO CIRCULANTE', 'PASSIVO NÃO CIRCULANTE', 'PATRIMÔNIO LÍQUIDO'])
    ]

    # Criação da tabela pivot
    tabela_pivot = pd.pivot_table(
        filtro,
        values='saldo_atual',
        index='mes',
        columns='nivel_2',
        aggfunc='first'
    )

    # Mesclagem com indicadores
    tabela_pivot = pd.merge(
        tabela_pivot, df_indices, on='mes', how='inner'
    )

    # Cálculos finais
    tabela_pivot['Ativo_Total'] = tabela_pivot['ATIVO CIRCULANTE'] + \
        tabela_pivot['ATIVO NÃO CIRCULANTE']
    tabela_pivot['Passivo_Total'] = tabela_pivot['PASSIVO CIRCULANTE'] + \
        tabela_pivot['PASSIVO NÃO CIRCULANTE']

    # Indicadores de liquidez
    tabela_pivot['Liquidez_Corrente'] = tabela_pivot['ATIVO CIRCULANTE'] / \
        tabela_pivot['PASSIVO CIRCULANTE']
    tabela_pivot['Liquidez_Imediata'] = tabela_pivot['Disponibilidade_Caixa'] / \
        tabela_pivot['PASSIVO CIRCULANTE']
    tabela_pivot['Liquidez_Geral'] = (tabela_pivot['ATIVO CIRCULANTE'] +
                                      tabela_pivot['ATIVO NÃO CIRCULANTE']) / \
        (tabela_pivot['PASSIVO CIRCULANTE'] +
         tabela_pivot['PASSIVO NÃO CIRCULANTE'])

    # Indicadores de solvência
    tabela_pivot['Solvencia_Geral'] = tabela_pivot['Ativo_Total'] / \
        tabela_pivot['Passivo_Total']
    tabela_pivot['Endividamento'] = tabela_pivot['Passivo_Total'] / \
        tabela_pivot['Ativo_Total']
    tabela_pivot['Endividamento_Geral'] = tabela_pivot['Endividamento']
    
    tabela_pivot["Margem_de_Lucro"] = (
        tabela_pivot["Lucro_Líquido"] / tabela_pivot["Receita_Líquida"])
    
    tabela_pivot["Retorno_Sobre_Patrimonio_Liquido"] = (
        tabela_pivot["Lucro_Líquido"] / tabela_pivot["PATRIMÔNIO LÍQUIDO"])
    
    tabela_pivot = tabela_pivot.rename(
        columns={"ATIVO CIRCULANTE": "Ativo_Circulante", "ATIVO NÃO CIRCULANTE": "Ativo_Nao_Circulante", "PASSIVO CIRCULANTE": "Passivo_Circulante", "PASSIVO NÃO CIRCULANTE": "Passivo_Nao_Circulante", "PATRIMÔNIO LÍQUIDO": "Patrimonio_Liquido"})

    return tabela_pivot


def prophet_ar2_forecast(df, target_col, horizon=6, yearly_seasonality=True):
    """
    Aplica Prophet com regressão autoregressiva (AR(2)) para previsão univariada.
    
    Parâmetros:
    -----------
    df : pd.DataFrame
        DataFrame contendo uma coluna datetime (índice ou coluna nomeada 'mes', 'data', etc.) e a série alvo.
    target_col : str
        Nome da coluna alvo (ex: 'Lucro_Líquido').
    horizon : int, opcional
        Quantidade de períodos futuros a prever (default=6).
    yearly_seasonality : bool, opcional
        Ativa sazonalidade anual (default=True).
    
    Retorna:
    --------
    dict :
        {'MAE': valor, 'RMSE': valor, 'MAPE': valor, 'forecast_df': DataFrame}
    """

    # --- Preparação ---
    s = df[target_col].astype(float).dropna()
    # tenta detectar coluna de datas
    if df.index.inferred_type == "datetime64":
        s.index = pd.to_datetime(s.index)
    elif 'ds' in df.columns:
        s.index = pd.to_datetime(df['ds'])
    elif 'mes' in df.columns:
        s.index = pd.to_datetime(df['mes'])
    else:
        raise ValueError(
            "⚠️ O DataFrame precisa ter um índice datetime ou uma coluna de datas ('ds' ou 'mes').")

    data = pd.DataFrame({'ds': s.index, 'y': s.values}).reset_index(drop=True)

    # criar lags
    data['lag1'] = data['y'].shift(1)
    data['lag2'] = data['y'].shift(2)
    data = data.dropna().reset_index(drop=True)

    # Split treino/test
    h = horizon
    train = data.iloc[:-h].copy()
    test = data.iloc[-h:].copy()

    # Treinar Prophet com regressors
    m = Prophet(yearly_seasonality=yearly_seasonality, daily_seasonality=False)
    m.add_regressor('lag1')
    m.add_regressor('lag2')
    m.fit(train[['ds', 'y', 'lag1', 'lag2']])

    # --- Previsão recursiva ---
    last_row = train.iloc[-1].copy()
    preds, pred_dates = [], []

    lag1 = float(last_row['y'])
    lag2 = float(last_row['lag1'])

    for i in range(h):
        next_ds = test['ds'].iloc[i]
        df_next = pd.DataFrame(
            {'ds': [next_ds], 'lag1': [lag1], 'lag2': [lag2]})
        forecast_next = m.predict(df_next)
        yhat = float(forecast_next['yhat'].iloc[0])
        preds.append(yhat)
        pred_dates.append(next_ds)
        lag2 = lag1
        lag1 = yhat

    # --- Avaliação ---
    y_true = test['y'].values
    y_pred = np.array(preds)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    # montar DataFrame de resultados
    results = pd.DataFrame(
        {'ds': pred_dates, 'y_true': y_true, 'y_pred': y_pred})

    # imprimir resultados
    print(f"📈 AR(2) via Prophet - coluna: {target_col}")
    print(f"MAE :  {mae:.4f}")
    print(f"RMSE:  {rmse:.4f}")
    print(f"MAPE:  {mape:.2f}%")

    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'forecast_df': results
    }


def forecast_future_periods(df, target_col, horizon=6, yearly_seasonality=True):
    """
    Usa Prophet com AR(2) para prever meses futuros após o último registro da série.
    """
    s = df[target_col].astype(float).dropna()

    # Detecta e converte o índice de data
    if df.index.inferred_type != "datetime64":
        if 'ds' in df.columns:
            s.index = pd.to_datetime(df['ds'])
        elif 'mes' in df.columns:
            s.index = pd.to_datetime(df['mes'])
        else:
            raise ValueError(
                "⚠️ O DataFrame precisa ter índice datetime ou coluna ('ds' ou 'mes').")

    data = pd.DataFrame({'ds': s.index, 'y': s.values}).reset_index(drop=True)
    data['lag1'] = data['y'].shift(1)
    data['lag2'] = data['y'].shift(2)
    data = data.dropna().reset_index(drop=True)

    # Treinar o modelo em todos os dados disponíveis
    m = Prophet(yearly_seasonality=yearly_seasonality, daily_seasonality=False)
    m.add_regressor('lag1')
    m.add_regressor('lag2')
    m.fit(data[['ds', 'y', 'lag1', 'lag2']])

    # Começar previsão a partir do último ponto conhecido
    last_row = data.iloc[-1].copy()
    lag1 = float(last_row['y'])
    lag2 = float(last_row['lag1'])
    last_date = data['ds'].iloc[-1]

    preds, dates = [], []

    for i in range(1, horizon + 1):
        next_date = last_date + pd.DateOffset(months=i)
        df_next = pd.DataFrame(
            {'ds': [next_date], 'lag1': [lag1], 'lag2': [lag2]})
        forecast_next = m.predict(df_next)
        yhat = float(forecast_next['yhat'].iloc[0])
        preds.append(yhat)
        dates.append(next_date)
        # Atualiza os lags
        lag2 = lag1
        lag1 = yhat

    forecast_df = pd.DataFrame({'ds': dates, 'forecast': preds})
    print("✅ Previsão futura concluída com sucesso!")
    return forecast_df
