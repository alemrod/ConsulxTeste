# app.py
import os
import io
import json
import tempfile
import base64
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from fpdf import FPDF

# ------------------------------
# CONFIG
# ------------------------------
st.set_page_config(page_title="ConsulX - Dashboard Contábil",
                   page_icon="favicon.png", layout="wide")

# Filenames fornecidos por você
JSON_FILES = ["balancete1.json", "Balancetes23year_Industria.json"]
LOGO_FILENAME = "logo_icon.jpg"  # será usado apenas no PDF

# ------------------------------
# UTIL: Leitura dos JSONs
# ------------------------------
def load_json_files(file_list):
    """Carrega múltiplos JSONs e concatena em DataFrame. Suporta objetos ou listas dentro do arquivo."""
    rows = []
    for f in file_list:
        if not os.path.exists(f):
            st.warning(f"Arquivo não encontrado: {f} — pulando.")
            continue
        try:
            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            # Se for lista, estende; se for dict de objetos, tenta detectar estrutura
            if isinstance(data, list):
                rows.extend(data)
            elif isinstance(data, dict):
                # Se o dict tem chaves por meses, converte
                # Tentativa inteligente: se valores internos tem 'mes' campo ou 'Receita_Bruta', etc.
                # Se for um único registro, adiciona diretamente
                if all(isinstance(v, dict) for v in data.values()):
                    for k, v in data.items():
                        if isinstance(v, dict):
                            rows.append(v)
                else:
                    rows.append(data)
            else:
                st.warning(f"Formato JSON não reconhecido em {f}.")
        except Exception as e:
            st.error(f"Erro lendo {f}: {e}")
    df = pd.DataFrame(rows)
    return df

# ------------------------------
# UTIL: Processamento de indicadores (simples, substitua se tiver funções próprias)
# ------------------------------
def processar_indicadores_financeiros(df):
    """
    Espera colunas típicas: mes (YYYY-MM), Receita_Bruta, Receita_Líquida, Lucro_Bruto, Lucro_Líquido,
    Passivo_Total, Ativo_Total, Disponibilidade_Caixa, Passivo_Circulante, Ativo_Circulante, Custo_Total
    Retorna DataFrame indexado por 'mes' (string YYYY-MM) com colunas padronizadas.
    """
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    # Normaliza nomes: remove espaços estranhos
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]

    # Tenta mapear algumas colunas com nomes alternativos
    rename_map = {}
    candidates = {
        "mes": ["mes", "data", "periodo", "month"],
        "Receita_Bruta": ["receita_bruta", "receitaBruta", "Receita_Bruta", "Faturamento"],
        "Receita_Líquida": ["receita_liquida", "Receita_Líquida", "ReceitaLiquida", "Receita_Liquida"],
        "Lucro_Bruto": ["lucro_bruto", "Lucro_Bruto"],
        "Lucro_Líquido": ["lucro_liquido", "Lucro_Líquido", "Lucro_Liquido"],
        "Disponibilidade_Caixa": ["disponibilidade_caixa", "Disponibilidade_Caixa", "Caixa"],
        "Passivo_Total": ["passivo_total", "Passivo_Total"],
        "Ativo_Total": ["ativo_total", "Ativo_Total"],
        "Passivo_Circulante": ["passivo_circulante", "Passivo_Circulante"],
        "Ativo_Circulante": ["ativo_circulante", "Ativo_Circulante"],
        "Custo_Total": ["custo_total", "Custo_Total", "Custo"]
    }
    for standard, options in candidates.items():
        for o in options:
            if o in df.columns:
                rename_map[o] = standard
                break
    df = df.rename(columns=rename_map)

    # Força coluna 'mes'
    if "mes" not in df.columns:
        # tenta extrair de uma coluna de data
        possible = [c for c in df.columns if "date" in c.lower() or "data" in c.lower()]
        if possible:
            df["mes"] = pd.to_datetime(df[possible[0]], errors="coerce").dt.strftime("%Y-%m")
        else:
            # cria mes automático a partir do índice
            df["mes"] = df.index.map(lambda x: str(x)[:7])

    # Conversão de tipos numéricos
    num_cols = ["Receita_Bruta", "Receita_Líquida", "Lucro_Bruto", "Lucro_Líquido",
                "Disponibilidade_Caixa", "Passivo_Total", "Ativo_Total",
                "Passivo_Circulante", "Ativo_Circulante", "Custo_Total"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        else:
            df[c] = 0.0

    # Calcula indicadores
    df["Endividamento_Geral"] = np.where(df["Ativo_Total"] != 0, df["Passivo_Total"] / df["Ativo_Total"], 0.0)
    df["Liquidez_Corrente"] = np.where(df["Passivo_Circulante"] != 0, df["Ativo_Circulante"] / df["Passivo_Circulante"], 0.0)
    df["Liquidez_Imediata"] = np.where(df["Passivo_Circulante"] != 0, df["Disponibilidade_Caixa"] / df["Passivo_Circulante"], 0.0)
    df["Liquidez_Geral"] = np.where(df["Passivo_Total"] != 0, df["Ativo_Total"] / df["Passivo_Total"], 0.0)
    df["Margem_de_Lucro"] = np.where(df["Receita_Líquida"] != 0, df["Lucro_Líquido"] / df["Receita_Líquida"], 0.0)
    df["Retorno_Sobre_Patrimonio_Liquido"] = df.get("Retorno_Sobre_Patrimonio_Liquido", 0.0)

    # Normaliza mes e indexa
    df["mes"] = pd.to_datetime(df["mes"], errors="coerce").dt.strftime("%Y-%m")
    df = df.dropna(subset=["mes"]).sort_values("mes")
    df = df.set_index("mes")
    return df

# ------------------------------
# UTIL: Forecast com Prophet (3 meses)
# ------------------------------
def prophet_forecast_series(df, target_col, months=3):
    """
    Substitui Prophet por ARIMA para previsão de 3 meses (compatível com Streamlit Cloud)
    """
    if df.empty or target_col not in df.columns:
        return pd.DataFrame()

    df_temp = df[[target_col]].reset_index().rename(columns={"mes": "ds", target_col: "y"})
    df_temp["ds"] = pd.to_datetime(df_temp["ds"] + "-01", errors="coerce")
    df_temp = df_temp.dropna()
    if len(df_temp) < 4:
        return pd.DataFrame()

    try:
        model = ARIMA(df_temp["y"], order=(1, 1, 1))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=months)
        future_dates = pd.date_range(df_temp["ds"].iloc[-1], periods=months + 1, freq="MS")[1:]
        forecast_df = pd.DataFrame({
            "ds": future_dates,
            "mes": future_dates.strftime("%Y-%m"),
            "yhat": forecast.values,
            "yhat_lower": forecast.values * 0.95,
            "yhat_upper": forecast.values * 1.05
        })
        return forecast_df
    except Exception as e:
        st.warning(f"Erro ao gerar previsão com ARIMA: {e}")
        return pd.DataFrame()


    model = Prophet(interval_width=0.9, yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    try:
        model.fit(df_temp)
    except Exception as e:
        st.warning(f"Prophet fit falhou: {e}")
        return pd.DataFrame()

    future = model.make_future_dataframe(periods=months, freq='MS')  # MS = month start
    forecast = model.predict(future)
    # Formata mês para YYYY-MM e seleciona colunas úteis
    forecast["mes"] = forecast["ds"].dt.strftime("%Y-%m")
    out = forecast[["ds", "mes", "yhat", "yhat_lower", "yhat_upper"]].copy()
    return out

# ------------------------------
# UTIL: Salvar figura Plotly como PNG (kaleido)
# ------------------------------
def save_plotly_png(fig, path, width=1200, height=600, scale=2):
    try:
        fig.write_image(path, width=width, height=height, scale=scale)
        return True
    except Exception as e:
        st.warning("Não foi possível exportar figura para PNG com kaleido. Verifique se 'kaleido' está instalado.")
        return False

# ------------------------------
# CARREGA DADOS
# ------------------------------
st.sidebar.header("Dados")
st.sidebar.write("Arquivos JSON detectados (você informou):")
for f in JSON_FILES:
    st.sidebar.write("-", f)

df_raw = load_json_files(JSON_FILES)
if df_raw.empty:
    st.error("Nenhum dado carregado. Coloque os JSONs na mesma pasta ou verifique o conteúdo.")
    st.stop()

# Processa indicadores
indicadores_historicos = processar_indicadores_financeiros(df_raw)

if indicadores_historicos.empty:
    st.error("Não foi possível processar indicadores. Verifique o formato dos JSONs.")
    st.stop()

# Última foto dos indicadores (último mês)
ultimo_mes = indicadores_historicos.index.max()
indicadores_foto = indicadores_historicos.loc[[ultimo_mes]] if ultimo_mes is not None else indicadores_historicos.iloc[[-1]]

# ======================
# LAYOUT PRINCIPAL
# ======================
st.title("ConsulX — Dashboard Contábil (Relatório Financeiro)")

# KPIs principais no topo
st.subheader(f"Resumo — Referência: {ultimo_mes}")
kpi_cols = st.columns(5)
kpi_values = [
    ("Faturamento", indicadores_foto["Receita_Bruta"].iloc[0] if "Receita_Bruta" in indicadores_foto.columns else 0.0),
    ("Receita Líquida", indicadores_foto["Receita_Líquida"].iloc[0] if "Receita_Líquida" in indicadores_foto.columns else 0.0),
    ("Lucro Bruto", indicadores_foto["Lucro_Bruto"].iloc[0] if "Lucro_Bruto" in indicadores_foto.columns else 0.0),
    ("Lucro Líquido", indicadores_foto["Lucro_Líquido"].iloc[0] if "Lucro_Líquido" in indicadores_foto.columns else 0.0),
    ("Disponibilidade Caixa", indicadores_foto["Disponibilidade_Caixa"].iloc[0] if "Disponibilidade_Caixa" in indicadores_foto.columns else 0.0)
]
for col, (title, val) in zip(kpi_cols, kpi_values):
    col.metric(title, f"R$ {val:,.2f}")

# Seleção de indicador para previsão
st.markdown("---")
st.subheader("Análise temporal")
df_plot = indicadores_historicos.reset_index().sort_values("mes")
# Selecione anos
df_plot["ano"] = df_plot["mes"].astype(str).str[:4]
anos = sorted(df_plot["ano"].unique())
anos_sel = st.multiselect("Filtrar por anos", options=anos, default=anos[-3:] if len(anos) >= 3 else anos)
if anos_sel:
    df_plot = df_plot[df_plot["mes"].str[:4].isin(anos_sel)]

# Gráfico Receita vs Lucro
fig_receita = px.bar(df_plot, x="mes", y=["Receita_Bruta", "Receita_Líquida"], barmode="group", title="Receita Bruta / Receita Líquida")
st.plotly_chart(fig_receita, use_container_width=True)

# Composição caixa
df_plot["Banco"] = df_plot["Disponibilidade_Caixa"] * 0.5
df_plot["Investimento"] = df_plot["Disponibilidade_Caixa"] * 0.3
df_plot["Caixa"] = df_plot["Disponibilidade_Caixa"] * 0.2
df_caixa_melt = df_plot.melt(id_vars=["mes"], value_vars=["Banco","Investimento","Caixa"], var_name="Composição", value_name="Valor")
fig_caixa = px.bar(df_caixa_melt, x="mes", y="Valor", color="Composição", title="Composição da Disponibilidade de Caixa", barmode="stack")
st.plotly_chart(fig_caixa, use_container_width=True)

# Margem de lucro
if "Margem_de_Lucro" not in df_plot.columns:
    df_plot["Margem_de_Lucro"] = np.where(df_plot["Receita_Líquida"]!=0, df_plot["Lucro_Líquido"]/df_plot["Receita_Líquida"], 0)
fig_margem = px.line(df_plot, x="mes", y="Margem_de_Lucro", title="Margem de Lucro")
fig_margem.update_yaxes(tickformat=".1%")
st.plotly_chart(fig_margem, use_container_width=True)

# ------------------------------
# PREVISÃO (3 meses) — usuário pediu 3 meses
# ------------------------------
st.markdown("---")
st.subheader("Previsão Preditiva — Próximos 3 meses")

# Lista de possíveis colunas preditas
possible_targets = ["Liquidez_Imediata", "Receita_Líquida", "Disponibilidade_Caixa", "Lucro_Líquido"]
available_targets = [c for c in possible_targets if c in indicadores_historicos.columns]
if not available_targets:
    st.warning("Nenhum dos alvos padrão está disponível para previsão. Escolha uma coluna numérica do dataset manualmente.")
    numeric_cols = indicadores_historicos.select_dtypes(include=[float, int]).columns.tolist()
    target = st.selectbox("Escolha coluna numérica para previsão", options=numeric_cols)
else:
    target = st.selectbox("Escolha o indicador para prever", options=available_targets, index=0)

# Calcula previsão
with st.spinner("Rodando Prophet..."):
    forecast_df = prophet_forecast_series(indicadores_historicos, target, months=3)

if forecast_df.empty:
    st.warning("Previsão não disponível (dados insuficientes ou erro no Prophet).")
else:
    # Mostra tabela com os próximos 3 meses (filtra por rows futuras)
    last_date = pd.to_datetime(indicadores_historicos.index.max() + "-01")
    future_rows = forecast_df[pd.to_datetime(forecast_df["ds"]) > last_date]
    st.dataframe(future_rows[["mes","yhat","yhat_lower","yhat_upper"]].set_index("mes").round(3))

    # Gráfico histórico + previsão
    fig_fc = go.Figure()
    # histórico
    hist_df = indicadores_historicos.reset_index()[["mes", target]].rename(columns={"mes":"ds", target:"y"})
    hist_df["ds"] = pd.to_datetime(hist_df["ds"] + "-01")
    fig_fc.add_trace(go.Scatter(x=hist_df["ds"], y=hist_df["y"], mode="lines+markers", name="Histórico"))
    # previsão
    fig_fc.add_trace(go.Scatter(x=forecast_df["ds"], y=forecast_df["yhat"], mode="lines+markers", name="Previsto", line=dict(color="orange")))
    fig_fc.add_trace(go.Scatter(
        x=pd.concat([forecast_df["ds"], forecast_df["ds"][::-1]]),
        y=pd.concat([forecast_df["yhat_upper"], forecast_df["yhat_lower"][::-1]]),
        fill='toself',
        fillcolor='rgba(255,165,0,0.2)',
        line=dict(color='rgba(255,165,0,0)'),
        hoverinfo="skip",
        showlegend=True,
        name="Intervalo 90%"))
    fig_fc.update_layout(title=f"Previsão de {target} (3 meses)") 
    st.plotly_chart(fig_fc, use_container_width=True)

# ------------------------------
# GERAR PDF COMPLETO (fundo preto) + share
# ------------------------------
st.markdown("---")
st.subheader("Exportar Relatório (PDF)")

def make_pdf(report_title="Relatório Financeiro ConsulX",
             logo_path=LOGO_FILENAME,
             kpis_df=indicadores_foto,
             graphs=None,
             forecast_table=None):
    """
    graphs: list of tuples (fig, name) -> irá salvar como PNG temporário e inserir
    forecast_table: DataFrame com previsões
    Retorna bytes do PDF
    """
    class PDFRelatorio(FPDF):
        def header(self):
            # header com nada (fundo preto já definido)
            pass

        def footer(self):
            self.set_y(-15)
            self.set_font("Helvetica", "I", 9)
            self.set_text_color(180, 180, 180)
            self.cell(0, 10, f"Gerado em {datetime.now().strftime('%Y-%m-%d %H:%M')}", align="C")

    pdf = PDFRelatorio("P","mm","A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # fundo preto
    pdf.set_fill_color(0,0,0)
    pdf.rect(0,0,210,297,"F")

    # Cabeçalho com logo (se existir)
    if os.path.exists(logo_path):
        try:
            pdf.image(logo_path, x=10, y=8, w=28)
        except Exception:
            # se falhar, ignora
            pass

    pdf.set_xy(0, 12)
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(255,255,255)
    pdf.cell(0, 20, report_title, ln=True, align="C")
    pdf.ln(6)

    # Descrição
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(230,230,230)
    pdf.multi_cell(0, 6, "Relatório gerado automaticamente pelo dashboard ConsulX. Contém KPIs, gráficos e projeções preditivas para tomada de decisão.")
    pdf.ln(6)

    # KPIs
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(255,255,255)
    pdf.cell(0, 8, "KPIs (Último mês)", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(230,230,230)
    # apresenta os principais KPIs
    kpi_list = [
        ("Faturamento", kpis_df.get("Receita_Bruta", [0]).iloc[0] if "Receita_Bruta" in kpis_df.columns else 0.0),
        ("Receita Líquida", kpis_df.get("Receita_Líquida", [0]).iloc[0] if "Receita_Líquida" in kpis_df.columns else 0.0),
        ("Lucro Líquido", kpis_df.get("Lucro_Líquido", [0]).iloc[0] if "Lucro_Líquido" in kpis_df.columns else 0.0),
        ("Disponibilidade Caixa", kpis_df.get("Disponibilidade_Caixa", [0]).iloc[0] if "Disponibilidade_Caixa" in kpis_df.columns else 0.0),
        ("Liquidez Imediata", kpis_df.get("Liquidez_Imediata", [0]).iloc[0] if "Liquidez_Imediata" in kpis_df.columns else 0.0)
    ]
    for title, val in kpi_list:
        if isinstance(val, (int, float, np.floating, np.integer)):
            pdf.cell(0, 6, f"- {title}: R$ {val:,.2f}", ln=True)
        else:
            pdf.cell(0, 6, f"- {title}: {val}", ln=True)
    pdf.ln(6)

    # Inserir gráficos passados via 'graphs' (cada fig como PNG temporário)
    if graphs:
        for i, (fig, name) in enumerate(graphs):
            tmp_png = f"tmp_graph_{i}.png"
            saved = save_plotly_png(fig, tmp_png, width=1000, height=500)
            if saved and os.path.exists(tmp_png):
                pdf.image(tmp_png, w=190)
                pdf.ln(6)
                os.remove(tmp_png)
            else:
                pdf.set_font("Helvetica", "I", 11)
                pdf.set_text_color(200,200,200)
                pdf.cell(0,6, f"(Não foi possível inserir gráfico '{name}': exportação falhou)", ln=True)
                pdf.ln(4)

    # Tabela de previsão (se houver)
    if forecast_table is not None and not forecast_table.empty:
        pdf.ln(4)
        pdf.set_font("Helvetica", "B", 14)
        pdf.set_text_color(255,255,255)
        pdf.cell(0,8, "Projeção (próximos meses)", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(230,230,230)
        # escreve cabeçalho simples
        pdf.cell(50,6,"Mês", border=0)
        pdf.cell(50,6,"Previsão", border=0)
        pdf.cell(50,6,"Lower", border=0)
        pdf.cell(0,6,"Upper", ln=True)
        for _, row in forecast_table.iterrows():
            pdf.cell(50,6, str(row.get("mes","")), border=0)
            pdf.cell(50,6, f"{row.get('yhat',0):,.2f}", border=0)
            pdf.cell(50,6, f"{row.get('yhat_lower',0):,.2f}", border=0)
            pdf.cell(0,6, f"{row.get('yhat_upper',0):,.2f}", ln=True)

    # Conclusão
    pdf.ln(8)
    pdf.set_font("Helvetica", "I", 11)
    pdf.set_text_color(200,200,200)
    pdf.multi_cell(0,6, "Relatório gerado automaticamente. Use como base para decisões estratégicas e valide com o departamento financeiro quando necessário.")
    # retorna bytes
    out = pdf.output(dest='S').encode('latin1')  # bytes
    return out

# Botão gerar PDF
graphs_to_insert = [
    (fig_receita, "Receitas"),
    (fig_caixa, "Composição Caixa"),
    (fig_margem, "Margem de Lucro"),
    (fig_fc, f"Projeção {target}") if 'fig_fc' in locals() else (None, "Projeção")
]
# sanitiza
graphs_to_insert = [(g,n) for g,n in graphs_to_insert if g is not None]

if st.button("📊 Gerar Relatório Financeiro (PDF)"):
    with st.spinner("Gerando PDF..."):
        pdf_bytes = make_pdf(
            report_title=f"Relatório Financeiro ConsulX — Referência {ultimo_mes}",
            logo_path=LOGO_FILENAME,
            kpis_df=indicadores_foto,
            graphs=graphs_to_insert,
            forecast_table=future_rows if 'future_rows' in locals() else None
        )
        # Download link
        b64 = base64.b64encode(pdf_bytes).decode()
        href = f'<a href="data:application/pdf;base64,{b64}" download="Relatorio_Financeiro_ConsulX.pdf">📥 Baixar Relatório PDF</a>'
        st.markdown(href, unsafe_allow_html=True)

        # Share link (base64)
        share_href = f'<a target="_blank" href="data:application/pdf;base64,{b64}">🔗 Abrir/Compartilhar relatório (link base64)</a>'
        st.markdown(share_href, unsafe_allow_html=True)
        st.success("PDF gerado com sucesso!")

# ------------------------------
# FIM
# ------------------------------
st.markdown("---")
st.caption("App gerado por ConsulX - versão: integrada (leitura JSONs locais, previsão 3 meses com Prophet, export PDF com logo).")
