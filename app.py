import tempfile
from fpdf import FPDF
from prophet import Prophet
import streamlit as st
import plotly.express as px
import base64

import pandas as pd
import os
import json
from utils.functions import processar_indicadores_financeiros, prophet_ar2_forecast, forecast_future_periods
from utils.db import load_all_rows_from_mongo
import plotly.graph_objects as go
# ======================
# CONFIGURAÇÕES GERAIS
# ======================
st.set_page_config(page_title="ConsulX - Dashboard Contábil",
                   page_icon="favicon.png", layout="wide")
# ======================== LÊ TODOS OS BALANCETES TEMPORAIS ========================

# ======================
# DB: conexão + leitura com cache
# ======================
with st.spinner("Carregando dados do MongoDB (isso pode demorar na primeira vez)..."):
    # Para debug/primeiro deploy: limite para evitar timeout (remova o limit em produção quando estiver seguro)
    all_rows = load_all_rows_from_mongo(limit=None)

# Cria DataFrame de forma segura
if not all_rows:
    st.warning(
        "Nenhum documento/processamento retornou dados. Verifique a coleção ou o extractor.")
    df_hist = pd.DataFrame()  # DataFrame vazio para evitar crashes
else:
    df_hist = pd.DataFrame(all_rows)

# Processa indicadores apenas se houver dados
if df_hist.empty:
    indicadores_historicos = pd.DataFrame()
    indicadores_foto = pd.DataFrame()
else:
    indicadores_historicos = processar_indicadores_financeiros(df_hist)
    ultimo_mes = indicadores_historicos.index.max()
    if pd.isna(ultimo_mes):
        indicadores_foto = pd.DataFrame()
    else:
        indicadores_foto = indicadores_historicos.loc[indicadores_historicos.index == ultimo_mes]

# Proteção ao pegar valores (evita IndexError que quebra a renderização)


def safe_get(ind_df, col):
    try:
        return float(ind_df[col].values[0])
    except Exception:
        return 0.0

# ======================
# FUNÇÃO DE FILTRO DE ANO
# ======================
def filtro_ano(df_plot):
    """
    Cria um seletor de ano no Streamlit e retorna o DataFrame filtrado.
    O campo 'mes' deve estar no formato 'YYYY-MM'.
    """
    # Extrair anos únicos ordenados
    df_plot["ano"] = df_plot["mes"].astype(str).str[:4]
    anos = sorted(df_plot["ano"].unique())

    # Caixa visual estilizada
    st.markdown("""
    <style>
        .year-filter-box {
            background-color: #ffffff;
            border-radius: 10px;
            box-shadow: 0 3px 10px rgba(0, 0, 0, 0.05);
            padding: 12px 20px;
            margin-bottom: 25px;
        }
        .year-filter-title {
            font-size: 15px;
            font-weight: 600;
            color: #333333;
            margin-bottom: 8px;
        }
    </style>
    """, unsafe_allow_html=True)


    # Filtro de ano (pode selecionar um ou vários)
    ano_selecionado = st.multiselect(
        "Selecione o(s) ano(s):",
        options=anos,
        default=anos[-1:],  # último ano por padrão
        label_visibility="collapsed"
    )

    st.markdown('</div>', unsafe_allow_html=True)

    # Filtra o DataFrame conforme o(s) ano(s) escolhidos
    df_filtrado = df_plot[df_plot["mes"].str[:4].isin(ano_selecionado)]

    return df_filtrado, ano_selecionado    






# Dados de exemplo
indicadores = [
    {
        "titulo": "Endividamento Geral",
        "descricao": "Grau em que os Ativos Totais são financiados por recursos de terceiros.",
        "memoria": "Passivo Total / Ativo Total",
        "valor": round(indicadores_foto['Endividamento_Geral'].values[0], 2),
        "tooltip": "Quanto mais perto de 0, melhor. Indica menor dependência de capital de terceiros."
    },
    {
        "titulo": "Margem Líquida de Lucro",
        "descricao": "Indica quanto de lucro líquido a empresa obtém para cada real de vendas (ou prestação de serviços).",
        "memoria": "Lucro Líquido / Receita Líquida",
        "valor": round(indicadores_foto['Margem_de_Lucro'].values[0], 2),
        "tooltip": "Quanto maior, melhor. Indica a eficiência da empresa em gerar lucro sobre a receita."
    },
    {
        "titulo": "ROE",
        "descricao": "Retorno Sobre Patrimônio Líquido - Grau de rentabilidade sobre o capital próprio.",
        "memoria": "Lucro Líquido / Patrimônio Líquido",
        "valor": round(indicadores_foto['Retorno_Sobre_Patrimonio_Liquido'].values[0], 2),
        "tooltip": "Quanto maior, melhor. Mostra o retorno obtido pelos acionistas sobre o capital investido."
    },
    {
        "titulo": "Liquidez Corrente",
        "descricao": "Capacidade de a empresa saldar suas dívidas a curto prazo, em até 360 dias.",
        "memoria": "Ativo Circulante / Passivo Circulante",
        "valor": round(indicadores_foto['Liquidez_Corrente'].values[0], 2),
        "tooltip": "O ideal é acima de 1. Mostra se a empresa tem recursos suficientes para cobrir dívidas de curto prazo."
    },
    {
        "titulo": "Liquidez Geral",
        "descricao": "Situação financeira da empresa a longo prazo, incluindo dívidas acima de 360 dias.",
        "memoria": "Ativo Total / Passivo Total",
        "valor": round(indicadores_foto['Liquidez_Geral'].values[0], 2),
        "tooltip": "O ideal é acima de 1. Indica a capacidade de a empresa quitar dívidas totais com ativos totais."
    },
    {
        "titulo": "Liquidez Imediata",
        "descricao": "Quanto a empresa possui de imediato no caixa, frente às obrigações exigíveis.",
        "memoria": "Disponível / Passivo Circulante",
        "valor": round(indicadores_foto['Liquidez_Imediata'].values[0], 2),
        "tooltip": "Quanto maior, melhor. Mostra o quanto a empresa tem de recursos imediatos para pagar dívidas."
    }
]
# ======================== INIT ========================


receita_bruta = indicadores_foto['Receita_Bruta'].values[0]
receita_liquida = indicadores_foto['Receita_Líquida'].values[0]
lucro_bruito = indicadores_foto['Lucro_Bruto'].values[0]
lucro_liquido = indicadores_foto['Lucro_Líquido'].values[0]
disponibilidade_caixa = indicadores_foto['Disponibilidade_Caixa'].values[0]







# ======================
# SIDEBAR (MENU LATERAL)
# ======================
with st.sidebar:
    st.markdown("""
    <style>
        /* Força o fundo da sidebar preto */
        [data-testid="stSidebar"] {
            background-color: #000000 !important;
        }

        /* Centraliza logo */
        .sidebar-logo {
            width: 120px !important;
            margin: 20px auto 30px auto;
            display: block;
        }

        /* Menu itens */
        .menu-item {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 15px 10px;
            border-radius: 12px;
            margin-bottom: 12px;
            font-size: 14px;
            cursor: pointer;
            transition: all 0.2s;
            width: 100%;
            text-align: center;
            color: #CCCCCC; /* texto cinza claro */
        }

        .menu-item:hover {
            background-color: #222222;
            color: #FFFFFF;
        }

        /* Ícone do menu */
        .menu-item img {
            width: 29px;
            height: 29px;
            margin-bottom: 8px;
            filter: brightness(0) invert(0.8);
        }

        /* Item ativo */
        .menu-item.active {
            background-color: #333333;
            color: white;
        }
        .menu-item.active img {
            filter: brightness(0) invert(1);
        }

        .menu-item span {
            display: block;
            margin-top: 0;
        }
    </style>

    <img class="sidebar-logo" src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAfQAAAH0CAIAAABEtEjdAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAE4mlUWHRYTUw6Y29tLmFkb2JlLnhtcAAAAAAAPD94cGFja2V0IGJlZ2luPSfvu78nIGlkPSdXNU0wTXBDZWhpSHpyZVN6TlRjemtjOWQnPz4KPHg6eG1wbWV0YSB4bWxuczp4PSdhZG9iZTpuczptZXRhLyc+CjxyZGY6UkRGIHhtbG5zOnJkZj0naHR0cDovL3d3dy53My5vcmcvMTk5OS8wMi8yMi1yZGYtc3ludGF4LW5zIyc+CgogPHJkZjpEZXNjcmlwdGlvbiByZGY6YWJvdXQ9JycKICB4bWxuczpBdHRyaWI9J2h0dHA6Ly9ucy5hdHRyaWJ1dGlvbi5jb20vYWRzLzEuMC8nPgogIDxBdHRyaWI6QWRzPgogICA8cmRmOlNlcT4KICAgIDxyZGY6bGkgcmRmOnBhcnNlVHlwZT0nUmVzb3VyY2UnPgogICAgIDxBdHRyaWI6Q3JlYXRlZD4yMDI1LTEwLTIwPC9BdHRyaWI6Q3JlYXRlZD4KICAgICA8QXR0cmliOkV4dElkPjFkN2E1NzFiLWRmN2EtNDAyZi05NDUwLWU5ZWYyYTE2YTM3ZTwvQXR0cmliOkV4dElkPgogICAgIDxBdHRyaWI6RmJJZD41MjUyNjU5MTQxNzk1ODA8L0F0dHJpYjpGYklkPgogICAgIDxBdHRyaWI6VG91Y2hUeXBlPjI8L0F0dHJpYjpUb3VjaFR5cGU+CiAgICA8L3JkZjpsaT4KICAgPC9yZGY6U2VxPgogIDwvQXR0cmliOkFkcz4KIDwvcmRmOkRlc2NyaXB0aW9uPgoKIDxyZGY6RGVzY3JpcHRpb24gcmRmOmFib3V0PScnCiAgeG1sbnM6ZGM9J2h0dHA6Ly9wdXJsLm9yZy9kYy9lbGVtZW50cy8xLjEvJz4KICA8ZGM6dGl0bGU+CiAgIDxyZGY6QWx0PgogICAgPHJkZjpsaSB4bWw6bGFuZz0neC1kZWZhdWx0Jz5Db25zdWxYIC0gTE9HTyBQUkVUQTwvcmRmOmxpPgogICA8L3JkZjpBbHQ+CiAgPC9kYzp0aXRsZT4KIDwvcmRmOkRlc2NyaXB0aW9uPgoKIDxyZGY6RGVzY3JpcHRp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" alt="Logo"/>

    <div class="menu-item active">
        <img src="https://img.icons8.com/?size=100&id=2797&format=png&color=FFFFFF"/>
        <span>Dashboard</span>
    </div>
    <div class="menu-item">
        <img src="https://img.icons8.com/?size=100&id=11220&format=png&color=FFFFFF"/>
        <span>Clientes</span>
    </div>
    <div class="menu-item">
        <img src="https://img.icons8.com/?size=100&id=2969&format=png&color=FFFFFF"/>
        <span>Configuração</span>
    </div>
    """, unsafe_allow_html=True)


# ======================
# MENU DE ABAS
# ======================
st.markdown("""
<style>
.stTabs [data-baseweb="tab-list"] {
    border-bottom: 2px solid #ddd;
}
.stTabs [data-baseweb="tab"] {
    color: #666;
    font-weight: 600;
    border-bottom: 2px solid transparent;
    padding: 10px 20px;
}
.stTabs [aria-selected="true"] {
    color: #000000;
    border-bottom: 2px solid #000000;
}
</style>
""", unsafe_allow_html=True)
abas = st.tabs(["Contábil", "Índices", "Projeção", "Analítico"])




with abas[1]:  # Aba "Índices"
    st.markdown("""
    <style>
    .metric-card {
        background-color: #f8f9fa;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 0 10px rgba(0,0,0,0.05);
        height: 250px; /* 🔹 altura fixa para todos os cards */
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        margin-bottom: 25px; /* espaçamento entre linhas */
        overflow: hidden; /* evita quebra visual se texto for grande */
    }
    .metric-card h4 {
        margin-bottom: 10px;
        font-size: 16px;
        color: #222;
    }
    .metric-card p {
        margin-bottom: 10px;
        font-size: 14px;
        color: #555;
        text-align: justify;
    }
    .metric-card h2 {
        text-align: center;
        color: #000;
        margin-top: auto;
    }
    </style>
    """, unsafe_allow_html=True)


# ======================
# ÍNDICES
# ======================

    cols_per_row = 3
    for i in range(0, len(indicadores), cols_per_row):
        cols = st.columns(cols_per_row)
        for col, indicador in zip(cols, indicadores[i:i+cols_per_row]):
            col.markdown(f"""
            <div class="metric-card">
                <div>
                    <h4>{indicador['titulo']}</h4>
                    <p>{indicador['descricao']}</p>
                    <p style="font-size: 14px; color: #777;">
                        <b>Memória de Cálculo:</b> {indicador['memoria']}
                    </p>
                </div>
                <p title="{indicador['tooltip']}" style="
                text-align:center;
                font-size:28px;
                font-weight:bold;
                color:#333;
                margin:8px 0 0 0;
                cursor:help;
                ">
                {indicador['valor']}
                </p>
            </div>
            """, unsafe_allow_html=True)




with abas[0]:  # Aba "Contábil"
    st.markdown("""
        <style>
            /* Container principal */
            .main {
                padding: 0rem 1rem 1rem 1rem !important;
            }

            /* Remover excesso de padding entre colunas */
            div[data-testid="column"] {
                padding-left: 0.4rem !important;
                padding-right: 0.4rem !important;
            }

            /* Painel contábil */
            .bignumber-card {
                background-color: #ffffff;
                border-radius: 10px;
                box-shadow: 0 3px 8px rgba(0, 0, 0, 0.07);
                border: 1px solid #f1f1f1;
                padding: 18px 10px;
                text-align: center;
                transition: all 0.2s ease-in-out;
                height: 90px;
            }
            .bignumber-card:hover {
                transform: translateY(-2px);
                box-shadow: 0 5px 12px rgba(0, 0, 0, 0.1);
            }

            /* Título da métrica */
            .metric-title {
                color: #3c3c3c;
                font-size: 14px;
                font-weight: 600;
                margin-bottom: 6px;
            }

            /* Valor da métrica */
            .metric-value {
                color: #3c3c3c;
                font-size: 18px;
                font-weight: 700;
                margin-top: 4px;
            }

            /* Linha separadora sutil entre seções */
            hr {
                margin-top: 1rem;
                margin-bottom: 1rem;
                border: none;
                border-top: 1px solid #eee;
            }
                
        </style>
        """, unsafe_allow_html=True)




    # ======================
    # MÉTRICAS SUPERIORES
    # ======================

    col1, col2, col3, col4, col5 = st.columns(5)
    metrics = [
        ("Faturamento", receita_bruta, "Total de receitas antes de qualquer dedução ou custo"),
        ("Receita Líquida", receita_liquida, "Receita após deduções de impostos, devoluções e descontos"),
        ("Lucro Bruto", lucro_bruito, "Receita líquida menos custos diretos de produção"),
        ("Lucro Líquido", lucro_liquido, "Lucro após todas as despesas, impostos e custos"),
        ("Disponibilidade de Caixa", disponibilidade_caixa, "Valor disponível em caixa e equivalentes de caixa")
    ]

    icon_url = "https://img.icons8.com/?size=100&id=77&format=png&color=000000"

    for col, (titulo, valor, descricao) in zip([col1, col2, col3, col4, col5], metrics):
        col.markdown(f"""
        <div class="bignumber-card" style="padding:10px; background-color:#f8f8f8; border-radius:8px; text-align:center;">
            <div class="metric-title" style="font-weight:bold; color:#333; display:flex; align-items:center; justify-content:center; gap:4px;">
                {titulo} 
                <img src="{icon_url}" title="{descricao}" style="width:16px; height:16px; cursor:help;">
            </div>
            <div class="metric-value" style="font-size:20px; color:#595959;">R$ {valor:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)




    # ======================
    # GRÁFICOS DE INDICADORES TEMPORAIS
    # ======================

    # Converter índice para coluna e ordenar
    df_plot = indicadores_historicos.reset_index().sort_values("mes")
    df_plot["mes"] = df_plot["mes"].astype(str)
    df_plot, anos_escolhidos = filtro_ano(df_plot)

    df_plot["Banco"] = df_plot["Disponibilidade_Caixa"] * 0.5
    df_plot["Investimento"] = df_plot["Disponibilidade_Caixa"] * 0.3
    df_plot["Caixa"] = df_plot["Disponibilidade_Caixa"] * 0.2

    # Função para criar gráfico de barras por indicador

    col1, col2 = st.columns(2)

    # 1️⃣ Receita Líquida / Receita Bruta
    with col1:
        fig_receita = px.bar(
            df_plot,
            x="mes",
            y=["Receita_Bruta", "Receita_Líquida"],
            barmode="group",
            title="RECEITA LÍQUIDA / RECEITA BRUTA",
            labels={
                "mes": "Mês",
                "value": "Valor (R$)",
                "variable": "Indicador"
            },
            color_discrete_map={
                "Receita_Bruta": "#B0B0B0",   # Cinza médio
                "Receita_Líquida": "#595959"  # Cinza escuro
            },
            text_auto=".2s"
        )

        fig_receita.update_traces(
            textposition="outside",
            marker_line_width=0.8,
            marker_line_color="#E0E0E0"
        )

        fig_receita.update_layout(
            title={
                "text": "RECEITA LÍQUIDA / RECEITA BRUTA",
                "x": 0.5,          # Centraliza o título
                "xanchor": "center",
                "yanchor": "top"
                
            },
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            xaxis_title="Mês",
            yaxis_title="Valor (R$)",
            bargap=0.25,
            font=dict(color="#333", size=13),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.25,
                xanchor="center",
                x=0.5,
                title_text=""
            ),
            hovermode="x unified"
        )

        st.plotly_chart(fig_receita, use_container_width=True)


    # 2️⃣ Disponibilidade de Caixa
    with col2:
        df_caixa_melt = df_plot.melt(
            id_vars=["mes"],
            value_vars=["Banco", "Investimento", "Caixa"],
            var_name="Composição",
            value_name="Valor"
        )

        fig_caixa = px.bar(
            df_caixa_melt,
            x="mes",
            y="Valor",
            color="Composição",
            barmode="stack",
            title="DISPONIBILIDADE DE CAIXA",
            labels={
                "mes": "Mês",
                "Valor": "Valor (R$)",
                "Composição": "Composição"
            },
            color_discrete_map={
                "Banco": "#D9D9D9",         # cinza claro
                "Investimento": "#A6A6A6",  # cinza médio
                "Caixa": "#595959"          # cinza escuro
            }
        )

        fig_caixa.update_traces(
            texttemplate="%{y:,.0f}",
            textposition="inside"
        )

        fig_caixa.update_layout(
            title={
                "text": "DISPONIBILIDADE DE CAIXA",
                "x": 0.5,       # centraliza o título
                "xanchor": "center",
                "yanchor": "top"
            },
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            xaxis_title="Mês",
            yaxis_title="Valor (R$)",
            font=dict(color="#333", size=11),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.25,
                xanchor="center",
                x=0.5,
                title_text=""
            ),
            bargap=0.2,
            hovermode="x unified"
        )

        st.plotly_chart(fig_caixa, use_container_width=True)


    # Segunda linha de gráficos
    col3, col4 = st.columns(2)

    # 3️⃣ Custo / Receita Líquida
    with col3:
        fig_custo = px.area(
            df_plot,
            x="mes",
            y=["Custo_Total", "Receita_Líquida"],
            title="CUSTO / RECEITA LÍQUIDA",
            labels={
                "mes": "Mês",
                "value": "Valor (R$)",
                "variable": "Indicador"
            },
            color_discrete_map={
                "Custo_Total": "#595959",    # cinza escuro
                "Receita_Líquida": "#D9D9D9" # dourado vibrante
            }
        )

        # 🔹 Linhas mais suaves e preenchimento translúcido
        fig_custo.update_traces(
            mode="lines",
            line=dict(width=3),
            opacity=0.4
        )

        # 🔹 Layout refinado e legendas bem posicionadas
        fig_custo.update_layout(
            title={
            "text": "CUSTO / RECEITA LÍQUIDA",
            "x": 0.5,       # centraliza o título
            "xanchor": "center",
            "yanchor": "top"
            },
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            xaxis_title="Mês",
            yaxis_title="Valor (R$)",
            font=dict(color="#333", size=13),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.2,
                xanchor="center",
                x=0.5,
                title_text=""
            ),
            hovermode="x unified"
        )

        st.plotly_chart(fig_custo, use_container_width=True)



    # 4️⃣ Receita Líquida (barra simples)
    with col4:
        fig_liquida = px.bar(
            df_plot,
            x="mes",
            y="Receita_Líquida",
            title="RECEITA LÍQUIDA",
            text_auto=".2s",
            color_discrete_sequence=["#595959"],
            labels={"mes": "Mês", "Receita_Líquida": "Valor (R$)"}
        )

        fig_liquida.update_layout(
            title={
            "text": "RECEITA LÍQUIDA",
            "x": 0.5,       # centraliza o título
            "xanchor": "center",
            "yanchor": "top"
            },
            plot_bgcolor="#ffffff",
            xaxis_title="Mês",
            yaxis_title="Receita Líquida (R$)",
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.25,
                xanchor="center",
                x=0.5
            )
        )

        st.plotly_chart(fig_liquida, use_container_width=True)
    

    col5, = st.columns(1)

    # 5️⃣ Margem de Lucro (%)
    with col5:
        # Calcula a margem de lucro
        df_plot["Margem_de_Lucro"] = (df_plot["Lucro_Líquido"] / df_plot["Receita_Líquida"])  # mantém como fração

        # Cria gráfico de linha
        fig_margem = px.line(
            df_plot,
            x="mes",
            y="Margem_de_Lucro",
            title="MARGEM DE LUCRO (%)",
            markers=True,
            labels={
                "mes": "Mês",
                "Margem_de_Lucro": "Margem (%)"
            },
            hover_data={
                "Margem_de_Lucro": ":.1%",  # Formato percentual
            }
        )

        # Ajusta visual da linha
        fig_margem.update_traces(
            line=dict(width=3, color="#595959"),
            marker=dict(size=8)
        )

        # Layout do gráfico
        fig_margem.update_layout(
            title={
                "text": "MARGEM DE LUCRO (%)",
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "top"
            },
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            font=dict(color="#333333", size=12),
            xaxis_title="Mês",
            yaxis_title="Margem (%)",
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="#E5E5E5", tickformat=".0%"),  # Eixo Y em porcentagem
            hovermode="x unified",
            title_font=dict(size=18, color="#000000"),
            annotations=[
                dict(
                    x=0.5,
                    y=1.08,
                    xref="paper",
                    yref="paper",
                    text="",  # sem texto visível
                    showarrow=False,
                    hovertext="A Margem de Lucro (%) indica a porcentagem da receita líquida que se transforma em lucro líquido, mostrando a eficiência da empresa em gerar lucro.",
                    hoverlabel=dict(bgcolor="white", font_size=12)
                )
            ]
        )

        # Exibe o gráfico
        st.plotly_chart(fig_margem, use_container_width=True)


with abas[2]:
    
    indicadores_historicos.index = pd.to_datetime(
        indicadores_historicos.index, format="%Y-%m")
    
    resultado = prophet_ar2_forecast(
    indicadores_historicos, target_col='Liquidez_Imediata', horizon=6)
    
    previsao_futura = forecast_future_periods(
        indicadores_historicos, target_col='Liquidez_Imediata', horizon=6)

    res = resultado['forecast_df']

    # ===========================
    # TÍTULO DA ABA / SEÇÃO
    # ===========================
    st.markdown("### Predição dos próximos 6 meses com base nos últimos 3 anos")

    # =====================================
    # 1️⃣ BACK TEST - COMPARATIVO REAL x PREVISTO
    # =====================================
    fig_backtest = go.Figure()

    # Linha prevista
    fig_backtest.add_trace(go.Scatter(
        x=res["ds"],
        y=res["y_pred"],
        mode='lines+markers',
        name='Previsto',
        line=dict(color='#a6a6a6', width=3),
        marker=dict(size=6)
    ))

    # Linha real (preta)
    fig_backtest.add_trace(go.Scatter(
        x=res["ds"],
        y=res["y_true"],
        mode='lines+markers',
        name='Real',
        line=dict(color='#000000', width=3),  # linha preta grossa
        marker=dict(size=6)
    ))

    fig_backtest.update_layout(
        title="<b>BACKTEST - REAL x PREVISTO</b><br><sup>Comparação entre valores reais e previstos da Liquidez Imediata.</sup>",
        xaxis_title="Mês",
        yaxis_title="Liquidez Imediata",
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(color="#333", size=12),
        xaxis=dict(showgrid=False, tickangle=-45),
        yaxis=dict(showgrid=True, gridcolor="#E5E5E5", tickformat=".0%"),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            y=-0.35,   # legenda mais afastada do eixo X
            x=0.5,
            xanchor="center",
            font=dict(size=11)
        )
    )

    st.plotly_chart(fig_backtest, use_container_width=True)



    # =====================================
    # 2️⃣ PREVISÃO FUTURA - LIQUIDEZ IMEDIATA
    # =====================================

    # Cálculos estatísticos
    mean_value = previsao_futura["forecast"].mean()
    std_value = previsao_futura["forecast"].std()

    # Gráfico com mais picos e responsividade ao cursor
    fig_previsao = px.line(
        previsao_futura,
        x="ds",
        y="forecast",
        labels={"ds": "Mês", "forecast": "Liquidez Imediata Prevista"},
        markers=True,  # exibe os pontos
        line_shape="linear"  # evita suavização excessiva
    )

    # Limites de controle e média histórica
    fig_previsao.add_hline(y=mean_value, line_dash="solid", line_color="#000", line_width=2,)
    fig_previsao.add_hline(y=mean_value + 3*std_value, line_dash="dot", line_color="#999")
    fig_previsao.add_hline(y=mean_value - 3*std_value, line_dash="dot", line_color="#999")

    # Estilo da linha principal
    fig_previsao.update_traces(line=dict(color="#a6a6a6", width=3), marker=dict(size=6, color="#595959", line=dict(width=1, color="#fff")))

    # Layout e storytelling
    fig_previsao.update_layout(
        title="<b>PREVISÃO FUTURA - LIQUIDEZ IMEDIATA</b><br><sup>Inclui média histórica e limites de variação para análise de tendência.</sup>",
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(color="#333", size=12),
        xaxis_title="Mês",
        yaxis_title="Liquidez Imediata (Prevista)",
        xaxis=dict(showgrid=False, tickangle=-45),
        yaxis=dict(showgrid=True, gridcolor="#E5E5E5", tickformat=".0%"),
        hovermode="x unified",
        annotations=[
            dict(
                text="Linha preta = Média histórica",
                xref="paper", yref="paper",
                x=0.99, y=0.9,
                showarrow=False,
                font=dict(size=12, color="#333"),
                align="left"
            )
        ],
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.35,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
            bgcolor="rgba(0,0,0,0)"
        )
    )

    # Exibe o gráfico
    st.plotly_chart(fig_previsao, use_container_width=True)





with abas[3]:  # Aba "Analítico"
    st.subheader("Métricas do Balancete")
    indicadores_historicos
    
    