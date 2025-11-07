from utils.functions import processar_indicadores_financeiros, extract_accounts, extract_mes_from_periodo
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import streamlit as st
import os

def _get_mongo_uri_from_secrets():
    try:
        return st.secrets["mongo"]["uri"]
    except Exception:
        return os.environ.get("MONGO_URI")


def get_db_client():
    uri = _get_mongo_uri_from_secrets()
    if not uri:
        st.error(
            "MONGO_URI não encontrado. Configure st.secrets ou a variável de ambiente 'MONGO_URI'.")
        st.stop()
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        # força a checagem de conexão rápida
        client.server_info()
        return client
    except Exception as e:
        st.error(f"Falha ao conectar no MongoDB: {e}")
        st.stop()


# @st.cache_data(ttl=60 * 30)  # cache por 30 minutos; ajusta se precisar
def load_all_rows_from_mongo(db_name="ConsulX_db", coll_name="industrial_nordeste", limit=None):
    """
    Carrega e processa os balancetes da coleção. Cacheado para evitar re-leitura a cada rerun.
    """
    client = get_db_client()
    db = client[db_name]
    colecao = db[coll_name]

    all_rows = []
    # opcional: colocar limit para testes locais (evitar timeouts no deploy)
    cursor = colecao.find()
    if limit:
        cursor = cursor.limit(limit)

    for doc in cursor:
        source_id = doc.get('_id')
        metadata = doc.get('metadata', {}) or {}
        periodo = metadata.get('periodo') or metadata.get(
            'period') or metadata.get('periodo_referencia')
        mes = extract_mes_from_periodo(periodo)

        candidate_sections = []
        for key in ('data', 'content', 'payload', 'balancete', 'document'):
            if key in doc and isinstance(doc[key], dict):
                # check values
                for v in doc[key].values():
                    if isinstance(v, dict) and 'descricao' in v:
                        candidate_sections.append(v)
                if isinstance(doc[key], dict) and 'descricao' in doc[key]:
                    candidate_sections.append(doc[key])

        if not candidate_sections:
            for v in doc.values():
                if isinstance(v, dict) and 'descricao' in v:
                    candidate_sections.append(v)

        for section in candidate_sections:
            contas = extract_accounts(section)
            for conta in contas:
                conta["mes"] = mes
                conta["source_id"] = source_id
            all_rows.extend(contas)

    return all_rows
