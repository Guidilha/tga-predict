import streamlit as st
import pandas as pd
from main import load_data, get_probabilities_2025

# 1. CONFIGURAÇÃO DA PÁGINA
st.set_page_config(
    page_title="TGA 2025 Predictor",
    page_icon="🏆",
    layout="centered"
)

# 2. CABEÇALHO DO DASHBOARD
st.title("🏆 The Game Awards Predictor")
st.markdown("""
Bem-vindo ao painel preditivo do TGA! Este dashboard utiliza **Machine Learning** (Regressão Logística) 
para analisar o histórico de premiações e calcular a probabilidade de vitória dos jogos na edição de 2025.
""")

@st.cache_data
def carregar_dados_em_cache():
    return load_data()

df_base, df_vencedores = carregar_dados_em_cache()

if df_base is None:
    st.error("Erro crítico: Ficheiros CSV não encontrados. Verifica a pasta do projeto.")
    st.stop()

# 4. BARRA LATERAL (MENU)
st.sidebar.header("⚙️ Configurações")
st.sidebar.markdown("Escolha a categoria que deseja prever:")

categorias_map = {
    "Game of the Year (GOTY)": "goty",
    "Melhor Narrativa": "narrative",
    "Melhor Jogo Independente": "indie",
    "Melhor Jogo para Família": "family"
}

categoria_selecionada = st.sidebar.selectbox(
    "Categoria:",
    list(categorias_map.keys())
)

categoria_tecnica = categorias_map[categoria_selecionada]

# 5. ÁREA PRINCIPAL E RESULTADOS
st.subheader(f"Análise para: **{categoria_selecionada}**")

# --- NOVO: ALERTAS ANALÍTICOS DINÂMICOS ---
if categoria_tecnica in ['narrative', 'family']:
    st.info("💡 **Nota Analítica (Grupo de Controlo):** Esta categoria possui uma forte componente de **subjetividade humana**. Ao contrário do GOTY, métricas objetivas de mercado e performance técnica têm menor poder preditivo aqui. O modelo utiliza estas categorias para testar os limites do algoritmo face a escolhas puramente qualitativas de um júri.")
elif categoria_tecnica == 'goty':
    st.success("📈 **Nota Analítica (Data Drift):** O modelo apresenta alta fiabilidade nesta categoria para edições recentes. Variáveis de engajamento atual (como volume de avaliações) demonstraram ser indicadores vitais para a previsão do GOTY na atualidade.")
# ------------------------------------------

if st.button("Calcular Probabilidades 🎲", type="primary"):
    
    with st.spinner("A treinar o modelo e a calcular probabilidades..."):
        df_resultados = get_probabilities_2025(df_base, df_vencedores, categoria_tecnica)
        
        if df_resultados is not None and not df_resultados.empty:
            df_resultados['Chance (%)'] = (df_resultados['Probabilidade'] * 100).round(2)
            
            vencedor = df_resultados.iloc[0]['Jogo']
            chance_vencedor = df_resultados.iloc[0]['Chance (%)']
            
            st.success(f"🏅 **Favorito estatístico:** {vencedor} ({chance_vencedor}%)")
            
            st.markdown("### 📊 Ranking de Probabilidades")
            df_grafico = df_resultados.set_index('Jogo')[['Chance (%)']]
            st.bar_chart(df_grafico)
            
            st.markdown("### 📋 Detalhamento dos Indicados")
            st.dataframe(
                df_resultados[['Jogo', 'Chance (%)']],
                use_container_width=True,
                hide_index=True 
            )
        else:
            st.warning("⚠️ Dados insuficientes para gerar previsões consistentes nesta categoria.")