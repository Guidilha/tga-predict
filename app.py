import streamlit as st
import pandas as pd
# Importa as funções que você já criou no seu arquivo principal
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

# 3. CARREGAMENTO DE DADOS (Com Cache)
# O @st.cache_data é um "pulo do gato": ele impede que o Streamlit leia os arquivos CSV 
# do zero toda vez que você clicar em um botão, deixando o site super rápido.
@st.cache_data
def carregar_dados_em_cache():
    return load_data()

df_base, df_vencedores = carregar_dados_em_cache()

if df_base is None:
    st.error("Erro crítico: Arquivos CSV não encontrados. Verifique a pasta do projeto.")
    st.stop()

# 4. BARRA LATERAL (MENU)
st.sidebar.header("⚙️ Configurações")
st.sidebar.markdown("Escolha a categoria que deseja prever:")

# Dicionário para deixar os nomes bonitos na tela e passar o código certo para a sua função
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

# Botão para gerar a previsão (dá uma sensação tátil de "rodar o modelo")
if st.button("Calcular Probabilidades 🎲", type="primary"):
    
    with st.spinner("Treinando modelo e calculando chances..."):
        # Aqui chamamos a SUA função do main.py!
        df_resultados = get_probabilities_2025(df_base, df_vencedores, categoria_tecnica)
        
        if df_resultados is not None and not df_resultados.empty:
            
            # Criando uma coluna amigável em porcentagem
            df_resultados['Chance (%)'] = (df_resultados['Probabilidade'] * 100).round(2)
            
            # Destacando o grande favorito (o primeiro da lista)
            vencedor = df_resultados.iloc[0]['Jogo']
            chance_vencedor = df_resultados.iloc[0]['Chance (%)']
            
            st.success(f"🏅 **Favorito estatístico:** {vencedor} ({chance_vencedor}%)")
            
            # Gráfico de Barras interativo
            st.markdown("### 📊 Ranking de Probabilidades")
            # O Streamlit precisa que o nome do jogo seja o índice para plotar o gráfico corretamente
            df_grafico = df_resultados.set_index('Jogo')[['Chance (%)']]
            st.bar_chart(df_grafico)
            
            # Tabela de Dados (Dataframe)
            st.markdown("### 📋 Detalhamento dos Indicados")
            st.dataframe(
                df_resultados[['Jogo', 'Chance (%)']],
                use_container_width=True,
                hide_index=True # Esconde aquele índice numérico (0, 1, 2...)
            )
            
        else:
            st.warning("⚠️ Dados insuficientes para gerar previsões consistentes nesta categoria.")