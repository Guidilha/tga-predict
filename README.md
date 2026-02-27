# 🏆 The Game Awards (TGA) Predictor: Prevendo Vencedores com Machine Learning

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Pandas](https://img.shields.io/badge/Pandas-Data_Manipulation-150458.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-Machine_Learning-F7931E.svg)

## 📌 Contexto do Projeto

Este projeto foi desenvolvido como um laboratório prático para explorar a aplicação de modelos de classificação em cenários de alta incerteza e dados desbalanceados. O objetivo central é **prever os vencedores do The Game Awards (TGA)** em diversas categorias (GOTY, Melhor Narrativa, Melhor Indie, Melhor Jogo para Família) com base em dados históricos de performance, engajamento e características técnicas dos jogos.

A premissa é investigar se as escolhas dos jurados do TGA seguem padrões estatísticos que podem ser mapeados por algoritmos, indo além do "hype" do momento.

## ⚙️ Arquitetura e Decisões Técnicas

A solução foi construída utilizando Python e as bibliotecas do ecossistema Scikit-Learn e Pandas. O pipeline de Machine Learning foi estruturado para garantir reprodutibilidade e modularidade.

### 1. Desafios de Modelagem
O principal desafio deste projeto foi o **desbalanceamento de classes**. Em qualquer categoria do TGA, há múltiplos indicados (classe 0 - não venceu), mas apenas um vencedor (classe 1 - venceu). Para mitigar o enviesamento do modelo:
* **Algoritmo Escolhido:** Optei pela **Regressão Logística** configurada com `class_weight='balanced'`. Isso penaliza o modelo mais severamente quando ele erra a classe minoritária (o vencedor), forçando-o a dar mais atenção a esses casos raros.
* **Probabilidade sobre Decisão Binária:** O modelo não cospe apenas `0` ou `1`. Ele utiliza `predict_proba()` para gerar uma **porcentagem de chance** de vitória, permitindo um ranqueamento mais realista dos indicados.

### 2. Engenharia de Features (Feature Engineering)
Os dados crus (como desenvolvedoras, publishers e datas de lançamento) precisavam ser transformados para que o modelo pudesse extrair valor:
* **Tratamento de Alta Cardinalidade:** Variáveis como `developer` e `publisher` possuem centenas de valores únicos. Para evitar uma explosão dimensional ao aplicar One-Hot Encoding, desenvolvi um transformador customizado (`AgrupadorCategoriasRaras`). Ele mantém apenas as Top 15 categorias mais frequentes e agrupa o resto em "Outros", melhorando a generalização do modelo.
* **Extração Temporal:** A data de lançamento (`release_date`) foi convertida em "dia do ano" (`day_of_year`), capturando a sazonalidade dos lançamentos (jogos lançados mais perto da premiação tendem a estar mais frescos na memória dos jurados).
* **Pipeline Scikit-Learn:** Toda a transformação de dados (escalonamento numérico com `StandardScaler` e codificação de categorias com `OneHotEncoder`) foi encapsulada em um `Pipeline` com `ColumnTransformer`, garantindo que não haja vazamento de dados (data leakage) entre treino e teste.

### 3. Divisão de Dados e Teste
* **Treino:** Histórico de vencedores e indicados de edições anteriores.
* **Teste/Predição:** A edição do ano corrente (2025).

## 🚀 Como Executar o Projeto

### Pré-requisitos
Certifique-se de ter o Python instalado e as bibliotecas necessárias.
```bash
pip install -r requirements.txt