import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Tuple, Optional, Any

class AgrupadorCategoriasRaras(BaseEstimator, TransformerMixin):
    """
    Transformador customizado para lidar com alta cardinalidade em variáveis categóricas.
    Mantém as 'top_n' categorias mais frequentes e agrupa o restante como 'Outros',
    evitando explosão dimensional no OneHotEncoder do pipeline principal.
    """
    def __init__(self, top_n: int = 15) -> None:
        self.top_n = top_n
        self.top_categories_ = {}

    def fit(self, X: pd.DataFrame, y: Any = None) -> 'AgrupadorCategoriasRaras':
        if hasattr(X, 'columns'):
            for col in X.columns:
                top = X[col].value_counts().nlargest(self.top_n).index.tolist()
                self.top_categories_[col] = set(top)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()
        if hasattr(X_copy, 'columns'):
            for col in X_copy.columns:
                if col in self.top_categories_:
                    X_copy[col] = X_copy[col].apply(
                        lambda x: x if x in self.top_categories_[col] else 'Outros'
                    )
        return X_copy


def load_data() -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Carrega os datasets contendo as características dos jogos e o histórico de vencedores.

    Returns:
        Tuple contendo (df_base, df_vencedores). Retorna (None, None) em caso de falha de I/O.
    """
    try:
        # header=1 indica que a primeira linha do CSV é metadado ou ignorável
        df_vencedores = pd.read_csv("Vencedores.csv", header=1)
        df_base = pd.read_csv("Base de Dados.csv")
        return df_base, df_vencedores
        
    except FileNotFoundError as e:
        print(f"Erro de I/O: Arquivo de dados não encontrado. Detalhes: {e}")
        return None, None
    except pd.errors.EmptyDataError:
        print("Erro de I/O: Um dos arquivos CSV está vazio.")
        return None, None


def get_probabilities_2025(
    df_base: pd.DataFrame, 
    df_vencedores: pd.DataFrame, 
    category_name: str
) -> Optional[pd.DataFrame]:
    """
    Treina um pipeline de Regressão Logística para calcular a probabilidade de vitória
    dos indicados na edição de 2025 de uma categoria específica do TGA.

    Args:
        df_base: DataFrame com as features técnicas e mercadológicas dos jogos.
        df_vencedores: DataFrame com o histórico de indicações e vitórias.
        category_name: Nome da coluna da categoria a ser analisada (ex: 'goty').

    Returns:
        DataFrame contendo os jogos de 2025 ordenados por probabilidade de vitória,
        ou None se os dados históricos forem insuficientes.
    """
    test_year = 2025

    # Isola os jogos indicados (1) e vencedores (2) para criar a variável alvo binária (target)
    df_cat = df_vencedores[df_vencedores[category_name].isin([1, 2])].copy()
    df_cat['target'] = np.where(df_cat[category_name] == 2, 1, 0)

    if df_cat.empty:
        return None

    df_merged = pd.merge(df_cat, df_base, on='name', how='left')

    # Engenharia de features: Sazonalidade (dia do ano) tende a influenciar a memória dos jurados
    try:
        df_merged['release_date_dt'] = pd.to_datetime(df_merged['release_date'], errors='coerce')
        df_merged['day_of_year'] = df_merged['release_date_dt'].dt.dayofyear
        df_merged['day_of_year'] = df_merged['day_of_year'].fillna(-1).astype(int)
        df_merged = df_merged.drop(columns=['release_date_dt'])
    except Exception:
        df_merged['day_of_year'] = -1

    df_merged = df_merged.drop(columns=['max_owners', 'min_owners'], errors='ignore')

    # Separação Temporal: Treinamos com dados históricos estritos e validamos com o ano corrente
    df_test = df_merged[df_merged['year'] == test_year].copy()
    df_train = df_merged[df_merged['year'] < test_year].copy()

    if df_test.empty or df_train.empty:
        return None

    y_train = df_train['target']
    test_game_names = df_test['name'].reset_index(drop=True)

    # Definição estrita das variáveis (features) que irão compor o modelo
    removable_non_tag_features_list = [
        'required_age', 'price', 'dlc_count', 'qtd_user_score',
        'user_score', 'metacritic_score', 'achievements', 'developers',
        'publishers', 'estimated_owners', 'discount', 'count_lang',
        'count_lang_audio', 'day_of_year', 'main_end_time', 'art_style'
    ]
    categorical_base_list = ['developers', 'publishers', 'art_style']
    all_category_columns = ['goty', 'narrative', 'indie', 'family']
    
    # Evita data leakage removendo colunas de target e identificadores textuais
    cols_to_drop_always = ['name', 'year', 'target', 'release_date'] + all_category_columns
    excluded_cols = set(cols_to_drop_always + removable_non_tag_features_list)

    tag_features_list = [
        col for col in df_merged.columns
        if col not in excluded_cols and pd.api.types.is_numeric_dtype(df_merged[col])
    ]

    metrics_and_others_list = [c for c in removable_non_tag_features_list if c not in categorical_base_list]

    final_numeric_features = tag_features_list + metrics_and_others_list
    final_categorical_features = categorical_base_list[:]

    X_train = df_train[final_numeric_features + final_categorical_features].copy()
    X_test = df_test[final_numeric_features + final_categorical_features].copy()

    # Tratamento de dados faltantes (Imputação simples focada em estabilidade do modelo)
    X_train[final_categorical_features] = X_train[final_categorical_features].fillna('Missing')
    X_test[final_categorical_features] = X_test[final_categorical_features].fillna('Missing')
    X_train[final_numeric_features] = X_train[final_numeric_features].fillna(0)
    X_test[final_numeric_features] = X_test[final_numeric_features].fillna(0)

    # Construção do Pipeline de Pré-processamento e Modelagem
    transformers = []
    if final_numeric_features:
        transformers.append(('num', Pipeline(steps=[('scaler', StandardScaler())]), final_numeric_features))

    high_cardinality_cols = ['developers', 'publishers']
    curr_high = [c for c in high_cardinality_cols if c in final_categorical_features]
    curr_low = [c for c in final_categorical_features if c not in high_cardinality_cols]

    if curr_high:
        transformers.append(('cat_high', Pipeline(steps=[
            ('grouper', AgrupadorCategoriasRaras(top_n=15)),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), curr_high))
        
    if curr_low:
        transformers.append(('cat_low', Pipeline(steps=[
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), curr_low))

    if not transformers:
        return None

    # O uso de class_weight='balanced' é crucial devido ao forte desbalanceamento natural da premiação (apenas 1 vencedor)
    model_pipeline = Pipeline(steps=[
        ('preprocessor', ColumnTransformer(transformers=transformers, remainder='drop')),
        ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=142))
    ])

    try:
        model_pipeline.fit(X_train, y_train)
    except ValueError as e:
        print(f"Erro matemático ao ajustar o modelo para a categoria {category_name}: {e}")
        return None

    y_pred_proba_test = model_pipeline.predict_proba(X_test)

    # Retorna as probabilidades isolando a classe 1 (Probabilidade de vitória)
    df_resultados = pd.DataFrame({
        'Jogo': test_game_names,
        'Probabilidade': y_pred_proba_test[:, 1] 
    })

    df_resultados = df_resultados.sort_values(by='Probabilidade', ascending=False).reset_index(drop=True)
    return df_resultados


def main_predictions_2025() -> None:
    """
    Função principal de orquestração via CLI (Interface de Linha de Comando).
    """
    print("="*60)
    print(" PREVISÕES DE PROBABILIDADE - THE GAME AWARDS 2025 ")
    print("="*60)

    df_base, df_vencedores = load_data()
    if df_base is None or df_vencedores is None:
        return

    categories = ['goty', 'narrative', 'indie', 'family']

    for category in categories:
        print(f"\n📁 CATEGORIA: {category.upper()}")
        print("-" * 40)

        df_probs = get_probabilities_2025(df_base, df_vencedores, category)

        if df_probs is not None and not df_probs.empty:
            for index, row in df_probs.iterrows():
                prob = row['Probabilidade'] * 100
                nome = row['Jogo']
                prefix = "🏆" if index == 0 else "  "
                print(f"{prefix} {nome:<30} | {prob:.2f}% de chance")
        else:
            print("  [!] Dados insuficientes para gerar previsões nesta categoria.")
        
        print("-" * 40)

if __name__ == "__main__":
    main_predictions_2025()