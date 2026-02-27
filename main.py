import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin
import warnings

# ------------------------------------------------------------------------------
# CONFIGURAÇÃO GERAL E FILTROS DE AVISO
# ------------------------------------------------------------------------------
warnings.filterwarnings('ignore')

# ------------------------------------------------------------------------------
# TRANSFORMADOR PARA AGRUPAR CATEGORIAS RARAS (DEV/PUB)
# ------------------------------------------------------------------------------
class AgrupadorCategoriasRaras(BaseEstimator, TransformerMixin):
    def __init__(self, top_n=15):
        self.top_n = top_n
        self.top_categories_ = {}

    def fit(self, X, y=None):
        if hasattr(X, 'columns'):
            for col in X.columns:
                top = X[col].value_counts().nlargest(self.top_n).index.tolist()
                self.top_categories_[col] = set(top)
        return self

    def transform(self, X):
        X_copy = X.copy()
        if hasattr(X_copy, 'columns'):
            for col in X_copy.columns:
                if col in self.top_categories_:
                    X_copy[col] = X_copy[col].apply(
                        lambda x: x if x in self.top_categories_[col] else 'Outros'
                    )
        return X_copy

# ------------------------------------------------------------------------------
# FUNÇÕES DE CARREGAMENTO E EXECUÇÃO
# ------------------------------------------------------------------------------

def load_data():
    try:
        df_vencedores = pd.read_csv("Vencedores.csv", header=1)
        df_base = pd.read_csv("Base de Dados.csv")
        return df_base, df_vencedores
    except FileNotFoundError as e:
        print(f"Erro Crítico: Arquivo não encontrado: {e}")
        return None, None
    except Exception as e:
        print(f"Erro desconhecido ao carregar dados: {e}")
        return None, None

def get_probabilities_2025(df_base, df_vencedores, category_name):
    """
    Treina o modelo com dados históricos e retorna as probabilidades para 2025.
    """
    test_year = 2025

    # 1. PREPARAÇÃO INICIAL E TARGET
    df_cat = df_vencedores[df_vencedores[category_name].isin([1, 2])].copy()
    df_cat['target'] = np.where(df_cat[category_name] == 2, 1, 0)

    if df_cat.empty:
        return None

    df_merged = pd.merge(df_cat, df_base, on='name', how='left')

    # 2. FEATURE ENGINEERING
    try:
        df_merged['release_date_dt'] = pd.to_datetime(df_merged['release_date'], errors='coerce')
        df_merged['day_of_year'] = df_merged['release_date_dt'].dt.dayofyear
        df_merged['day_of_year'] = df_merged['day_of_year'].fillna(-1).astype(int)
        df_merged = df_merged.drop(columns=['release_date_dt'])
    except Exception:
        df_merged['day_of_year'] = -1

    df_merged = df_merged.drop(columns=['max_owners', 'min_owners'], errors='ignore')

    # 3. DIVISÃO DE DADOS (TREINO < 2025, TESTE == 2025)
    df_test = df_merged[df_merged['year'] == test_year].copy()
    df_train = df_merged[df_merged['year'] < test_year].copy()

    if df_test.empty or df_train.empty:
        return None

    y_train = df_train['target']
    test_game_names = df_test['name'].reset_index(drop=True)

    # 4. SELEÇÃO DE FEATURES
    removable_non_tag_features_list = [
        'required_age', 'price', 'dlc_count', 'qtd_user_score',
        'user_score', 'metacritic_score', 'achievements', 'developers',
        'publishers', 'estimated_owners', 'discount', 'count_lang',
        'count_lang_audio', 'day_of_year', 'main_end_time', 'art_style'
    ]
    categorical_base_list = ['developers', 'publishers', 'art_style']

    all_category_columns = ['goty', 'narrative', 'indie', 'family']
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

    # Tratamento de Nulos
    X_train[final_categorical_features] = X_train[final_categorical_features].fillna('Missing')
    X_test[final_categorical_features] = X_test[final_categorical_features].fillna('Missing')
    X_train[final_numeric_features] = X_train[final_numeric_features].fillna(0)
    X_test[final_numeric_features] = X_test[final_numeric_features].fillna(0)

    # 5. PIPELINE
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
        transformers.append(('cat_low', Pipeline(steps=[('encoder', OneHotEncoder(handle_unknown='ignore'))]), curr_low))

    if not transformers:
        return None

    model_pipeline = Pipeline(steps=[
        ('preprocessor', ColumnTransformer(transformers=transformers, remainder='drop')),
        ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=142))
    ])

    # 6. TREINO E PREVISÃO
    try:
        model_pipeline.fit(X_train, y_train)
    except ValueError:
        return None

    y_pred_proba_test = model_pipeline.predict_proba(X_test)

    # 7. ORGANIZAR RESULTADOS
    df_resultados = pd.DataFrame({
        'Jogo': test_game_names,
        'Probabilidade': y_pred_proba_test[:, 1] # Pega probabilidade da classe 1 (Vencer)
    })

    # Ordenar do maior para o menor
    df_resultados = df_resultados.sort_values(by='Probabilidade', ascending=False).reset_index(drop=True)

    return df_resultados

def main_predictions_2025():
    print("="*60)
    print(" PREVISÕES DE PROBABILIDADE - THE GAME AWARDS 2025 ")
    print("="*60)

    df_base, df_vencedores = load_data()
    if df_base is None: return

    CATEGORIES = ['goty', 'narrative', 'indie', 'family']

    for category in CATEGORIES:
        print(f"\n📁 CATEGORIA: {category.upper()}")
        print("-" * 40)

        df_probs = get_probabilities_2025(df_base, df_vencedores, category)

        if df_probs is not None:
            # Exibe os resultados formatados
            for index, row in df_probs.iterrows():
                prob = row['Probabilidade'] * 100
                nome = row['Jogo']

                # Formatação visual: Ícone para o favorito (>50% ou o primeiro da lista)
                prefix = "🏆" if index == 0 else "  "
                print(f"{prefix} {nome:<30} | {prob:.2f}% de chance")
        else:
            print("  [!] Dados insuficientes para gerar previsões nesta categoria.")

        print("-" * 40)

if __name__ == "__main__":
    main_predictions_2025()