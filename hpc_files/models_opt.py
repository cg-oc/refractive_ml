from matminer.featurizers.structure import DensityFeatures
from matminer.featurizers.composition import ElementProperty
from sklearn.linear_model import ElasticNet, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import GroupKFold
from sklearn.base import clone
from optuna.samplers import TPESampler
from functools import partial
import numpy as np
import pandas as pd
import optuna
import os
import time

# SILENCIA AVISOS
from sklearn.exceptions import ConvergenceWarning
import warnings

warnings.filterwarnings("ignore", category=ConvergenceWarning) # caso lasso não convirja
optuna.logging.set_verbosity(optuna.logging.WARNING) # avisos gerais do optuna
np.seterr(over='ignore'); # avisos do numpy

df = pd.read_pickle("dielectric_constant.pkl")
print(f"Quantidade de valores inválidos: {df.isna().sum().sum()}")

y= df.pop("n")

# Instancia extratores
dens_feat = DensityFeatures()
magpie_feat = ElementProperty.from_preset("magpie")

# Extração de features
features_combinadas = [
    dens_feat.featurize(s) + magpie_feat.featurize(s.composition) 
    for s in df["structure"]
]
labels_combinados = dens_feat.feature_labels() + magpie_feat.feature_labels()

# 3. Criar o DataFrame de atributos e unir com o df original (já sem 'structure')
df_features = pd.DataFrame(features_combinadas, columns=labels_combinados, index=df.index)

X = pd.concat([df, df_features], axis=1)

features_excluidas = ["cif", "material_id", "meta", "poscar", "structure", 
    "space_group", "formula", "e_electronic", "e_total", "poly_total", "poly_electronic"]

X.drop(features_excluidas, axis=1, inplace=True);

# Definição do atributo categórico
coluna_categorica = ['pot_ferroelectric']

# Codifica apenas categorica para 0 e 1, deixando apenas uma delas no final.
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='if_binary'), coluna_categorica)
    ],
    remainder='passthrough'
)

TAMANHO_TESTE = 0.2
SEMENTE_ALEATORIA = 42

grupos = df['formula']

# 2. Configurar o separador de grupos
gss = GroupShuffleSplit(
    n_splits=1, 
    test_size=TAMANHO_TESTE,
    random_state=SEMENTE_ALEATORIA
)

for idx_treino, idx_teste in gss.split(X, y, groups=grupos):
    x_treino = X.iloc[idx_treino]
    x_teste = X.iloc[idx_teste]
    y_treino = y.iloc[idx_treino]
    y_teste = y.iloc[idx_teste]
    
    # Evita data leakage ao dividir pela formula das obs. de treino
    grupos_treino = grupos.iloc[idx_treino]

def instancia_pipeline(trial, algoritmo, parametros):

    """Constrói um Pipeline do Scikit-Learn integrando pré-processamento e o modelo.

    Atua como uma fábrica (Factory) que padroniza o pipeline base
    (preprocessor + VarianceThreshold) e adiciona dinamicamente transformações de
    atributos (PCA) e do alvo (TransformedTargetRegressor com log1p/expm1)
    conforme as escolhas de hiperparâmetros sugeridas pelo Optuna.

    Parameters
    ----------
    trial : optuna.Trial
        Objeto de tentativa do Optuna utilizado paraamostrar decisões
        arquiteturais (ex: usar PCA e/ou transformação logarítmica).
    algoritmo : type
        Classe do regressor do Scikit-Learn a ser instanciada (ex: Ridge,
        KernelRidge, SVR, RandomForestRegressor).
    parametros : dict
        Dicionário contendo os hiperparâmetros específicos do algoritmo já
        amostrados pelo Optuna.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Pipeline completo configurado com pré-processamento, seleção/redução de
        dimensionalidade opcional e o regressor envelopado.
    """

    # Define etapas fudnamentais do pipeline
    etapas_pipe = [preprocessor, VarianceThreshold(threshold=0)]

    #  Adiciona StandardScaler ao pipeline caso optuna escolha usá-lo
    usa_padronizador = trial.suggest_categorical('usar_padronizador', [True, False])
    if usa_padronizador:
        etapas_pipe.append(StandardScaler())

    # Adiciona PCA ao pipeline caso optuna escolha usá-lo
    usa_pca = trial.suggest_categorical('usar_pca', [True, False])
    if usa_pca:
        num_componentes_pca = trial.suggest_int('num_componentes', 10, 70)
        etapas_pipe.append(
            PCA(
                n_components=num_componentes_pca, 
                random_state=SEMENTE_ALEATORIA 
                )
        )

    # versão "base" do modelo
    modelo = algoritmo(**parametros)

    # Atualiza modelo caso optuna deseja usar transformação log
    transforma_log = trial.suggest_categorical("transforma_log", [True, False])
    if transforma_log:
        modelo = TransformedTargetRegressor(
            regressor=modelo,
            func=np.log1p,
            inverse_func=np.expm1
        )
    # Por último, adiciona o modelo à lista do pipeline
    etapas_pipe.append(modelo)

    return make_pipeline(*etapas_pipe)

def funcao_objetivo(trial, funcao_instancia, X, y, grupos, num_folds):
    """Função objetivo genérica para otimização com validação cruzada por grupos.

    Executa a avaliação de um modelo via GroupKFold mantendo a integridade dos
    grupos químicos entre treino e validação. Inclui tratamento de erros numéricos
    (overflow na reconstrução exponencial do log) e integra a poda antecipada
    (pruning) do Optuna passo a passo a cada fold.

    Parameters
    ----------
    trial : optuna.Trial
        Objeto de tentativa do Optuna contendo o estado da otimização atual.
    funcao_instancia : callable
        Função fábrica responsável por receber o `trial` e retornar um Pipeline
        configurado (ex: `instancia_pipe_ridge`).
    X : pandas.DataFrame ou numpy.ndarray
        Matriz de atributos de treino.
    y : pandas.Series ou numpy.ndarray
        Vetor da variável alvo de treino.
    grupos : array-like
        Identificadores dos grupos (ex: fórmulas químicas) usados para divisão do GroupKFold.
    num_folds : int
        Número de fatias (folds) para a validação cruzada.

    Returns
    -------
    float
        Média do RMSE (Root Mean Squared Error) obtida nos folds de validação.

    Raises
    ------
    optuna.TrialPruned
        Se a predição gerar valores numéricos inválidos (infinito/overflow no expm1)
        ou se o podador (pruner) determinar que a tentativa corrente tem desempenho
        inferior ao histórico.
    """

    modelo_base = funcao_instancia(trial)

    metricas = []

    gkf = GroupKFold(num_folds)

    for passo, fold in enumerate(gkf.split(X, y, groups=grupos)):

        indice_treino, indice_teste = fold

        X_treino_, X_teste_ = X.iloc[indice_treino], X.iloc[indice_teste]
        y_treino_, y_teste_ = y.iloc[indice_treino], y.iloc[indice_teste]

        modelo = clone(modelo_base)
        modelo.fit(X_treino_, y_treino_)
        
        y_verdadeiro = y_teste_
        
        try:
            y_previsto = modelo.predict(X_teste_)
            RMSE = root_mean_squared_error(y_verdadeiro, y_previsto)
        except ValueError:
            # Se a predição estourar o limite numérico (infinito), aborta esta tentativa
            raise optuna.TrialPruned()

        metricas.append(RMSE)
        trial.report(np.mean(metricas), passo)

        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(metricas)

def otimiza_hiper(pipeline, X, y, grupos, nome_estudo, num_trials=100):
    """Gerencia e executa o estudo de otimização de hiperparâmetros no Optuna.

    Configura o banco de dados SQLite para persistência dos resultados, inicializa
    o podador (`MedianPruner`) e o amostrador (`TPESampler`), mapeia a função objetivo
    via `partial` e executa as tentativas de otimização.

    Parameters
    ----------
    pipeline : callable
        Função de fábrica específica do modelo que recebe um `trial` e retorna o
        pipeline montado (ex: `instancia_pipe_ridge`).
    X : pandas.DataFrame
        Conjunto de dados de entrada de treino.
    y : pandas.Series
        Variável alvo de treino.
    grupos : array-like
        Identificadores dos grupos para a divisão do GroupKFold.
    nome_estudo : str
        Nome identificador do estudo no Optuna (utilizado também para o arquivo .db).
    num_trials : int, optional
        Quantidade total de tentativas (trials) a serem executadas (default é 100).

    Returns
    -------
    optuna.study.Study
        Objeto do estudo do Optuna concluído, contendo o histórico de tentativas,
        métricas e o `best_trial`.
    """
    
    # cria pasta para arquivos .db do optuna caso ela não exista ainda
    os.makedirs("optuna_files2", exist_ok=True)

    NUM_TRIALS_AQUECIMENTO = 5
    NUM_PASSOS_AQUECIMENTO = 3
    NUM_FOLDS = 5

    PODADOR = optuna.pruners.MedianPruner(
        n_startup_trials=NUM_TRIALS_AQUECIMENTO,
        n_warmup_steps=NUM_PASSOS_AQUECIMENTO,
    )

    estudo = optuna.create_study(
        direction="minimize",
        study_name=nome_estudo,
        storage=f"sqlite:///optuna_files2/{nome_estudo}.db",
        load_if_exists=True,
        sampler=TPESampler(),
        pruner=PODADOR,
    )

    funcao_parcial = partial(
        funcao_objetivo,
        funcao_instancia=pipeline,
        X=X,
        y=y,
        grupos=grupos,
        num_folds=NUM_FOLDS
    )

    estudo.optimize(funcao_parcial, n_trials=num_trials)

    return estudo

def instancia_pipe_ridge(trial):
    parametros = {
        # espaço de busca
        "alpha": trial.suggest_float("alpha", 0.01, 10_000, log=True)
    }

    return instancia_pipeline(trial, Ridge, parametros)

def instancia_pipe_lasso(trial):
    parametros = {
        # espaço de busca
        "alpha": trial.suggest_float("alpha", 1e-4, 10, log=True),
        # argumento fixo
        "max_iter": 10_000
    }

    return instancia_pipeline(trial, Lasso, parametros)

def instancia_pipe_elasticnet(trial):
    parametros = {
        # espaço de busca
        "alpha": trial.suggest_float("alpha", 1e-5, 100, log=True),
        "l1_ratio": trial.suggest_float("l1_ratio", 0, 1),
        # argumento fixo
        "max_iter": 10_000
    }

    return instancia_pipeline(trial, ElasticNet, parametros)

def instancia_pipe_floresta_aleatoria(trial):

    parametros = {
        # espaço de busca
        "n_estimators": trial.suggest_int("num_arvores", 5, 300, log=True),
        "criterion": trial.suggest_categorical("critério", ["squared_error", "absolute_error", "poisson"]),
        "min_samples_split": trial.suggest_int("min_exemplos_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_exemplos_folha", 1, 20),
        "max_features": trial.suggest_float("num_max_atributos", 0.1, 1),
        # argumentos fixos
        "n_jobs": -1,
        "bootstrap": True,
        "random_state": SEMENTE_ALEATORIA,
    }

    return instancia_pipeline(trial, RandomForestRegressor, parametros)

def instancia_pipe_svr(trial):

  parametros = {
    # espaço de busca
    'C': trial.suggest_float('C', 1e-1, 1e3, log=True),
    "epsilon": trial.suggest_float('epsilon', 1e-3, 1.0, log=True),
    "gamma": trial.suggest_float('gamma', 1e-4, 1.0, log=True),
    # arg fixo
    'kernel': 'rbf'
  }

  return instancia_pipeline(trial, SVR, parametros)

def instancia_pipe_krr(trial):

  parametros = {
    # espaço de busca
    'alpha': trial.suggest_float('alpha', 1e-4, 1.0, log=True),
    "gamma": trial.suggest_float('gamma', 1e-4, 1.0, log=True),
    "kernel": trial.suggest_categorical('kernel', ['rbf', 'laplacian', 'linear', 'polynomial']),    
  }

    # caso kernel seja polinomial, decide o grau do polinomio
  if parametros['kernel'] == 'polynomial':
      parametros['degree'] = trial.suggest_int('degree', 2, 4)
  
  return instancia_pipeline(trial, KernelRidge, parametros)

def formata_tempo(segundos):
    """Converte segundos em formato legível de horas, minutos e segundos."""
    horas = int(segundos // 3600)
    minutos = int((segundos % 3600) // 60)
    segs = segundos % 60
    
    if horas > 0:
        return f"{horas}h {minutos}m {segs:.1f}s"
    elif minutos > 0:
        return f"{minutos}m {segs:.1f}s"
    else:
        return f"{segs:.2f}s"

pipe_modelos = (
   instancia_pipe_ridge, instancia_pipe_lasso, 
   instancia_pipe_elasticnet, instancia_pipe_floresta_aleatoria,
   instancia_pipe_svr, instancia_pipe_krr
)

nome_modelos = ("ridge", "lasso", "elasticnet", "floresta_aleatoria", "svr", "krr")

estudos = {nome: 0 for nome in nome_modelos}
NUM_TRIALS = 1_000

TEMPO_INICIO_PROGRAMA = time.perf_counter()

for nome, modelo_pipe, in zip(nome_modelos, pipe_modelos):

    tempo_inicio_estudo = time.perf_counter()

    estudos[nome] = otimiza_hiper(
        modelo_pipe, 
        X=x_treino, y=y_treino, 
        grupos=grupos_treino, 
        nome_estudo=f"{nome}_poda",
        num_trials=NUM_TRIALS
    )

    tempo_estudo = time.perf_counter() - tempo_inicio_estudo
    tempo_total_acumulado = time.perf_counter() - TEMPO_INICIO_PROGRAMA

    print(
        f"Concluído estudo do modelo {nome}!"
        f"\n{'—'*50}\n"
        f"⏱️  Duração: {formata_tempo(tempo_estudo)}\n"
        f"⌛ Tempo total acumulado do programa    : {formata_tempo(tempo_total_acumulado)}\n"
        f"{'—'*50}\n"
    )