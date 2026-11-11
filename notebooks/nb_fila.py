"""Fila de revisão abaixo do ponto de corte: a melhor configuração da tese.

Funções curtas para o notebook didático. O fluxo recomendado pela tese, como
hipótese de trabalho a validar em coorte nova, é:

1. preservar os vínculos aceitos pelo ponto de corte do OpenRecLink;
2. na *grey area* (região abaixo do ponto de corte do OpenRecLink), estimar a
   probabilidade calibrada fora da amostra de cada potencial par: XGBoost sobre
   os 29 subescores e a nota final do CdR, calibração isotônica, validação
   cruzada aninhada agrupada por óbito (REFREC);
3. ordenar a fila de revisão manual por essa probabilidade, isolada (O3) ou
   fundida ao escore do OpenRecLink por posição recíproca (F4), e escolher entre
   elas pela curva de esforço, com a incerteza dada pelo bootstrap agrupado;
4. se houver LLM, usá-lo só como apoio: o MATCH do LLM segue ao revisor humano.

A política v3.1 de quatro ações e o revisor LLM simulado servem de contraste. Na
tese, a automação não reduziu a revisão manual além da ordenação e acrescentou
falsos positivos. Nenhuma função aqui chama um LLM real.

Tempo estimado na base padrão de ``synthetic_linkage.gerar_base`` (cerca de 30
mil potenciais pares), extrapolado da medição em 25.198 potenciais pares (31,5 s
com 300 árvores): ``prob_calibrada_oof`` faz 30 ajustes de XGBoost (5 externos x
5 internos + 5 finais), cerca de 38 s; ``prob_por_linha`` custa o mesmo; o
bootstrap com 2.000 réplicas, de 2 a 4 s; a política v3.1, menos de 1 s. O tempo
cresce em proporção ao número de árvores: ``n_estimators=200`` corta cerca de um
terço.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib.resources import files

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from synthetic_linkage import SUBESCORES

from gzcmd_record_linkage.bands import BandAssigner
from gzcmd_record_linkage.config import load_config
from gzcmd_record_linkage.guardrails import apply_guardrails
from gzcmd_record_linkage.policy_engine import (
    Budget,
    Costs,
    HumanBudget,
    HumanError,
    LLMError,
    PolicyEngineV3,
)
from gzcmd_record_linkage.ranking import rrf, validate_order

__all__ = [
    "ACOES_V31",
    "ATRIBUTOS",
    "CUSTOS_POR_MODO",
    "C_H",
    "C_LLM",
    "E_HUMANO",
    "CONTAGENS_LLM_HAIKU_V4",
    "N_ESTIMATORS_TESE",
    "ORCAMENTO_LLM_TESE",
    "POTENCIAIS_PARES_ABAIXO_TESE",
    "TAXAS_LLM_HAIKU_V4",
    "Particao",
    "atributos",
    "bootstrap_diferenca",
    "curva_esforco",
    "faixas_da_nota",
    "mesma_carga",
    "meta_da_replica",
    "orcamento_llm_escalado",
    "ordem_por_escore",
    "ordenacoes",
    "particoes_aninhadas",
    "politica_v31",
    "prob_calibrada_oof",
    "prob_por_linha",
    "recuperados_com_carga",
    "revisoes_para_meta",
    "revisor_llm_simulado",
    "rotulo",
    "taxas_llm_por_faixa",
]

# Os 30 atributos do classificador da tese: 29 subescores e a nota final.
ATRIBUTOS: tuple[str, ...] = (*SUBESCORES, "nota final")
# Número de árvores do XGBoost da tese (QW1); é o padrão das funções abaixo.
N_ESTIMATORS_TESE = 300

# Custos da política v3.1 na tese, em revisões humanas equivalentes.
C_H = 1.0
C_LLM = 0.005
E_HUMANO = 0.05  # erro humano hipotético, igual para FP e FN
CUSTOS_POR_MODO: dict[str, tuple[float, float]] = {
    "vigilancia": (10.0, 50.0),  # (C_FP, C_FN): prioriza a sensibilidade
    "confirmacao": (100.0, 20.0),  # prioriza o VPP das decisões positivas
}
ORCAMENTO_LLM_TESE: dict[str, int] = {"vigilancia": 2000, "confirmacao": 1000}
POTENCIAIS_PARES_ABAIXO_TESE = 61_485
ACOES_V31: tuple[str, ...] = ("MATCH", "NONMATCH", "LLM_REVIEW", "DEFERRED")

# Contagens agregadas do revisor de referência da tese (Claude Haiku 5.5,
# instruções v4, dossiê sem as saídas do motor), por modo e faixa da nota final:
# (decisões, pares verdadeiros, não pares, FP, FN, UNSURE). Somam três coortes:
# 1.774 decisões na vigilância e 459 na confirmação. Fonte: repositório
# phd-research, gzcmd/error_rates_by_band_v4.yaml (prompt: v4, dossie: sem_mo),
# bloco by_mode.<modo>.models.claude-haiku-5-5.estimates_by_band. Só agregados,
# sem dado por par. As linhas claude-haiku-5-5 de
# data/taxas_erro_revisor_por_banda.csv vêm de outra coorte (1.410 e 435
# decisões, data/sprint3b/anthropic/) e não são usadas aqui. Faixas ausentes não
# tiveram decisões.
CONTAGENS_LLM_HAIKU_V4: dict[str, dict[str, tuple[int, int, int, int, int, int]]] = {
    "vigilancia": {
        "grey_low": (95, 2, 93, 7, 2, 7),
        "grey_mid": (596, 43, 553, 14, 2, 35),
        "grey_high": (963, 43, 920, 26, 4, 44),
        "near_high": (98, 34, 64, 2, 10, 7),
        "high": (22, 22, 0, 0, 5, 3),
    },
    "confirmacao": {
        "grey_high": (298, 22, 276, 4, 3, 9),
        "near_high": (98, 34, 64, 2, 8, 8),
        "high": (63, 61, 2, 0, 11, 9),
    },
}

Auditoria = Callable[[int, str, np.ndarray, np.ndarray], None]


# ---------------------------------------------------------------------------
# Atributos e rótulo
# ---------------------------------------------------------------------------


def rotulo(df: pd.DataFrame, y_col: str = "PAR") -> np.ndarray:
    """Classe segundo a referência operacional: 1 se PAR vale 1 ou 2, senão 0."""
    return pd.to_numeric(df[y_col], errors="coerce").isin([1, 2]).to_numpy(dtype=int)


def atributos(df: pd.DataFrame) -> pd.DataFrame:
    """Os 30 atributos do classificador: 29 subescores e a nota final do CdR.

    Ficam de fora identificadores, passo, escore do OpenRecLink, referência
    operacional e verdade latente. Valores ausentes viram zero, como no CdR.
    """
    faltantes = [coluna for coluna in ATRIBUTOS if coluna not in df.columns]
    if faltantes:
        raise KeyError(f"colunas ausentes: {faltantes}")
    bloco = df.loc[:, list(ATRIBUTOS)].apply(pd.to_numeric, errors="coerce")
    return bloco.astype(float).fillna(0.0)


# ---------------------------------------------------------------------------
# Probabilidade calibrada fora da amostra
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class Particao:
    """Uma partição externa e suas partições internas (índices posicionais).

    ``internas`` traz pares (ajuste, previsão) de índices absolutos, todos
    contidos em ``treino``.
    """

    fold: int
    treino: np.ndarray
    teste: np.ndarray
    internas: tuple[tuple[np.ndarray, np.ndarray], ...]


def _divisor(grupos: np.ndarray | None, n_splits: int, seed: int):
    if grupos is None:
        return StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    return StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed)


def _checar_separacao(
    y: np.ndarray, grupos: np.ndarray | None, ajuste: np.ndarray, previsao: np.ndarray
) -> None:
    if len(np.unique(y[ajuste])) != 2:
        raise ValueError("partição de ajuste sem as duas classes")
    if grupos is not None and np.intersect1d(grupos[ajuste], grupos[previsao]).size:
        raise ValueError("vazamento: o mesmo grupo no ajuste e na previsão")


def particoes_aninhadas(
    y: np.ndarray,
    grupos: np.ndarray | None = None,
    *,
    n_externas: int = 5,
    n_internas: int = 5,
    seed: int = 42,
) -> list[Particao]:
    """Partições da validação cruzada aninhada, estratificadas pela classe.

    Com ``grupos`` (REFREC), nenhum óbito aparece no ajuste e na previsão da
    mesma partição, nos dois níveis. Sem ``grupos``, a partição é por potencial
    par, que deixa potenciais pares do mesmo óbito dos dois lados. As sementes
    seguem a tese: ``seed`` no nível externo e ``seed + fold + 1`` no interno.
    """
    y = np.asarray(y, dtype=int)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("y precisa conter as duas classes (0 e 1)")
    if grupos is not None:
        grupos = np.asarray(grupos)
        if len(grupos) != len(y):
            raise ValueError("grupos e y precisam ter o mesmo tamanho")
    x_vazio = np.zeros((len(y), 1))
    externo = _divisor(grupos, n_externas, seed)
    particoes = []
    for fold, (treino, teste) in enumerate(externo.split(x_vazio, y, grupos)):
        _checar_separacao(y, grupos, treino, teste)
        interno = _divisor(grupos, n_internas, seed + fold + 1)
        grupos_treino = None if grupos is None else grupos[treino]
        internas = []
        for ajuste_rel, previsao_rel in interno.split(
            x_vazio[treino], y[treino], grupos_treino
        ):
            ajuste, previsao = treino[ajuste_rel], treino[previsao_rel]
            _checar_separacao(y, grupos, ajuste, previsao)
            internas.append((ajuste, previsao))
        particoes.append(Particao(fold, treino, teste, tuple(internas)))
    return particoes


def _modelo_xgb(
    y: np.ndarray,
    *,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    seed: int,
    n_jobs: int | None,
):
    """XGBoost com os hiperparâmetros da tese (QW1), em CPU."""
    from xgboost import XGBClassifier

    positivos = int(np.sum(y))
    return XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=(len(y) - positivos) / positivos,
        tree_method="hist",
        device="cpu",
        eval_metric="logloss",
        n_jobs=n_jobs,
        random_state=seed,
        verbosity=0,
    )


def _oof_aninhado(
    df: pd.DataFrame,
    *,
    y_col: str,
    grupos: str | None,
    n_externas: int,
    n_internas: int,
    seed: int,
    parametros_xgb: dict,
    auditoria: Auditoria | None,
) -> pd.Series:
    x = atributos(df).to_numpy(dtype=float)
    y = rotulo(df, y_col)
    g = None if grupos is None else df[grupos].to_numpy()
    p_final = np.full(len(y), np.nan)
    particoes = particoes_aninhadas(
        y, g, n_externas=n_externas, n_internas=n_internas, seed=seed
    )
    for parte in particoes:
        # 1) Probabilidades internas fora da amostra, só no treino externo.
        p_interna = np.full(len(y), np.nan)
        for ajuste, previsao in parte.internas:
            modelo = _modelo_xgb(y[ajuste], seed=seed, **parametros_xgb)
            modelo.fit(x[ajuste], y[ajuste])
            p_interna[previsao] = modelo.predict_proba(x[previsao])[:, 1]
            if auditoria is not None:
                auditoria(parte.fold, "interna", ajuste, previsao)
        if not np.isfinite(p_interna[parte.treino]).all():
            raise ValueError("previsões internas incompletas")
        # 2) Calibrador isotônico ajustado nessas probabilidades internas.
        calibrador = IsotonicRegression(out_of_bounds="clip")
        calibrador.fit(p_interna[parte.treino], y[parte.treino])
        # 3) Modelo reajustado no treino externo; calibrado no teste externo.
        modelo = _modelo_xgb(y[parte.treino], seed=seed, **parametros_xgb)
        modelo.fit(x[parte.treino], y[parte.treino])
        bruta = modelo.predict_proba(x[parte.teste])[:, 1]
        p_final[parte.teste] = calibrador.predict(bruta)
        if auditoria is not None:
            auditoria(parte.fold, "externa", parte.treino, parte.teste)
    if not np.isfinite(p_final).all():
        raise ValueError("previsões externas incompletas")
    return pd.Series(p_final, index=df.index, name="p_cal_oof")


def prob_calibrada_oof(
    df: pd.DataFrame,
    y_col: str = "PAR",
    grupos: str = "REFREC",
    n_externas: int = 5,
    n_internas: int = 5,
    seed: int = 42,
    n_estimators: int = N_ESTIMATORS_TESE,
    *,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    n_jobs: int | None = None,
    auditoria: Auditoria | None = None,
) -> pd.Series:
    """Probabilidade calibrada fora da amostra de cada potencial par (O3).

    Validação cruzada aninhada agrupada por ``grupos`` (REFREC) nos dois níveis.
    Em cada partição externa: (1) XGBoost nas partições internas produz
    probabilidades fora da amostra do treino; (2) a regressão isotônica é
    ajustada nelas; (3) o XGBoost reajustado em todo o treino externo pontua o
    teste externo, e o calibrador transforma esses escores. Nenhum ajuste vê o
    teste externo. O alvo é ``PAR`` em 1 ou 2 (referência operacional).

    ``auditoria(fold, etapa, ajuste, previsao)``, se informada, é chamada a cada
    ajuste com os índices posicionais usados; serve para testar o agrupamento.
    """
    return _oof_aninhado(
        df,
        y_col=y_col,
        grupos=grupos,
        n_externas=n_externas,
        n_internas=n_internas,
        seed=seed,
        parametros_xgb={
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "n_jobs": n_jobs,
        },
        auditoria=auditoria,
    )


def prob_por_linha(
    df: pd.DataFrame,
    y_col: str = "PAR",
    n_externas: int = 5,
    n_internas: int = 5,
    seed: int = 42,
    n_estimators: int = N_ESTIMATORS_TESE,
    *,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    n_jobs: int | None = None,
    auditoria: Auditoria | None = None,
) -> pd.Series:
    """Mesmo modelo de ``prob_calibrada_oof``, com partição por potencial par.

    Serve para mostrar o otimismo da partição por linha: potenciais pares do
    mesmo óbito caem no treino e no teste, e as métricas melhoram sem que o
    modelo generalize melhor para óbitos novos.
    """
    return _oof_aninhado(
        df,
        y_col=y_col,
        grupos=None,
        n_externas=n_externas,
        n_internas=n_internas,
        seed=seed,
        parametros_xgb={
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "n_jobs": n_jobs,
        },
        auditoria=auditoria,
    )


# ---------------------------------------------------------------------------
# Ordenações da fila e curva de esforço
# ---------------------------------------------------------------------------


def _alinhar(valores: pd.Series | np.ndarray, df: pd.DataFrame) -> np.ndarray:
    """Alinha uma Series pelo índice de ``df`` ou confere o tamanho de um vetor."""
    if isinstance(valores, pd.Series):
        if not valores.index.is_unique or not df.index.isin(valores.index).all():
            raise ValueError("a Series precisa cobrir o índice de df, sem repetição")
        vetor = valores.loc[df.index].to_numpy(dtype=float)
    else:
        vetor = np.asarray(valores, dtype=float)
        if vetor.shape != (len(df),):
            raise ValueError("o vetor precisa ter uma posição por linha de df")
    if not np.isfinite(vetor).all():
        raise ValueError("valores precisam ser finitos")
    return vetor


def ordem_por_escore(escores: pd.Series | np.ndarray, seed: int = 42) -> np.ndarray:
    """Posições ordenadas do maior para o menor escore.

    Empates são desfeitos por uma permutação aleatória com semente fixa, a mesma
    regra de ``gzcmd_record_linkage.ranking``; nunca pela ordem das linhas nem
    pelo rótulo.
    """
    valores = np.asarray(escores, dtype=float)
    if not np.isfinite(valores).all():
        raise ValueError("escores precisam ser finitos")
    prioridade = np.argsort(np.random.default_rng(seed).permutation(len(valores)))
    return np.lexsort((prioridade, -valores)).astype(np.int64)


def ordenacoes(
    df_abaixo: pd.DataFrame,
    p_cal: pd.Series | np.ndarray,
    *,
    seed: int = 42,
    k: float = 60,
) -> dict[str, np.ndarray]:
    """Filas de revisão na *grey area*, como posições de ``df_abaixo``.

    * O1: escore do OpenRecLink, decrescente;
    * O2: nota final do CdR, decrescente;
    * O3: probabilidade calibrada fora da amostra, decrescente;
    * F4: fusão por posição recíproca (RRF, k = 60) de O3 e O1, pelo ``rrf`` do
      pacote. Os empates seguem ``ordem_por_escore`` com a mesma semente.

    ``p_cal`` pode ser a Series da base inteira (alinhada pelo índice) ou um
    vetor com uma posição por linha de ``df_abaixo``.
    """
    p = _alinhar(p_cal, df_abaixo)
    ordens = {
        "O1": ordem_por_escore(df_abaixo["escore_orl"], seed),
        "O2": ordem_por_escore(df_abaixo["nota final"], seed),
        "O3": ordem_por_escore(p, seed),
    }
    ordens["F4"] = rrf(
        [ordens["O3"], ordens["O1"]], k=k, seed=seed, input_kind="orders"
    )
    return ordens


def curva_esforco(ordem: np.ndarray, y: np.ndarray) -> pd.Series:
    """Pares verdadeiros recuperados após cada revisão, seguindo ``ordem``.

    ``y`` é posicional (valores maiores que zero contam como par verdadeiro).
    Supõe revisor perfeito: todo par verdadeiro revisado é recuperado. O índice
    da Series é o número de revisões (1, 2, ...).
    """
    rotulos = np.asarray(y) > 0
    ordem = np.asarray(ordem)
    validate_order(ordem, len(rotulos))
    acumulado = np.cumsum(rotulos[ordem].astype(int))
    indice = pd.RangeIndex(1, len(rotulos) + 1, name="revisões")
    return pd.Series(acumulado, index=indice, name="pares verdadeiros recuperados")


def revisoes_para_meta(curva: pd.Series | np.ndarray, meta: int) -> int:
    """Menor número de revisões que recupera ``meta`` pares verdadeiros."""
    valores = np.asarray(curva)
    if meta < 1:
        raise ValueError("a meta precisa ser de pelo menos um par verdadeiro")
    if len(valores) == 0 or meta > valores[-1]:
        alcance = int(valores[-1]) if len(valores) else 0
        raise ValueError(f"meta {meta} inalcançável: a fila recupera {alcance}")
    return int(np.searchsorted(valores, meta, side="left") + 1)


def recuperados_com_carga(curva: pd.Series | np.ndarray, k: int) -> int:
    """Pares verdadeiros recuperados com ``k`` revisões (k além da fila: todos)."""
    valores = np.asarray(curva)
    if k < 0:
        raise ValueError("a carga não pode ser negativa")
    if k == 0 or len(valores) == 0:
        return 0
    return int(valores[min(k, len(valores)) - 1])


# ---------------------------------------------------------------------------
# Bootstrap agrupado por óbito
# ---------------------------------------------------------------------------


def meta_da_replica(meta: int, total: int, total_replica: int) -> int:
    """Regra híbrida da tese para a meta de cada réplica do bootstrap.

    Vale a meta nominal quando a réplica tem pelo menos ``total`` pares
    verdadeiros (o total da amostra original); com menos, vale a mesma fração
    ``meta / total`` dos pares verdadeiros da réplica, arredondada para cima.
    A tese usou frações arredondadas (0,25; 0,64; 1,0); aqui a fração é exata,
    em aritmética inteira. A regra muda o estimando entre réplicas.
    """
    if total_replica >= total:
        return meta
    return -(-meta * total_replica // total)


def _revisoes_na_replica(rotulos: np.ndarray, copias: np.ndarray, meta: int) -> int:
    # Cada cópia de um potencial par conta como uma revisão e fica junto do
    # original, na mesma posição da fila.
    acumulado = np.cumsum(np.repeat(rotulos, copias))
    return int(np.searchsorted(acumulado, meta, side="left") + 1)


def bootstrap_diferenca(
    df_abaixo: pd.DataFrame,
    ordem_a: np.ndarray,
    ordem_b: np.ndarray,
    meta: int,
    grupos: str = "REFREC",
    n: int = 2000,
    seed: int = 42,
    *,
    y_col: str = "PAR",
) -> pd.Series:
    """Diferença de revisões (A menos B) para atingir ``meta``, por bootstrap.

    Sorteiam-se, com reposição, os óbitos (``grupos``) com potenciais pares em
    ``df_abaixo``; cada potencial par entra tantas vezes quanto seu óbito foi
    sorteado, e as duas filas são lidas na mesma réplica (bootstrap pareado). O
    modelo não é reajustado e as filas não são recalculadas: cada cópia herda a
    posição do original (na tese, as fusões eram refeitas sobre as filas
    reamostradas). Réplicas sem par verdadeiro são descartadas; a meta de cada
    réplica segue ``meta_da_replica``. Valores negativos favorecem A.
    """
    y = rotulo(df_abaixo, y_col)
    ordem_a = np.asarray(ordem_a)
    ordem_b = np.asarray(ordem_b)
    validate_order(ordem_a, len(y))
    validate_order(ordem_b, len(y))
    total = int(y.sum())
    if not 1 <= meta <= total:
        raise ValueError(f"meta deve estar entre 1 e {total} pares verdadeiros")
    codigos, unicos = pd.factorize(df_abaixo[grupos])
    n_grupos = len(unicos)
    rotulos_a, codigos_a = y[ordem_a], codigos[ordem_a]
    rotulos_b, codigos_b = y[ordem_b], codigos[ordem_b]
    rng = np.random.default_rng(seed)
    diferencas = []
    for _ in range(n):
        sorteio = rng.integers(0, n_grupos, size=n_grupos)
        copias = np.bincount(sorteio, minlength=n_grupos)
        total_replica = int((y * copias[codigos]).sum())
        if total_replica == 0:
            continue
        meta_r = meta_da_replica(meta, total, total_replica)
        revisoes_a = _revisoes_na_replica(rotulos_a, copias[codigos_a], meta_r)
        revisoes_b = _revisoes_na_replica(rotulos_b, copias[codigos_b], meta_r)
        diferencas.append(revisoes_a - revisoes_b)
    if not diferencas:
        raise ValueError("nenhuma réplica teve par verdadeiro")
    valores = np.asarray(diferencas, dtype=float)
    inferior, mediana, superior = np.quantile(valores, [0.025, 0.5, 0.975])
    return pd.Series(
        {
            "meta": meta,
            "réplicas válidas": len(valores),
            "diferença mediana (A menos B)": float(mediana),
            "IC95 inferior": float(inferior),
            "IC95 superior": float(superior),
            "proporção em que A exige menos": float((valores < 0).mean()),
        },
        name="bootstrap agrupado por óbito",
    )


# ---------------------------------------------------------------------------
# Política v3.1 e revisor LLM simulado (contraste)
# ---------------------------------------------------------------------------


def _config_padrao():
    caminho = files("gzcmd_record_linkage") / "gzcmd_v3_config.yaml"
    return load_config(str(caminho))


def faixas_da_nota(nota: pd.Series) -> pd.Series:
    """Faixas da nota final do motor (low, grey_low, ..., high), do YAML padrão."""
    return BandAssigner.from_config(_config_padrao()).assign(nota)


def taxas_llm_por_faixa(modo: str) -> dict[str, dict[str, float]]:
    """Taxas do revisor de referência da tese por faixa, a partir das contagens.

    Definições dos agregados da tese: ``e_fp`` = FP / não pares, ``e_fn`` = FN /
    pares verdadeiros (UNSURE fica nos denominadores) e ``e_unsure`` = UNSURE /
    decisões, em ``CONTAGENS_LLM_HAIKU_V4[modo]``. Regra de reserva, como na
    tese: com denominador vazio, ``e_fp`` e ``e_fn`` voltam às proporções fixas
    do YAML padrão do pacote; ``e_unsure`` sem decisões na faixa usa a
    proporção de UNSURE do modo inteiro. Denominadores pequenos, mas não
    vazios, ficam como estão (por exemplo, ``e_fn`` = 2/2 = 1,0 em grey_low na
    vigilância).
    """
    contagens = CONTAGENS_LLM_HAIKU_V4[modo]
    total_decisoes = sum(linha[0] for linha in contagens.values())
    total_unsure = sum(linha[5] for linha in contagens.values())
    unsure_do_modo = total_unsure / total_decisoes
    fixas = _config_padrao().llm_review.error_rates_by_band
    taxas = {}
    for faixa, fixa in fixas.items():
        decisoes, pares, nao_pares, fp, fn, unsure = contagens.get(faixa, (0,) * 6)
        taxas[faixa] = {
            "e_fp": fp / nao_pares if nao_pares else float(fixa["e_fp"]),
            "e_fn": fn / pares if pares else float(fixa["e_fn"]),
            "e_unsure": unsure / decisoes if decisoes else unsure_do_modo,
        }
    return taxas


TAXAS_LLM_HAIKU_V4: dict[str, dict[str, dict[str, float]]] = {
    modo: taxas_llm_por_faixa(modo) for modo in CONTAGENS_LLM_HAIKU_V4
}


def orcamento_llm_escalado(modo: str, n_abaixo: int) -> int:
    """Orçamento do LLM da tese (2.000 ou 1.000) na escala da base sintética.

    Proporcional a ``n_abaixo / 61.485`` (potenciais pares abaixo do corte na
    tese), com mínimo de 1.
    """
    escala = n_abaixo / POTENCIAIS_PARES_ABAIXO_TESE
    return max(1, int(round(ORCAMENTO_LLM_TESE[modo] * escala)))


def revisor_llm_simulado(
    df: pd.DataFrame,
    taxas_por_faixa: Mapping[str, Mapping[str, float]] | None = None,
    seed: int = 42,
    *,
    modo: str = "vigilancia",
    y_col: str = "PAR",
    faixa_col: str = "band",
) -> pd.Series:
    """Revisor LLM SIMULADO: não é um LLM real e erra de propósito.

    Sem ``taxas_por_faixa``, usa ``TAXAS_LLM_HAIKU_V4[modo]``. Para cada
    potencial par, com as taxas da sua faixa da nota final: um não par recebe
    MATCH (falso positivo) com probabilidade ``e_fp``; um par verdadeiro recebe
    NONMATCH com probabilidade ``e_fn``; UNSURE sai com probabilidade
    ``e_unsure``, truncada quando ``e + e_unsure`` passa de 1; o restante recebe
    a resposta correta. Assim ``e_fp`` e ``e_fn`` têm a definição dos agregados
    (UNSURE nos denominadores). A classe vem de ``y_col`` (referência
    operacional). Serve só para mostrar que aceitar o MATCH do LLM introduz
    falsos positivos.
    """
    taxas = TAXAS_LLM_HAIKU_V4[modo] if taxas_por_faixa is None else taxas_por_faixa
    faixas = df[faixa_col].astype(str).to_numpy()
    faltantes = set(faixas) - set(taxas)
    if faltantes:
        raise ValueError(f"faixas sem taxas do LLM: {sorted(faltantes)}")
    e_fp = np.array([taxas[faixa]["e_fp"] for faixa in faixas], dtype=float)
    e_fn = np.array([taxas[faixa]["e_fn"] for faixa in faixas], dtype=float)
    e_unsure = np.array([taxas[faixa]["e_unsure"] for faixa in faixas], dtype=float)
    par_verdadeiro = rotulo(df, y_col).astype(bool)
    erro = np.where(par_verdadeiro, e_fn, e_fp)
    errada = np.where(par_verdadeiro, "NONMATCH", "MATCH")
    correta = np.where(par_verdadeiro, "MATCH", "NONMATCH")
    sorteio = np.random.default_rng(seed).random(len(df))
    limite_unsure = np.minimum(erro + e_unsure, 1.0)
    resposta = np.select(
        [sorteio < erro, sorteio < limite_unsure], [errada, "UNSURE"], correta
    )
    return pd.Series(resposta, index=df.index, name="resposta_llm", dtype=object)


def _destinos(acao: pd.Series, resposta: pd.Series) -> pd.Series:
    condicoes = [
        acao.eq("MATCH").to_numpy(),
        acao.eq("NONMATCH").to_numpy(),
        acao.eq("DEFERRED").to_numpy(),
        resposta.eq("MATCH").to_numpy(),
        resposta.eq("NONMATCH").to_numpy(),
        resposta.eq("UNSURE").to_numpy(),
    ]
    rotulos = [
        "vínculo automático",
        "rejeição automática",
        "revisão humana",
        "vínculo pelo LLM",
        "rejeição pelo LLM",
        "revisão humana (LLM inconclusivo)",
    ]
    destino = np.select(condicoes, rotulos, default="sem destino")
    return pd.Series(destino, index=acao.index, name="destino", dtype=object)


def _resumo_politica(decisoes: pd.DataFrame, y_col: str, orcamento: int) -> pd.Series:
    par_verdadeiro = rotulo(decisoes, y_col).astype(bool)
    acao = decisoes["action"]
    resposta = decisoes["resposta_llm"]
    auto_match = acao.eq("MATCH").to_numpy()
    auto_non = acao.eq("NONMATCH").to_numpy()
    llm_match = resposta.eq("MATCH").to_numpy()
    llm_non = resposta.eq("NONMATCH").to_numpy()
    llm_unsure = resposta.eq("UNSURE").to_numpy()
    humano = acao.eq("DEFERRED").to_numpy() | llm_unsure
    rejeitados = auto_non | llm_non
    return pd.Series(
        {
            "potenciais pares": len(decisoes),
            "pares verdadeiros (referência)": int(par_verdadeiro.sum()),
            "aceitos automaticamente (MATCH)": int(auto_match.sum()),
            "rejeitados automaticamente (NONMATCH)": int(auto_non.sum()),
            "enviados ao humano (DEFERRED)": int(acao.eq("DEFERRED").sum()),
            "enviados ao LLM (LLM_REVIEW)": int(acao.eq("LLM_REVIEW").sum()),
            "orçamento do LLM": int(orcamento),
            "LLM: MATCH": int(llm_match.sum()),
            "LLM: NONMATCH": int(llm_non.sum()),
            "LLM: UNSURE": int(llm_unsure.sum()),
            "revisões humanas": int(humano.sum()),
            "falsos positivos automáticos": int((auto_match & ~par_verdadeiro).sum()),
            "falsos positivos do LLM": int((llm_match & ~par_verdadeiro).sum()),
            "pares verdadeiros aceitos sem humano": int(
                ((auto_match | llm_match) & par_verdadeiro).sum()
            ),
            "pares verdadeiros na revisão humana": int((humano & par_verdadeiro).sum()),
            "pares verdadeiros rejeitados": int((rejeitados & par_verdadeiro).sum()),
        },
        name="política v3.1",
    )


def politica_v31(
    df_abaixo: pd.DataFrame,
    p_cal: pd.Series | np.ndarray,
    modo: str = "vigilancia",
    *,
    taxas_llm: Mapping[str, Mapping[str, float]] | None = None,
    orcamento_llm: int | None = None,
    seed: int = 42,
    y_col: str = "PAR",
) -> tuple[pd.DataFrame, pd.Series]:
    """Política v3.1 de quatro ações do pacote, aplicada na *grey area*.

    Guardas da v3.1 (nota final < 5: NONMATCH; nota final >= 9 com p >= 0,9:
    MATCH) e, fora delas, a ação de menor *expected loss* entre aceitar como
    vínculo (MATCH), rejeitar (NONMATCH), revisão por LLM (LLM_REVIEW) e revisão
    humana (DEFERRED). Custos da tese: C_H = 1; C_LLM = 0,005; C_FP/C_FN = 10/50
    na vigilância e 100/20 na confirmação; erro humano hipotético de 0,05.
    Orçamento humano ilimitado; orçamento do LLM de ``orcamento_llm_escalado``.
    Taxas do LLM, no custo esperado e no revisor simulado: as do revisor de
    referência da tese no mesmo modo (``TAXAS_LLM_HAIKU_V4``), salvo
    ``taxas_llm``.

    A política não vê ``y_col``; a referência entra depois, para simular o LLM
    (``revisor_llm_simulado``) e resumir o resultado. Retorna as decisões,
    alinhadas a ``df_abaixo``, e um resumo com contagens e falsos positivos.
    """
    if modo not in CUSTOS_POR_MODO:
        raise ValueError(f"modo deve ser um de {sorted(CUSTOS_POR_MODO)}")
    taxas = TAXAS_LLM_HAIKU_V4[modo] if taxas_llm is None else taxas_llm
    nota = pd.to_numeric(df_abaixo["nota final"], errors="coerce")
    quadro = pd.DataFrame(
        {"nota final": nota.to_numpy(dtype=float), "p_cal": _alinhar(p_cal, df_abaixo)}
    )
    quadro["band"] = faixas_da_nota(quadro["nota final"])
    if quadro["band"].isna().any():
        raise ValueError("nota final fora das faixas do motor")
    faltantes = set(quadro["band"]) - set(taxas)
    if faltantes:
        raise ValueError(f"faixas sem taxas do LLM: {sorted(faltantes)}")
    quadro["guardrail"] = apply_guardrails(quadro, policy_version="v3.1").guardrail
    c_fp, c_fn = CUSTOS_POR_MODO[modo]
    limite = (
        orcamento_llm_escalado(modo, len(quadro))
        if orcamento_llm is None
        else int(orcamento_llm)
    )
    motor = PolicyEngineV3(
        Costs(
            false_positive=c_fp,
            false_negative=c_fn,
            llm_review=C_LLM,
            human_review=C_H,
        ),
        {
            faixa: LLMError(e_fp=t["e_fp"], e_fn=t["e_fn"], e_unsure=t["e_unsure"])
            for faixa, t in taxas.items()
        },
        Budget(llm_max=limite),
        policy_version="v3.1",
        human_error=HumanError(E_HUMANO, E_HUMANO),
        human_budget=HumanBudget(None),
    )
    saida = motor.triage(quadro)
    saida[y_col] = df_abaixo[y_col].to_numpy()
    no_llm = saida["action"].eq("LLM_REVIEW")
    respostas = revisor_llm_simulado(saida.loc[no_llm], taxas, seed, y_col=y_col)
    saida["resposta_llm"] = respostas.reindex(saida.index)
    saida["destino"] = _destinos(saida["action"], saida["resposta_llm"])
    colunas = [
        "nota final",
        "p_cal",
        "band",
        "guardrail",
        "action",
        "decision_reason",
        "resposta_llm",
        "destino",
        y_col,
    ]
    decisoes = saida.loc[:, colunas].set_axis(df_abaixo.index, axis=0)
    return decisoes, _resumo_politica(decisoes, y_col, limite)


def mesma_carga(
    df_abaixo: pd.DataFrame,
    ordens: Mapping[str, np.ndarray],
    decisoes: pd.DataFrame,
    *,
    y_col: str = "PAR",
    orcamento: int | None = None,
) -> pd.DataFrame:
    """Compara, na mesma carga humana, as filas ordenadas com a política v3.1.

    Revisor humano perfeito simulado: recupera todo par verdadeiro revisado e
    não produz falso positivo. Duas leituras da política:

    * LLM aceita o MATCH: revisão humana = DEFERRED + UNSURE do LLM;
    * MATCH do LLM ao humano (variante que a tese indica caso se use o LLM,
      ainda como hipótese exploratória): revisão humana = DEFERRED + UNSURE +
      MATCH do LLM, sem aceitação independente pelo LLM.

    Cada fila recebe ``orcamento`` revisões (por padrão, a carga da variante) e
    informa também quantas revisões precisaria para igualar a variante.
    """
    if not decisoes.index.equals(df_abaixo.index):
        raise ValueError("decisoes precisa estar alinhado a df_abaixo")
    par_verdadeiro = rotulo(df_abaixo, y_col).astype(bool)
    acao = decisoes["action"].to_numpy()
    resposta = decisoes["resposta_llm"]
    auto = acao == "MATCH"
    no_llm = acao == "LLM_REVIEW"
    llm_match = resposta.eq("MATCH").to_numpy()
    humano = (acao == "DEFERRED") | resposta.eq("UNSURE").to_numpy()
    humano_variante = humano | llm_match
    recuperados_variante = int((par_verdadeiro & (auto | humano_variante)).sum())
    carga = int(humano_variante.sum()) if orcamento is None else int(orcamento)
    linhas = [
        {
            "estratégia": "v3.1, LLM aceita o MATCH",
            "revisões humanas": int(humano.sum()),
            "pares verdadeiros recuperados": int(
                (par_verdadeiro & (auto | llm_match | humano)).sum()
            ),
            "pares avaliados pelo LLM": int(no_llm.sum()),
            "falsos positivos": int((~par_verdadeiro & (auto | llm_match)).sum()),
        },
        {
            "estratégia": "v3.1, MATCH do LLM ao humano",
            "revisões humanas": int(humano_variante.sum()),
            "pares verdadeiros recuperados": recuperados_variante,
            "pares avaliados pelo LLM": int(no_llm.sum()),
            "falsos positivos": int((~par_verdadeiro & auto).sum()),
        },
    ]
    for nome, ordem in ordens.items():
        curva = curva_esforco(ordem, par_verdadeiro)
        para_igualar = (
            revisoes_para_meta(curva, recuperados_variante)
            if recuperados_variante > 0
            else 0
        )
        linhas.append(
            {
                "estratégia": f"fila {nome}",
                "revisões humanas": carga,
                "pares verdadeiros recuperados": recuperados_com_carga(curva, carga),
                "pares avaliados pelo LLM": 0,
                "falsos positivos": 0,
                "revisões para igualar a variante": para_igualar,
            }
        )
    return pd.DataFrame(linhas)
