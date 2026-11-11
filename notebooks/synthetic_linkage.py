"""Base sintética didática da fila de revisão abaixo do corte do OpenRecLink.

O gerador imita, em escala menor e sem nenhum dado real, o cenário da tese. Cada
óbito (REFREC) recebe potenciais pares (COMPREC) gerados pela blocagem em três
passos (6, 9 e 11), e no máximo um deles é o par verdadeiro. Cada potencial par
traz os 29 subescores do Comparador de Registros (CdR), com os mesmos nomes do
extrato pós-CdR, a nota final do CdR, o escore do OpenRecLink e o ponto de corte
do seu passo.

Mecanismos simulados (nenhum resultado da tese é imposto):

* qualidade do registro: entre os pares verdadeiros, uma parte vem de registros
  precários (nome com grafia diferente, nascimento ausente ou aproximado, mãe e
  endereço ausentes), concentrados no passo 11;
* o OpenRecLink e o CdR comparam os mesmos registros de modo diferente. O escore
  do OpenRecLink soma pesos de concordância, no estilo de Fellegi e Sunter, de
  uma comparação fonética do nome, do nascimento com tolerância (ausência é
  neutra; data próxima recebe peso parcial), do sexo e do município. Os
  subescores do CdR comparam fragmentos exatos do nome, o nome da mãe, padrões
  fixos de erro na data (um dígito, inversões), município e endereço, e
  registram campo ausente ou data apenas próxima como zero, igual à
  discordância;
* a nota final do CdR é uma soma ponderada fixa dos subescores, com ruído,
  levada à escala de 0 a 10 por uma potência côncava;
* há negativos difíceis: homônimos (nome igual, mãe e nascimento diferentes) e
  familiares (mesma mãe e endereço, primeiro nome e nascimento diferentes);
* o passo 11 tem chave de blocagem fraca (código fonético do último nome e
  sexo): concentra a maior parte dos potenciais pares;
* a referência operacional ``PAR`` segue a tese: o ponto de corte define
  ``PAR = 1`` para todo potencial par acima dele; abaixo dele, uma revisão
  manual de origem simulada recupera só parte dos pares verdadeiros
  (``PAR = 2``), com chance maior quanto maior o escore (viés de verificação).

As colunas ``par_verdadeiro``, ``registro_precario`` e ``tipo_par`` são a verdade
latente da simulação. A tese conhece só ``PAR``: as avaliações didáticas usam
``PAR``, e as colunas latentes servem apenas para explicar o mecanismo, nunca
como atributos.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

__all__ = [
    "COLUNAS_LATENTES",
    "PASSOS",
    "SUBESCORES",
    "SUBESCORES_DTNASC",
    "SUBESCORES_ENDERECO",
    "SUBESCORES_MAE",
    "SUBESCORES_MUNICIPIO",
    "SUBESCORES_NOME",
    "TIPOS_PAR",
    "ParametrosBase",
    "gerar_base",
    "prob_revisao_manual",
    "sem_colunas_latentes",
]

PASSOS: tuple[int, int, int] = (6, 9, 11)
_IP_9 = PASSOS.index(9)
_IP_11 = PASSOS.index(11)

_FRAGMENTOS = (
    "prim frag igual",
    "ult frag igual",
    "qtd frag iguais",
    "qtd frag raros",
    "qtd frag comuns",
    "qtd frag muito parec",
    "qtd frag abrev",
)
SUBESCORES_NOME: tuple[str, ...] = tuple(f"NOME {nome}" for nome in _FRAGMENTOS)
SUBESCORES_MAE: tuple[str, ...] = tuple(f"NOMEMAE {nome}" for nome in _FRAGMENTOS)
SUBESCORES_DTNASC: tuple[str, ...] = (
    "DTNASC dt iguais",
    "DTNASC dt ap 1digi",
    "DTNASC dt inv dia",
    "DTNASC dt inv mes",
    "DTNASC dt inv ano",
)
SUBESCORES_MUNICIPIO: tuple[str, ...] = (
    "CODMUNRES uf igual",
    "CODMUNRES uf prox",
    "CODMUNRES local igual",
    "CODMUNRES local prox",
)
SUBESCORES_ENDERECO: tuple[str, ...] = (
    "ENDERECO via igual",
    "ENDERECO via prox",
    "ENDERECO numero igual",
    "ENDERECO compl prox",
    "ENDERECO texto prox",
    "ENDERECO tokens jacc",
)
SUBESCORES: tuple[str, ...] = (
    SUBESCORES_NOME
    + SUBESCORES_MAE
    + SUBESCORES_DTNASC
    + SUBESCORES_MUNICIPIO
    + SUBESCORES_ENDERECO
)

TIPOS_PAR: tuple[str, ...] = ("aleatorio", "homonimo", "familiar", "verdadeiro")
_ALEATORIO, _HOMONIMO, _FAMILIAR, _VERDADEIRO = range(len(TIPOS_PAR))
COLUNAS_LATENTES: tuple[str, ...] = ("par_verdadeiro", "registro_precario", "tipo_par")

# Grafia do nome nos pares verdadeiros: (exata, erro de digitação, variante),
# por qualidade do registro. Variante = grafias diferentes de mesmo som e
# abreviações, que a comparação fonética tolera e a de fragmentos exatos não.
_P_GRAFIA_BOM = (0.75, 0.12, 0.13)
_P_GRAFIA_PRECARIO = (0.40, 0.35, 0.25)

# Estados da comparação de nascimento. Os cinco primeiros têm subescore próprio
# no CdR; "proxima" (data diferente, mas perto), "diferente" e "ausente" viram
# todos os subescores iguais a zero no CdR.
_ESTADOS_DTNASC = (
    "igual",
    "ap_1digi",
    "inv_dia",
    "inv_mes",
    "inv_ano",
    "proxima",
    "diferente",
    "ausente",
)
_P_DTNASC_BOM = np.array([0.88, 0.04, 0.015, 0.015, 0.01, 0.02, 0.01, 0.01])
_P_DTNASC_PRECARIO = np.array([0.15, 0.05, 0.02, 0.02, 0.01, 0.30, 0.05, 0.40])
_P_DTNASC_HOMONIMO = np.array([0.02, 0.02, 0.01, 0.01, 0.01, 0.08, 0.79, 0.06])
_P_DTNASC_FAMILIAR = np.array([0.00, 0.02, 0.01, 0.01, 0.03, 0.08, 0.79, 0.06])
# Uma linha por passo (6, 9, 11). No passo 6, a chave de blocagem simulada
# inclui o ano de nascimento, o que aumenta a concordância casual.
_P_DTNASC_ALEATORIO = np.array(
    [
        [0.05, 0.06, 0.02, 0.02, 0.01, 0.30, 0.48, 0.06],
        [0.005, 0.02, 0.005, 0.005, 0.005, 0.06, 0.84, 0.06],
        [0.005, 0.02, 0.005, 0.005, 0.005, 0.06, 0.84, 0.06],
    ]
)
# Peso do nascimento no escore do OpenRecLink: data próxima ganha peso parcial e
# ausência é neutra, ao contrário do CdR, que registra ambas como zero.
_PESO_ORL_DTNASC = np.array([5.0, 2.0, 1.5, 1.5, 1.0, 2.0, -3.5, 0.0])

# Ausências e mudanças por qualidade do registro (pares verdadeiros) e para os
# não pares.
_MAE_AUSENTE = {"precario": 0.80, "bom": 0.10, "nao_par": 0.20}
_ENDERECO_AUSENTE = {"precario": 0.75, "bom": 0.20, "nao_par": 0.25}
_MUNICIPIO_IGUAL = {"precario": 0.75, "bom": 0.88}

_DESLOCAMENTO_ENDERECO = (0.0, 0.05, -0.05, -0.10, 0.0, 0.0)
_RUIDO_ENDERECO = (0.15, 0.10, 0.22, 0.25, 0.12, 0.10)

# Pesos fixos da nota final simulada. O nome pesa muito (homônimos ficam com nota
# alta); fragmentos raros e nome da mãe pesam pouco, o que deixa informação para
# o classificador.
_PESOS_NOTA: dict[str, float] = {
    "NOME qtd frag iguais": 3.0,
    "NOME prim frag igual": 1.0,
    "NOME ult frag igual": 1.0,
    "NOME qtd frag muito parec": 0.9,
    "NOME qtd frag raros": 0.2,
    "NOMEMAE qtd frag iguais": 0.8,
    "NOMEMAE prim frag igual": 0.25,
    "NOMEMAE ult frag igual": 0.25,
    "DTNASC dt iguais": 1.3,
    "DTNASC dt ap 1digi": 0.8,
    "DTNASC dt inv dia": 0.6,
    "DTNASC dt inv mes": 0.6,
    "DTNASC dt inv ano": 0.5,
    "CODMUNRES local igual": 0.6,
    "CODMUNRES local prox": 0.2,
    "CODMUNRES uf igual": 0.2,
}
_PESO_NOTA_ENDERECO = 0.6
_SOMA_MAXIMA_NOTA = 10.1


@dataclass(frozen=True)
class ParametrosBase:
    """Parâmetros do gerador; tuplas por passo seguem a ordem de ``PASSOS``.

    Os valores padrão miram a ordem de grandeza da tese em escala menor, sem
    garantia: cerca de 30 mil potenciais pares; perto de 85% dos pares com
    ``PAR > 0`` acima do ponto de corte (211 de 247 na tese); de 30 a 40
    ``PAR == 2`` abaixo dele (36 na tese, entre 61.485 potenciais pares).
    """

    n_obitos: int = 6_000
    prop_obitos_com_par: float = 0.05
    # Tese: 80, 127 e 40 dos 247 pares verdadeiros nos passos 6, 9 e 11.
    prop_pares_por_passo: tuple[float, float, float] = (0.32, 0.52, 0.16)
    # Média de não pares por óbito em cada passo; o passo 11 domina o volume.
    candidatos_por_obito: tuple[float, float, float] = (0.30, 0.25, 4.4)
    # Forma da binomial negativa do passo 11 (menor = mais heterogêneo).
    forma_passo_11: float = 2.0
    prop_homonimos: float = 0.06
    prop_familiares: float = 0.04
    # Proporção de pares verdadeiros vindos de registros precários, por passo.
    prop_precarios: tuple[float, float, float] = (0.05, 0.08, 0.45)
    # Ponto de corte do escore do OpenRecLink em cada passo.
    ponto_de_corte: tuple[float, float, float] = (12.0, 12.5, 13.0)
    # Revisão manual de origem: P(recuperar) = sigmoide(a + b * (escore - corte)).
    revisao_intercepto: float = 1.0
    revisao_inclinacao: float = 0.55
    # Potência que leva a soma ponderada dos subescores à escala de 0 a 10.
    expoente_nota: float = 0.6

    def __post_init__(self) -> None:
        if self.n_obitos < 1:
            raise ValueError("n_obitos deve ser positivo")
        por_passo = {
            "prop_pares_por_passo": self.prop_pares_por_passo,
            "candidatos_por_obito": self.candidatos_por_obito,
            "prop_precarios": self.prop_precarios,
            "ponto_de_corte": self.ponto_de_corte,
        }
        for nome, valores in por_passo.items():
            if len(valores) != len(PASSOS):
                raise ValueError(f"{nome} precisa de um valor por passo {PASSOS}")
        proporcoes = [
            self.prop_obitos_com_par,
            self.prop_homonimos,
            self.prop_familiares,
            *self.prop_precarios,
        ]
        if not all(0.0 <= valor <= 1.0 for valor in proporcoes):
            raise ValueError("proporções devem estar entre 0 e 1")
        if self.prop_homonimos + self.prop_familiares > 1.0:
            raise ValueError("homônimos e familiares somam mais que 1")
        if min(self.prop_pares_por_passo) < 0 or sum(self.prop_pares_por_passo) <= 0:
            raise ValueError("prop_pares_por_passo precisa de pesos não negativos")
        if min(self.candidatos_por_obito) < 0 or self.forma_passo_11 <= 0:
            raise ValueError("médias de candidatos e forma devem ser positivas")
        if self.expoente_nota <= 0:
            raise ValueError("expoente_nota deve ser positivo")

    @property
    def corte_por_passo(self) -> dict[int, float]:
        """Ponto de corte do OpenRecLink indexado pelo número do passo."""
        return dict(zip(PASSOS, self.ponto_de_corte, strict=True))


def gerar_base(
    n_obitos: int | None = None,
    seed: int = 42,
    parametros: ParametrosBase | None = None,
) -> pd.DataFrame:
    """Gera os potenciais pares sintéticos no formato do extrato pós-CdR.

    ``n_obitos``, quando informado, substitui ``parametros.n_obitos``. A saída é
    determinística para a mesma semente e os mesmos parâmetros, e contém só
    códigos numéricos ou categóricos (nenhum nome, endereço ou data).

    Colunas:

    * ``REFREC``, ``COMPREC``, ``PASSO``: identificadores e passo de blocagem;
    * os 29 subescores de ``SUBESCORES`` e ``nota final`` (CdR, de 0 a 10);
    * ``escore_orl``: escore do OpenRecLink; ``ponto_de_corte``: corte do passo;
      ``acima_corte``: ``escore_orl >= ponto_de_corte``;
    * ``PAR``: referência operacional, como na tese. 1 = acima do ponto de corte
      (aceito pelo corte, mesmo que seja um não par: erros acima do corte ficam
      fora do alcance da referência); 2 = par verdadeiro abaixo do corte
      recuperado pela revisão manual de origem simulada; 0 = os demais,
      inclusive pares verdadeiros abaixo do corte que essa revisão não achou;
    * ``par_verdadeiro``, ``registro_precario`` e ``tipo_par``: verdade latente,
      que a tese não tem.

    O carregador do pacote (``load_comparador_csv``) e o dossiê
    (``build_dossier``) aceitam a base gravada com ``sep=";"`` e ``decimal=","``.
    Ao usar o classificador genérico do pacote, remova antes as colunas latentes
    com ``sem_colunas_latentes``, pois ele usa toda coluna numérica.
    """
    par = parametros if parametros is not None else ParametrosBase()
    if n_obitos is not None:
        par = replace(par, n_obitos=n_obitos)
    rng = np.random.default_rng(seed)

    candidatos = _potenciais_pares(rng, par)
    ip = candidatos["ip"].to_numpy()
    verdadeiro = candidatos["verdadeiro"].to_numpy(dtype=bool)
    tipo = _tipos(rng, verdadeiro, par)
    chance_precario = np.asarray(par.prop_precarios, dtype=float)[ip]
    precario = verdadeiro & (rng.random(len(tipo)) < chance_precario)

    nome, nome_orl = _bloco_nome(rng, tipo, ip, precario)
    mae = _bloco_mae(rng, tipo, precario)
    estado_dtnasc = _estado_dtnasc(rng, tipo, ip, precario)
    dtnasc = {
        coluna: (estado_dtnasc == indice).astype(int)
        for indice, coluna in enumerate(SUBESCORES_DTNASC)
    }
    municipio = _bloco_municipio(rng, tipo, ip, precario)
    sexo_igual = _sexo_igual(rng, tipo, ip)
    endereco = _bloco_endereco(
        rng, tipo, precario, municipio["CODMUNRES local igual"]
    )
    subescores = {**nome, **mae, **dtnasc, **municipio, **endereco}

    nota = _nota_final(rng, subescores, par.expoente_nota)
    escore = _escore_openreclink(rng, nome_orl, estado_dtnasc, sexo_igual, municipio)
    corte = np.asarray(par.ponto_de_corte, dtype=float)[ip]
    acima = escore >= corte
    sorteio = rng.random(len(escore))
    recuperado = sorteio < prob_revisao_manual(escore - corte, par)
    # Referência operacional como na tese: o corte define PAR = 1; abaixo dele,
    # só a revisão manual de origem cria PAR = 2.
    referencia = np.select([acima, verdadeiro & recuperado], [1, 2], 0)

    colunas: dict[str, object] = {
        "REFREC": candidatos["REFREC"].to_numpy(),
        "COMPREC": candidatos["COMPREC"].to_numpy(),
        "PASSO": np.asarray(PASSOS)[ip],
    }
    colunas.update({coluna: subescores[coluna] for coluna in SUBESCORES})
    colunas.update(
        {
            "nota final": nota,
            "escore_orl": escore,
            "ponto_de_corte": corte,
            "acima_corte": acima,
            "PAR": referencia.astype(int),
            "par_verdadeiro": verdadeiro.astype(int),
            "registro_precario": precario.astype(int),
            "tipo_par": np.asarray(TIPOS_PAR)[tipo],
        }
    )
    quadro = pd.DataFrame(colunas)
    ordenado = quadro.sort_values(["PASSO", "REFREC", "COMPREC"], kind="mergesort")
    return ordenado.reset_index(drop=True)


def prob_revisao_manual(
    distancia_ao_corte: np.ndarray | float,
    parametros: ParametrosBase | None = None,
) -> np.ndarray:
    """Probabilidade de a revisão manual de origem recuperar um par verdadeiro.

    Cresce com ``escore - corte`` (negativa abaixo do corte): pares verdadeiros
    de escore mais alto têm mais chance de virar ``PAR == 2``. É o viés de
    verificação da referência operacional; acima do corte o valor não é usado.
    """
    par = parametros if parametros is not None else ParametrosBase()
    distancia = np.asarray(distancia_ao_corte, dtype=float)
    logito = par.revisao_intercepto + par.revisao_inclinacao * distancia
    return 1.0 / (1.0 + np.exp(-logito))


def sem_colunas_latentes(df: pd.DataFrame) -> pd.DataFrame:
    """Devolve a base como a tese a conheceria: sem a verdade latente."""
    return df.drop(columns=list(COLUNAS_LATENTES), errors="ignore")


# ---------------------------------------------------------------------------
# Etapas internas
# ---------------------------------------------------------------------------


def _potenciais_pares(rng: np.random.Generator, par: ParametrosBase) -> pd.DataFrame:
    """Sorteia óbitos, notificações e potenciais pares de cada passo."""
    n = par.n_obitos
    com_par = np.flatnonzero(rng.random(n) < par.prop_obitos_com_par)
    pesos = np.asarray(par.prop_pares_por_passo, dtype=float)
    ip_par = rng.choice(len(PASSOS), size=len(com_par), p=pesos / pesos.sum())
    # As notificações 0..len(com_par)-1 são as dos pares verdadeiros; as demais
    # só aparecem como não pares. Qualquer notificação pode ser candidata de
    # vários óbitos, como na blocagem real.
    n_notificacoes = len(com_par) + 2 * n
    refs = [com_par]
    comps = [np.arange(len(com_par))]
    ips = [ip_par]
    for ip, media in enumerate(par.candidatos_por_obito):
        forma = par.forma_passo_11
        if ip == _IP_11:
            quantos = rng.negative_binomial(forma, forma / (forma + media), size=n)
        else:
            quantos = rng.poisson(media, size=n)
        ref = np.repeat(np.arange(n), quantos)
        refs.append(ref)
        comps.append(rng.integers(0, n_notificacoes, size=len(ref)))
        ips.append(np.full(len(ref), ip))
    verdadeiro = np.zeros(sum(len(ref) for ref in refs), dtype=bool)
    verdadeiro[: len(com_par)] = True
    quadro = pd.DataFrame(
        {
            "ref": np.concatenate(refs),
            "comp": np.concatenate(comps),
            "ip": np.concatenate(ips),
            "verdadeiro": verdadeiro,
        }
    )
    # Um potencial par aparece uma vez; o par verdadeiro vem primeiro e tem
    # precedência sobre um sorteio casual da mesma notificação.
    quadro = quadro.drop_duplicates(["ref", "comp"], keep="first")
    quadro = quadro.reset_index(drop=True)
    # Códigos permutados: a ordem dos identificadores não revela a classe.
    codigo_ref = rng.permutation(n)
    codigo_comp = rng.permutation(n_notificacoes)
    quadro["REFREC"] = [f"R{c:06d}" for c in codigo_ref[quadro["ref"].to_numpy()]]
    quadro["COMPREC"] = [f"C{c:07d}" for c in codigo_comp[quadro["comp"].to_numpy()]]
    return quadro


def _tipos(
    rng: np.random.Generator, verdadeiro: np.ndarray, par: ParametrosBase
) -> np.ndarray:
    sorteio = rng.random(len(verdadeiro))
    limite_familiar = par.prop_homonimos + par.prop_familiares
    tipo = np.select(
        [sorteio < par.prop_homonimos, sorteio < limite_familiar],
        [_HOMONIMO, _FAMILIAR],
        _ALEATORIO,
    )
    return np.where(verdadeiro, _VERDADEIRO, tipo)


def _bernoulli(rng: np.random.Generator, prob: np.ndarray) -> np.ndarray:
    prob = np.asarray(prob, dtype=float)
    return (rng.random(prob.shape) < prob).astype(int)


def _subescores_de_nome(
    prefixo: str,
    prim: np.ndarray,
    ult: np.ndarray,
    qtd: np.ndarray,
    raros: np.ndarray,
    comuns: np.ndarray,
    muito_parec: np.ndarray,
    abrev: np.ndarray,
) -> dict[str, np.ndarray]:
    valores = (
        prim.astype(int),
        ult.astype(int),
        np.round(qtd, 4),
        np.round(raros, 4),
        np.round(comuns, 4),
        np.round(muito_parec, 4),
        np.round(abrev, 4),
    )
    return {
        f"{prefixo} {fragmento}": valor
        for fragmento, valor in zip(_FRAGMENTOS, valores, strict=True)
    }


def _bloco_nome(
    rng: np.random.Generator,
    tipo: np.ndarray,
    ip: np.ndarray,
    precario: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Nome do falecido: subescores do CdR e comparação fonética do OpenRecLink."""
    n = len(tipo)
    verdadeiro = tipo == _VERDADEIRO
    homonimo = tipo == _HOMONIMO
    familiar = tipo == _FAMILIAR
    # Grafia nos pares verdadeiros; registros precários erram mais.
    sorteio = rng.random(n)
    p_exata = np.where(precario, _P_GRAFIA_PRECARIO[0], _P_GRAFIA_BOM[0])
    p_digitacao = np.where(precario, _P_GRAFIA_PRECARIO[1], _P_GRAFIA_BOM[1])
    exata = verdadeiro & (sorteio < p_exata)
    digitacao = verdadeiro & ~exata & (sorteio < p_exata + p_digitacao)
    variante = verdadeiro & ~exata & ~digitacao
    grupos = [exata | homonimo, digitacao, variante, familiar]

    # Comparação fonética do OpenRecLink (tolera grafia). A chave de blocagem
    # fixa o código fonético do primeiro nome nos passos 6 e 9 e o do último
    # nome no passo 11.
    no_11 = ip == _IP_11
    casual_prim = np.where(no_11, 0.15, 1.0)
    casual_ult = np.where(no_11, 1.0, 0.12)
    fon_prim = _bernoulli(rng, np.select(grupos, [1.0, 0.85, 0.90, 0.10], casual_prim))
    fon_ult = _bernoulli(rng, np.select(grupos, [1.0, 0.85, 0.85, 0.95], casual_ult))
    similaridade = np.select(
        grupos,
        [
            rng.uniform(0.92, 1.0, n),
            rng.uniform(0.85, 0.97, n),
            rng.uniform(0.78, 0.95, n),
            rng.uniform(0.55, 0.80, n),
        ],
        np.clip(
            0.45 + 0.15 * fon_prim + 0.10 * fon_ult + rng.normal(0.0, 0.06, n),
            0.0,
            1.0,
        ),
    )

    # Subescores do CdR: igualdade exata de fragmentos.
    prim = _bernoulli(rng, np.select(grupos, [0.99, 0.60, 0.50, 0.05], 0.80 * fon_prim))
    ult = _bernoulli(rng, np.select(grupos, [0.97, 0.60, 0.30, 0.90], 0.80 * fon_ult))
    alta = np.where(rng.random(n) < 0.75, 1.0, rng.uniform(0.75, 0.95, n))
    ruido = rng.normal(0.0, 0.10, n)
    casual = np.clip(0.10 + 0.30 * prim + 0.25 * ult + ruido, 0.0, 1.0)
    qtd = np.select(
        grupos,
        [
            alta,
            rng.uniform(0.40, 0.80, n),
            rng.uniform(0.20, 0.50, n),
            rng.uniform(0.30, 0.70, n),
        ],
        casual,
    )
    parecidos = np.clip(qtd + rng.uniform(0.0, 0.15, n), 0.0, 1.0)
    muito_parec = np.select(
        [digitacao, variante],
        [rng.uniform(0.70, 1.0, n), rng.uniform(0.40, 0.80, n)],
        parecidos,
    )
    # Homônimos compartilham sobretudo fragmentos comuns; pares verdadeiros,
    # também os raros. As faixas se sobrepõem de propósito.
    fator_raros = np.select(
        [verdadeiro, homonimo, familiar],
        [
            rng.uniform(0.40, 1.0, n),
            rng.uniform(0.0, 0.60, n),
            rng.uniform(0.30, 0.80, n),
        ],
        rng.uniform(0.0, 0.50, n),
    )
    fator_comuns = np.where(
        homonimo, rng.uniform(0.85, 1.0, n), rng.uniform(0.40, 1.0, n)
    )
    chance_abrev = np.select([variante, digitacao], [0.40, 0.10], 0.04)
    abrev = np.where(rng.random(n) < chance_abrev, rng.uniform(0.10, 0.60, n), 0.0)
    raros = qtd * fator_raros
    comuns = qtd * fator_comuns
    subescores = _subescores_de_nome(
        "NOME", prim, ult, qtd, raros, comuns, muito_parec, abrev
    )
    orl = {"fon_prim": fon_prim, "fon_ult": fon_ult, "similaridade": similaridade}
    return subescores, orl


def _bloco_mae(
    rng: np.random.Generator, tipo: np.ndarray, precario: np.ndarray
) -> dict[str, np.ndarray]:
    """Nome da mãe: concorda nos pares verdadeiros e também entre familiares."""
    n = len(tipo)
    verdadeiro = tipo == _VERDADEIRO
    mesma_mae = verdadeiro | (tipo == _FAMILIAR)
    chance_ausente = np.select(
        [precario, verdadeiro],
        [_MAE_AUSENTE["precario"], _MAE_AUSENTE["bom"]],
        _MAE_AUSENTE["nao_par"],
    )
    ausente = rng.random(n) < chance_ausente
    prim = _bernoulli(rng, np.select([verdadeiro, mesma_mae], [0.93, 0.90], 0.12))
    ult = _bernoulli(rng, np.select([verdadeiro, mesma_mae], [0.90, 0.88], 0.15))
    alta = np.where(rng.random(n) < 0.70, 1.0, rng.uniform(0.60, 0.95, n))
    qtd = np.where(mesma_mae, alta, rng.uniform(0.0, 0.35, n))
    muito_parec = np.clip(qtd + rng.uniform(0.0, 0.15, n), 0.0, 1.0)
    fator_raros = np.where(
        mesma_mae, rng.uniform(0.75, 1.0, n), rng.uniform(0.0, 0.50, n)
    )
    comuns = qtd * rng.uniform(0.40, 1.0, n)
    abrev = np.where(rng.random(n) < 0.04, rng.uniform(0.10, 0.50, n), 0.0)
    blocos = _subescores_de_nome(
        "NOMEMAE", prim, ult, qtd, qtd * fator_raros, comuns, muito_parec, abrev
    )
    # O CdR registra campo ausente como zero, sem distinguir da discordância.
    return {coluna: np.where(ausente, 0, valores) for coluna, valores in blocos.items()}


def _estado_dtnasc(
    rng: np.random.Generator,
    tipo: np.ndarray,
    ip: np.ndarray,
    precario: np.ndarray,
) -> np.ndarray:
    """Sorteia o estado da comparação de nascimento (índice em _ESTADOS_DTNASC)."""
    n = len(tipo)
    probs = np.empty((n, len(_ESTADOS_DTNASC)))
    probs[tipo == _HOMONIMO] = _P_DTNASC_HOMONIMO
    probs[tipo == _FAMILIAR] = _P_DTNASC_FAMILIAR
    probs[(tipo == _VERDADEIRO) & ~precario] = _P_DTNASC_BOM
    probs[precario] = _P_DTNASC_PRECARIO
    for indice in range(len(PASSOS)):
        probs[(ip == indice) & (tipo == _ALEATORIO)] = _P_DTNASC_ALEATORIO[indice]
    acumulada = probs.cumsum(axis=1)
    sorteio = rng.random(n)
    estado = (sorteio[:, None] >= acumulada).sum(axis=1)
    return np.minimum(estado, len(_ESTADOS_DTNASC) - 1)


def _bloco_municipio(
    rng: np.random.Generator,
    tipo: np.ndarray,
    ip: np.ndarray,
    precario: np.ndarray,
) -> dict[str, np.ndarray]:
    n = len(tipo)
    verdadeiro = tipo == _VERDADEIRO
    # O passo 9 usa o município de residência na chave de blocagem simulada.
    casual = np.where(ip == _IP_9, 0.85, 0.08)
    chance_local = np.select(
        [precario, verdadeiro, tipo == _FAMILIAR, tipo == _HOMONIMO],
        [_MUNICIPIO_IGUAL["precario"], _MUNICIPIO_IGUAL["bom"], 0.70, 0.12],
        casual,
    )
    local_igual = rng.random(n) < chance_local
    local_prox = ~local_igual & (rng.random(n) < np.where(verdadeiro, 0.50, 0.15))
    uf_igual = local_igual | local_prox | (rng.random(n) < 0.60)
    uf_prox = ~uf_igual & (rng.random(n) < 0.30)
    return {
        "CODMUNRES uf igual": uf_igual.astype(int),
        "CODMUNRES uf prox": uf_prox.astype(int),
        "CODMUNRES local igual": local_igual.astype(int),
        "CODMUNRES local prox": local_prox.astype(int),
    }


def _sexo_igual(
    rng: np.random.Generator, tipo: np.ndarray, ip: np.ndarray
) -> np.ndarray:
    """Sexo não está entre os 29 subescores; entra só no escore do OpenRecLink."""
    # A chave fraca do passo 11 inclui o sexo: todo candidato casual concorda.
    casual = np.where(ip == _IP_11, 1.0, 0.50)
    chance = np.select(
        [tipo == _VERDADEIRO, tipo == _HOMONIMO, tipo == _FAMILIAR],
        [0.98, 0.97, 0.50],
        casual,
    )
    return rng.random(len(tipo)) < chance


def _bloco_endereco(
    rng: np.random.Generator,
    tipo: np.ndarray,
    precario: np.ndarray,
    local_igual: np.ndarray,
) -> dict[str, np.ndarray]:
    n = len(tipo)
    verdadeiro = tipo == _VERDADEIRO
    mesmo_domicilio = verdadeiro | (tipo == _FAMILIAR)
    chance_ausente = np.select(
        [precario, verdadeiro],
        [_ENDERECO_AUSENTE["precario"], _ENDERECO_AUSENTE["bom"]],
        _ENDERECO_AUSENTE["nao_par"],
    )
    ausente = rng.random(n) < chance_ausente
    casual = np.clip(rng.beta(1.2, 6.0, n) + 0.10 * local_igual, 0.0, 1.0)
    base = np.where(mesmo_domicilio, rng.beta(5.0, 2.0, n), casual)
    saida: dict[str, np.ndarray] = {}
    for coluna, deslocamento, ruido in zip(
        SUBESCORES_ENDERECO, _DESLOCAMENTO_ENDERECO, _RUIDO_ENDERECO, strict=True
    ):
        valores = np.clip(base + deslocamento + rng.normal(0.0, ruido, n), 0.0, 1.0)
        saida[coluna] = np.where(ausente, 0.0, np.round(valores, 4))
    return saida


def _nota_final(
    rng: np.random.Generator, subescores: dict[str, np.ndarray], expoente: float
) -> np.ndarray:
    """Nota final do CdR simulada: soma ponderada fixa com ruído, de 0 a 10.

    A potência côncava só muda a escala (a ordem dos potenciais pares é a da
    soma): sobe as notas baixas, de modo que candidatos casuais com parte do
    nome em comum caiam perto de 4 a 5, e muitos entrem na faixa [5, 9).
    """
    n = len(subescores[SUBESCORES[0]])
    soma = np.zeros(n)
    for coluna, peso in _PESOS_NOTA.items():
        soma += peso * subescores[coluna]
    endereco = np.mean([subescores[c] for c in SUBESCORES_ENDERECO], axis=0)
    soma += _PESO_NOTA_ENDERECO * endereco + rng.normal(0.0, 0.35, n)
    proporcao = np.clip(soma, 0.0, _SOMA_MAXIMA_NOTA) / _SOMA_MAXIMA_NOTA
    return np.round(10.0 * proporcao**expoente, 2)


def _escore_openreclink(
    rng: np.random.Generator,
    nome_orl: dict[str, np.ndarray],
    estado_dtnasc: np.ndarray,
    sexo_igual: np.ndarray,
    municipio: dict[str, np.ndarray],
) -> np.ndarray:
    """Escore do OpenRecLink simulado: pesos de concordância, uma casa decimal.

    Pesos ilustrativos, não a configuração real do OpenRecLink da tese.
    """
    local_igual = municipio["CODMUNRES local igual"] == 1
    local_prox = municipio["CODMUNRES local prox"] == 1
    peso = (
        np.where(nome_orl["fon_prim"] == 1, 3.0, -2.5)
        + np.where(nome_orl["fon_ult"] == 1, 3.5, -2.5)
        + 3.0 * (nome_orl["similaridade"] - 0.6)
        + _PESO_ORL_DTNASC[estado_dtnasc]
        + np.where(sexo_igual, 0.8, -4.0)
        + np.select([local_igual, local_prox], [2.0, 0.5], -1.0)
        + rng.normal(0.0, 0.25, len(estado_dtnasc))
    )
    return np.round(peso, 1)
