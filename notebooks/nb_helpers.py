"""Cálculos e controles do notebook didático, fora da biblioteca publicada.

A calibração por âncoras é uma demonstração, não uma função de ``run_v3``.
Dependências de apresentação são importadas somente ao usar os controles.
"""

from __future__ import annotations

import os
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

from gzcmd_record_linkage.bands import BandAssigner
from gzcmd_record_linkage.calibration import PlattModel
from gzcmd_record_linkage.classifier import GZCMDClassifier
from gzcmd_record_linkage.config import BandDefinition, GZCMDConfig
from gzcmd_record_linkage.guardrails import apply_guardrails
from gzcmd_record_linkage.policy_engine import Costs, PolicyEngineV3
from gzcmd_record_linkage.runner import build_engine_from_config

__all__ = [
    "expected_calibration_error",
    "brier_score",
    "llm_review_stub",
    "reliability_points",
    "log_loss",
    "platt_newton_trace",
    "AnchorSpec",
    "load_anchor_spec",
    "anchor_masks",
    "anchor_two_point_guess",
    "AnchorFit",
    "fit_anchor_platt",
    "clamp_update",
    "simulate_anchor_windows",
    "expected_losses",
    "decision_thresholds",
    "thresholds_table",
    "build_engine",
    "review_status",
    "NOME_PESOS_LOADER",
    "MACD_JANELAS_LOADER",
    "nome_score",
    "macd_flags",
    "auc_separacao",
    "assign_bands_with_edges",
    "band_transitions",
    "guardrail_diff",
    "Deslizante",
    "Escolha",
    "Caixa",
    "interagir",
    "plot_reliability",
    "plot_platt_explorer",
    "plot_costs_explorer",
    "plot_anchor_comparison",
    "plot_bands_guardrails",
    "plot_feature_explorer",
    "plot_budget_explorer",
    "plot_anchor_windows",
    "classifier_importances",
    "model_comparison_metrics",
    "plot_model_comparison",
    "plot_classifier_importances",
    "ece_noise_floor",
    "paired_bootstrap_diff",
    "custo_esperado_por_par",
    "fmt_br",
]


def classifier_importances(clf: GZCMDClassifier) -> pd.DataFrame:
    """Média das importâncias das árvores nos folds internos, não causalidade.

    RF usa redução de impureza; XGBoost usa ganho. A sigmoide de calibração
    não tem importância por feature. Nesta versão do scikit-learn, colunas
    inteiramente vazias no treino são descartadas pelo preenchimento; usamos
    os nomes transformados, que acompanham as importâncias em cada versão.
    """
    if clf.pipeline_ is None or clf.features_ is None:
        raise ValueError("Classificador ainda não ajustado.")
    names = clf.pipeline_["imputer"].get_feature_names_out(clf.features_)
    folds = clf.pipeline_["classifier"].calibrated_classifiers_
    values = np.asarray([fold.estimator.feature_importances_ for fold in folds])
    return pd.DataFrame({
        "feature": names,
        "importância média": values.mean(axis=0),
        "desvio entre folds": values.std(axis=0),
        "folds": len(folds),
    }).sort_values("importância média", ascending=False, ignore_index=True)


def model_comparison_metrics(
    y_true, predictions, *, p_true, include_oracle: bool = False,
) -> pd.DataFrame:
    """Boletim no mesmo teste; MAE vs p* é referência da nota, não verdade geral."""
    from sklearn.metrics import average_precision_score

    predictions = dict(predictions)
    if include_oracle:
        predictions["p* (oráculo)"] = p_true
    rows = []
    for name, probabilities in predictions.items():
        y, p = _as_pair(y_true, probabilities)
        _, truth = _as_pair(y, p_true)
        rows.append({
            "modelo": name,
            "AUC-ROC": roc_auc_score(y, p),
            "AUC-PR (AP)": average_precision_score(y, p),
            "ECE": expected_calibration_error(y, p),
            "Brier": brier_score(y, p),
            "log-loss": log_loss(y, p),
            "MAE vs p* (nota)": float(np.mean(np.abs(p - truth))),
        })
    return pd.DataFrame(rows).set_index("modelo")


def plot_model_comparison(y_true, predictions, *, p_true):
    """Reliability, ROC e PR no mesmo teste, sem mostrar a figura."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for name, probabilities in predictions.items():
        points = reliability_points(y_true, probabilities)
        axes[0].plot(points["conf"], points["acc"], "o-", label=name)
        fpr, tpr, _ = roc_curve(y_true, probabilities)
        axes[1].plot(fpr, tpr, label=name)
        precision, recall, _ = precision_recall_curve(y_true, probabilities)
        axes[2].plot(recall, precision, label=name)
    axes[0].plot([0, 1], [0, 1], ":", color="gray", label="calibração ideal")
    axes[1].plot([0, 1], [0, 1], ":", color="gray", label="aleatório")
    axes[2].axhline(np.mean(y_true), linestyle=":", color="gray",
                   label="prevalência")
    labels = [
        (
            "Promessa versus frequência", "Probabilidade média prevista",
            "Frequência observada de match",
        ),
        (
            "Curva ROC", "Pessoas diferentes aceitas por engano (1 − especificidade)",
            "Sensibilidade (recall)",
        ),
        ("Curva PR", "Sensibilidade (recall)", "Precisão (VPP)"),
    ]
    for ax, (title, xlabel, ylabel) in zip(axes, labels, strict=True):
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel, xlim=(0, 1), ylim=(0, 1.02))
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig, model_comparison_metrics(y_true, predictions, p_true=p_true)


def plot_classifier_importances(classifiers, *, top_n=10):
    """Top pistas de cada modelo, média dos folds internos calibrados."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(classifiers), figsize=(14, 6), squeeze=False)
    summaries = []
    for ax, (name, clf) in zip(axes.ravel(), classifiers.items(), strict=True):
        summary = classifier_importances(clf).head(top_n).assign(modelo=name)
        summaries.append(summary)
        ordered = summary.iloc[::-1]
        ax.barh(ordered["feature"], ordered["importância média"])
        ax.set(title=f"{name}: importância por impureza/ganho, não causalidade",
               xlabel="Importância média nos folds internos", ylabel="Pista numérica")
    fig.tight_layout()
    return fig, pd.concat(summaries, ignore_index=True)


def _as_pair(
    y_true: Sequence[float] | np.ndarray | pd.Series,
    p_pred: Sequence[float] | np.ndarray | pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """Converte entradas para arrays float 1-D alinhados e validados.

    Aceita listas, ``np.ndarray`` ou ``pd.Series``. Garante mesmo tamanho,
    não-vazio e ausência de NaN, pré-condições das métricas de calibração.
    """
    y = np.asarray(y_true, dtype=float).ravel()
    p = np.asarray(p_pred, dtype=float).ravel()
    if y.shape != p.shape:
        raise ValueError(
            f"y_true e p_pred devem ter o mesmo tamanho: {y.shape} != {p.shape}"
        )
    if y.size == 0:
        raise ValueError("y_true e p_pred não podem ser vazios.")
    if np.isnan(y).any() or np.isnan(p).any():
        raise ValueError("y_true e p_pred não podem conter NaN.")
    return y, p


def expected_calibration_error(
    y_true: Sequence[float] | np.ndarray | pd.Series,
    p_pred: Sequence[float] | np.ndarray | pd.Series,
    *,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error (binning explícito). Ver DEC-08 / Fase 2.4.

    Mede o desalinhamento médio entre a *confiança* prevista e a *acurácia*
    observada, agrupando as previsões em ``n_bins`` faixas de largura igual em
    ``[0, 1]``:

    ``ECE = soma_b (|B_b| / N) * |acc(B_b) - conf(B_b)|``

    onde, para cada bin ``b``, ``conf(B_b)`` é a média de ``p_pred`` e
    ``acc(B_b)`` é a fração empírica de positivos (``y_true``). Bins vazios são
    ignorados (contribuição zero), o que torna a métrica robusta.

    Parameters
    ----------
    y_true:
        Rótulos binários em ``{0, 1}``.
    p_pred:
        Probabilidades previstas em ``[0, 1]``.
    n_bins:
        Número de faixas de largura igual (padrão 10). Deve ser >= 1.

    Returns
    -------
    float
        ECE em ``[0, 1]`` (0 = perfeitamente calibrado).
    """
    points = reliability_points(y_true, p_pred, n_bins=n_bins)
    n_total = int(points["n"].sum())
    ece = 0.0
    for row in points.itertuples(index=False):
        ece += (row.n / n_total) * row.gap
    return float(ece)


def brier_score(
    y_true: Sequence[float] | np.ndarray | pd.Series,
    p_pred: Sequence[float] | np.ndarray | pd.Series,
) -> float:
    """Brier score = média((p - y)^2). Ver DEC-08 / Fase 2.4.

    É o erro quadrático médio entre a probabilidade prevista e o rótulo. Para
    um previsor constante ``p = c`` num conjunto com prevalência ``f`` de
    positivos, vale ``f*(c-1)^2 + (1-f)*c^2``; com ``c = f`` reduz a ``f*(1-f)``.
    """
    y, p = _as_pair(y_true, p_pred)
    return float(np.mean((p - y) ** 2))


#: Taxas de erro de fallback (banda ausente em ``error_rates_by_band``).
#: Espelham a banda mais pessimista da config (``low``): e_fp=0.10, e_fn=0.15.
_FALLBACK_ERROR_RATES: dict[str, float] = {"e_fp": 0.10, "e_fn": 0.15}


def llm_review_stub(
    df_review: pd.DataFrame,
    *,
    seed: int = 42,
    error_rates_by_band: Mapping[str, Mapping[str, float]] | None = None,
    target_col: str = "TARGET",
    band_col: str = "band",
) -> pd.Series:
    """Stub determinístico da revisão clerical/LLM (R-05 / Fase 3.3).

    **Isto é uma SIMULAÇÃO, não um LLM real.** Não há chamada de rede nem modelo
    de linguagem. O stub representa um revisor *quase-oráculo*: ele "enxerga" o
    rótulo verdadeiro (``target_col``) e devolve a decisão correta na maioria das
    vezes, mas **erra com taxas dependentes da banda**, exatamente o perfil de
    confiabilidade que a config atribui ao revisor (``e_fp``/``e_fn`` por banda).
    Serve para demonstrar, de forma reprodutível e offline, o *efeito* da etapa
    de revisão sobre as métricas finais, sem depender de uma API ao vivo.

    Mecânica (determinística por ``seed``):

    - Sorteia-se ``u ~ Uniforme(0, 1)`` para cada linha, **em ordem de linha**,
      com ``np.random.default_rng(seed)`` (uma única sequência → reprodutível).
    - Para uma banda ``b``, usa ``rates = error_rates_by_band[b]`` (ou o fallback
      ``{e_fp: 0.10, e_fn: 0.15}`` se a banda não estiver no mapa).
    - Se o rótulo verdadeiro é MATCH (``target == 1``): o revisor erra (diz
      ``NONMATCH``) quando ``u < e_fn``; senão acerta (``MATCH``).
    - Se o rótulo verdadeiro é NÃO-MATCH (``target == 0``): o revisor erra (diz
      ``MATCH``) quando ``u < e_fp``; senão acerta (``NONMATCH``).

    Parameters
    ----------
    df_review:
        Subconjunto de pares roteados a ``LLM_REVIEW``. Precisa conter
        ``target_col`` (rótulo verdadeiro, ``{0, 1}``) e ``band_col`` (nome da
        banda). A ordem das linhas determina o sorteio (reprodutibilidade).
    seed:
        Semente do gerador. Mesma semente + mesmo ``df_review`` → mesma saída.
    error_rates_by_band:
        Mapa ``{banda: {"e_fp": float, "e_fn": float}}`` (tipicamente
        ``cfg.llm_review.error_rates_by_band``). Se ``None``, usa o fallback
        para todas as linhas.
    target_col, band_col:
        Nomes das colunas de rótulo verdadeiro e banda.

    Returns
    -------
    pd.Series
        Decisões em ``{"MATCH", "NONMATCH"}``, dtype ``string``, alinhadas ao
        ``df_review.index``. Série vazia se ``df_review`` for vazio.
    """
    if target_col not in df_review.columns:
        raise KeyError(f"df_review precisa da coluna de rótulo '{target_col}'.")
    if band_col not in df_review.columns:
        raise KeyError(f"df_review precisa da coluna de banda '{band_col}'.")

    rates_by_band = error_rates_by_band or {}
    n = len(df_review)
    if n == 0:
        return pd.Series([], dtype="string", index=df_review.index)

    rng = np.random.default_rng(seed)
    u = rng.random(n)  # uma sequência, em ordem de linha → determinística

    targets = df_review[target_col].to_numpy()
    bands = df_review[band_col].to_numpy()

    decisions: list[str] = []
    for i in range(n):
        rates = rates_by_band.get(str(bands[i]), _FALLBACK_ERROR_RATES)
        e_fp = float(rates["e_fp"])
        e_fn = float(rates["e_fn"])
        if int(targets[i]) == 1:
            decisions.append("NONMATCH" if u[i] < e_fn else "MATCH")
        else:
            decisions.append("MATCH" if u[i] < e_fp else "NONMATCH")

    return pd.Series(decisions, index=df_review.index, dtype="string")


def reliability_points(y_true, p_pred, *, n_bins=10) -> pd.DataFrame:
    """Resume faixas não vazias; bordas internas pertencem à faixa à direita.

    O clipping é dos índices das faixas, não das probabilidades. Assim preserva
    inclusive o comportamento histórico de ECE para previsões fora de [0, 1].
    """
    if n_bins < 1:
        raise ValueError("n_bins deve ser >= 1.")
    y, p = _as_pair(y_true, p_pred)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(np.floor(p * n_bins + 1e-9), 0, n_bins - 1).astype(int)
    rows = []
    for b in range(n_bins):
        mask = bins == b
        count = int(mask.sum())
        if count:
            conf, acc = float(p[mask].mean()), float(y[mask].mean())
            rows.append((b, edges[b], edges[b + 1], count, conf, acc, abs(acc - conf)))
    return pd.DataFrame(rows, columns=["bin", "lo", "hi", "n", "conf", "acc", "gap"])


def log_loss(y_true, p_pred, *, eps=1e-15) -> float:
    """Multa média por surpresa, aceitando também rótulos suaves em [0, 1]."""
    y, p = _as_pair(y_true, p_pred)
    if not 0 < eps < 0.5:
        raise ValueError("eps deve estar entre 0 e 0,5.")
    if not np.isfinite(y).all() or ((y < 0) | (y > 1)).any():
        raise ValueError("Os alvos devem ser finitos e estar em [0, 1].")
    if not np.isfinite(p).all():
        raise ValueError("As probabilidades devem ser finitas.")
    p = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def _sigmoid(z):
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def _newton_logistic(x, t, w, l2, *, max_iter=100, tol=1e-10, init=None):
    """Newton da biblioteca, estendido a alvos suaves e pesos por linha.

    Cada linha registra os parâmetros e a perda APÓS o passo, e o gradiente,
    Hessiana e delta que produziram esse passo. A perda é a soma ponderada
    com L2 apenas na inclinação. ``init`` é um ``PlattModel``.
    """
    t, x = _as_pair(t, x)
    w = np.ones_like(x) if w is None else np.asarray(w, dtype=float).ravel()
    if not np.isfinite(x).all() or not np.isfinite(t).all():
        raise ValueError("Notas e alvos devem ser finitos.")
    if ((t < 0) | (t > 1)).any():
        raise ValueError("Os alvos devem estar em [0, 1].")
    if w.shape != x.shape or not np.isfinite(w).all() or (w < 0).any():
        raise ValueError("Os pesos devem ser finitos, não negativos e alinhados.")
    if w.sum() <= 0 or l2 < 0 or not np.isfinite(l2):
        raise ValueError("A soma dos pesos deve ser positiva e l2 não negativo.")
    if max_iter < 1 or tol <= 0 or not np.isfinite(tol):
        raise ValueError("max_iter e tol devem ser positivos.")
    p0 = (np.sum(w * t) + 0.5) / (np.sum(w) + 1.0)
    b = float(np.log(p0 / (1.0 - p0))) if init is None else init.intercept
    a = 0.0 if init is None else init.slope
    if not np.isfinite([a, b]).all():
        raise ValueError("O modelo inicial deve ter parâmetros finitos.")
    rows = []
    converged = False
    for step in range(1, max_iter + 1):
        p = _sigmoid(b + a * x)
        curvature = w * (p * (1.0 - p))
        residual = w * (p - t)
        g0 = float(np.sum(residual))
        g1 = float(np.sum(residual * x) + l2 * a)
        h00 = float(np.sum(curvature))
        h01 = float(np.sum(curvature * x))
        h11 = float(np.sum(curvature * x * x) + l2)
        h = np.array([[h00, h01], [h01, h11]], dtype=np.float64)
        g = np.array([g0, g1], dtype=np.float64)
        try:
            delta = np.linalg.solve(h, g)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError("Falha no ajuste Platt: Hessiana singular.") from exc
        b -= float(delta[0])
        a -= float(delta[1])
        if not np.isfinite([a, b, *delta]).all():
            rows.append((step, b, a, g0, g1, h00, h01, h11, *delta, float("nan")))
            break
        z = b + a * x
        loss = float(np.sum(w * (np.logaddexp(0.0, z) - t * z)) + 0.5 * l2 * a * a)
        rows.append((step, b, a, g0, g1, h00, h01, h11, *delta, loss))
        if not np.isfinite([a, b, loss, *delta]).all():
            break
        if float(np.max(np.abs(delta))) < tol:
            converged = True
            break
    trace = pd.DataFrame(rows, columns=[
        "passo", "intercepto", "inclinacao", "g_b", "g_a", "h_bb", "h_ba",
        "h_aa", "delta_b", "delta_a", "perda",
    ])
    trace.attrs["convergiu"] = converged
    trace.attrs["status"] = "ajustado" if converged else "não convergiu"
    model = PlattModel(intercept=b, slope=a) if np.isfinite([a, b]).all() else None
    return model, trace


def platt_newton_trace(
    nota, y, *, l2=1e-3, max_iter=100, tol=1e-10, weights=None, init=None,
) -> pd.DataFrame:
    """Expõe Newton; a = inclinação, b = intercepto, attrs['convergiu'] = sucesso."""
    _, trace = _newton_logistic(
        nota, y, weights, l2, max_iter=max_iter, tol=tol, init=init,
    )
    return trace


@dataclass(frozen=True)
class AnchorSpec:
    """Regras, alvos suaves e limites por janela da demonstração de âncoras."""

    pos_op: str
    pos_cut: float
    pos_target: float
    pos_min_count: int
    neg_op: str
    neg_cut: float
    neg_target: float
    neg_min_count: int
    max_abs_delta_a: float
    max_abs_delta_b: float

    def __post_init__(self):
        if self.pos_op not in _ANCHOR_OPS or self.neg_op not in _ANCHOR_OPS:
            raise ValueError("Operador de âncora inválido.")
        if not 0 < self.neg_target < self.pos_target < 1:
            raise ValueError("Os alvos devem satisfazer 0 < negativo < positivo < 1.")
        values = [self.pos_cut, self.neg_cut,
                  self.max_abs_delta_a, self.max_abs_delta_b]
        if not np.isfinite(values).all():
            raise ValueError("Cortes e limites das âncoras devem ser finitos.")
        if min(self.pos_min_count, self.neg_min_count,
               self.max_abs_delta_a, self.max_abs_delta_b) < 0:
            raise ValueError("Contagens mínimas e limites não podem ser negativos.")


_ANCHOR_OPS = {">=": np.greater_equal, ">": np.greater,
               "<=": np.less_equal, "<": np.less}


def load_anchor_spec(config_path) -> AnchorSpec:
    """Lê as âncoras e salvaguardas da YAML crua, não do parser da biblioteca."""
    raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    pos, neg = raw["anchors"]["pos_anchor"], raw["anchors"]["neg_anchor"]

    def parse(rule):
        match = re.match(r"^\s*nota_final\s*(>=|>|<=|<)\s*([0-9.]+)\s*$", rule)
        if match is None:
            raise ValueError(f"Regra de âncora não suportada: {rule!r}.")
        return match[1], float(match[2])

    pos_op, pos_cut = parse(pos["rule"])
    neg_op, neg_cut = parse(neg["rule"])
    safeguards = raw["calibration"]["safeguards"]
    return AnchorSpec(
        pos_op, pos_cut, float(pos["target_mean_probability"]), int(pos["min_count"]),
        neg_op, neg_cut, float(neg["target_mean_probability"]), int(neg["min_count"]),
        float(safeguards["max_abs_delta_a_per_window"]),
        float(safeguards["max_abs_delta_b_per_window"]),
    )


def anchor_masks(nota, spec) -> tuple[np.ndarray, np.ndarray]:
    """Seleciona os dois grupos de referência, que não podem se sobrepor."""
    x = np.asarray(nota, dtype=float).ravel()
    if not np.isfinite(x).all():
        raise ValueError("As notas das âncoras devem ser finitas.")
    pos = _ANCHOR_OPS[spec.pos_op](x, spec.pos_cut)
    neg = _ANCHOR_OPS[spec.neg_op](x, spec.neg_cut)
    if (pos & neg).any():
        raise ValueError("As regiões das âncoras não podem se sobrepor.")
    return pos, neg


def anchor_two_point_guess(nota, spec) -> PlattModel:
    """Resolve duas equações em log-odds nas notas médias de cada grupo."""
    x = np.asarray(nota, dtype=float).ravel()
    pos, neg = anchor_masks(x, spec)
    if not pos.any() or not neg.any():
        raise ValueError("O chute exige pelo menos uma âncora de cada grupo.")
    mean_pos, mean_neg = float(x[pos].mean()), float(x[neg].mean())
    if mean_pos == mean_neg:
        raise ValueError("As médias das notas das âncoras devem ser diferentes.")
    lp = np.log(spec.pos_target / (1.0 - spec.pos_target))
    ln = np.log(spec.neg_target / (1.0 - spec.neg_target))
    a = float((lp - ln) / (mean_pos - mean_neg))
    return PlattModel(intercept=float(lp - a * mean_pos), slope=a)


@dataclass(frozen=True)
class AnchorFit:
    """Resultado didático, incluindo diagnóstico mesmo quando o ajuste congela."""

    model: PlattModel | None
    status: str
    n_pos: int
    n_neg: int
    mean_s_pos: float
    mean_s_neg: float
    fitted_mean_pos: float
    fitted_mean_neg: float
    n_iter: int
    converged: bool = False


def fit_anchor_platt(
    nota, spec, *, l2=1e-3, max_iter=100, tol=1e-10, balance=True,
    enforce_min_count=True, fallback=None,
) -> AnchorFit:
    """Ajusta só as âncoras, sem rótulos observados e com pesos equilibrados.

    Congela quando falta um grupo ou seu mínimo exigido. Nesse caso devolve
    exatamente ``fallback`` (possivelmente None), sem executar Newton.
    Se Newton esgota max_iter ou produz valores não finitos, o status é
    ``não convergiu`` e o modelo também é fallback, nunca parâmetros NaN.
    """
    x = np.asarray(nota, dtype=float).ravel()
    pos, neg = anchor_masks(x, spec)
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    frozen = not n_pos or not n_neg or (enforce_min_count and (
        n_pos < spec.pos_min_count or n_neg < spec.neg_min_count
    ))
    model, n_iter, converged = fallback, 0, False
    if not frozen:
        selected = pos | neg
        t = np.where(pos[selected], spec.pos_target, spec.neg_target)
        n = n_pos + n_neg
        w = np.where(pos[selected], n / (2 * n_pos), n / (2 * n_neg))
        if not balance:
            w = np.ones(n)
        try:
            candidate, trace = _newton_logistic(
                x[selected], t, w, l2, max_iter=max_iter, tol=tol,
                init=anchor_two_point_guess(x, spec),
            )
            n_iter = len(trace)
            converged = trace.attrs["convergiu"]
            model = candidate if converged else fallback
        except RuntimeError:
            # Uma Hessiana singular não produz parâmetros utilizáveis.
            model = fallback

    def mean(mask, fitted=False):
        if not mask.any() or (fitted and model is None):
            return float("nan")
        values = x[mask]
        if fitted:
            values = _sigmoid(model.intercept + model.slope * values)
        return float(values.mean())

    status = "congelado" if frozen else ("ajustado" if converged else "não convergiu")
    return AnchorFit(
        model, status,
        n_pos, n_neg, mean(pos), mean(neg), mean(pos, True), mean(neg, True),
        n_iter, converged,
    )


def clamp_update(prev, new, *, max_abs_delta_a, max_abs_delta_b):
    """Limita separadamente a mudança de inclinação e intercepto entre modelos."""
    if min(max_abs_delta_a, max_abs_delta_b) < 0:
        raise ValueError("Os limites de atualização não podem ser negativos.")
    da, db = new.slope - prev.slope, new.intercept - prev.intercept
    a = prev.slope + float(np.clip(da, -max_abs_delta_a, max_abs_delta_a))
    b = prev.intercept + float(np.clip(db, -max_abs_delta_b, max_abs_delta_b))
    return (PlattModel(intercept=b, slope=a),
            bool(abs(da) > max_abs_delta_a), bool(abs(db) > max_abs_delta_b))


def simulate_anchor_windows(windows, spec, *, init, l2=1e-3) -> pd.DataFrame:
    """Simula janelas; sem modelo inicial, janela congelada tem parâmetros NaN."""
    prev = init
    rows = []
    for window, nota in enumerate(windows, 1):
        fit = fit_anchor_platt(nota, spec, l2=l2, fallback=prev)
        raw = fit.model
        if prev is None or raw is None:
            used, clamped_a, clamped_b = raw, False, False
        else:
            used, clamped_a, clamped_b = clamp_update(
                prev, raw, max_abs_delta_a=spec.max_abs_delta_a,
                max_abs_delta_b=spec.max_abs_delta_b,
            )
        rows.append((window, fit.n_pos, fit.n_neg, fit.status,
                     raw.slope if raw else float("nan"),
                     raw.intercept if raw else float("nan"),
                     used.slope if used else float("nan"),
                     used.intercept if used else float("nan"),
                     (-used.intercept / used.slope
                      if used and used.slope else float("nan")),
                     clamped_a, clamped_b))
        prev = used
    return pd.DataFrame(rows, columns=[
        "janela", "n_pos", "n_neg", "status", "a_bruto", "b_bruto", "a_usado",
        "b_usado", "s_estrela", "travou_a", "travou_b",
    ])


def expected_losses(p, *, c_fp, c_fn, c_llm, e_fp, e_fn) -> pd.DataFrame:
    """Calcula as três perdas e o EVR, com o desempate automático do engine."""
    index = p.index if isinstance(p, pd.Series) else None
    p = np.atleast_1d(np.asarray(p, dtype=float))
    loss_match, loss_non = (1.0 - p) * c_fp, p * c_fn
    loss_llm = c_llm + (1.0 - p) * np.asarray(e_fp) * c_fp + p * np.asarray(e_fn) * c_fn
    base_loss = np.minimum(loss_match, loss_non)
    return pd.DataFrame({
        "loss_match": loss_match, "loss_non": loss_non, "loss_llm": loss_llm,
        "base_choice": np.where(loss_match <= loss_non, "MATCH", "NONMATCH"),
        "base_loss": base_loss, "evr": base_loss - loss_llm,
    }, index=index)


def decision_thresholds(*, c_fp, c_fn, c_llm, e_fp, e_fn) -> dict[str, float]:
    """Ponto de virada e intervalo onde revisar vence ambas as ações automáticas.

    A janela é intersectada com [0, 1]; seus dois limites são NaN se vazia.
    Custos de erro devem ser positivos e taxas de erro estar em [0, 1].
    """
    if not np.isfinite([c_fp, c_fn, c_llm, e_fp, e_fn]).all():
        raise ValueError("Custos e taxas devem ser finitos.")
    if c_fp <= 0 or c_fn <= 0 or c_llm < 0:
        raise ValueError("Custos de erro devem ser positivos; revisão não negativa.")
    if not (0 <= e_fp <= 1 and 0 <= e_fn <= 1):
        raise ValueError("As taxas de erro devem estar em [0, 1].")
    denom_lo = c_fn * (1.0 - e_fn) + e_fp * c_fp
    denom_hi = c_fp * (1.0 - e_fp) + e_fn * c_fn
    lo = (c_llm + e_fp * c_fp) / denom_lo if denom_lo else float("inf")
    hi = (c_fp * (1.0 - e_fp) - c_llm) / denom_hi if denom_hi else -float("inf")
    lo, hi = max(0.0, lo), min(1.0, hi)
    if lo >= hi:
        lo = hi = float("nan")
    return {"virada": c_fp / (c_fp + c_fn), "llm_inf": lo, "llm_sup": hi}


def thresholds_table(cfg) -> pd.DataFrame:
    """Tabela dos limiares por modo e banda, diretamente da configuração."""
    rows = []
    for mode, policy in cfg.decision_policy.modes.items():
        for band in cfg.bands.definitions:
            rates = cfg.llm_review.error_rates_by_band[band.name]
            thresholds = decision_thresholds(
                c_fp=policy.false_positive_cost, c_fn=policy.false_negative_cost,
                c_llm=policy.llm_review_cost, e_fp=rates["e_fp"], e_fn=rates["e_fn"],
            )
            rows.append({"modo": mode, "banda": band.name, **thresholds})
    return pd.DataFrame(rows)


def build_engine(
    cfg: GZCMDConfig, mode, *, c_fp=None, c_fn=None, c_llm=None, llm_max=None,
) -> PolicyEngineV3:
    """Constrói o engine real com orçamento novo, permitindo explorar custos.

    Reutiliza o construtor do runner, inclusive suas validações e o comportamento
    quando a revisão está desativada. Nenhum estado é compartilhado entre chamadas.
    """
    engine = build_engine_from_config(cfg, mode=mode)
    costs = engine.costs
    engine.costs = Costs(
        false_positive=costs.false_positive if c_fp is None else float(c_fp),
        false_negative=costs.false_negative if c_fn is None else float(c_fn),
        llm_review=costs.llm_review if c_llm is None else float(c_llm),
    )
    if llm_max is not None and cfg.llm_review.enabled:
        if llm_max < 0:
            raise ValueError("O orçamento não pode ser negativo.")
        engine.budget.llm_max = int(llm_max)
    return engine


def review_status(out, *, min_auto_match, max_auto_nonmatch) -> pd.DataFrame:
    """Rotula a saída real, sem selecionar novamente nem alterar decisões.

    Prioridades: ``force``, ``trava``, ``evr`` e ``nenhuma``, nessa ordem.
    Status: ``revisado``, ``de fora``, ``não pediu`` e ``guardrail``.
    ALWAYS_* é guardrail mesmo quando ``review_requested`` é verdadeiro.
    """
    guard = out.get("guardrail", pd.Series(pd.NA, index=out.index, dtype="string"))
    deterministic = guard.isin(["ALWAYS_MATCH", "ALWAYS_NONMATCH"])
    force = guard.eq("FORCE_REVIEW").fillna(False)
    cap = pd.Series(False, index=out.index)
    if min_auto_match is not None:
        cap |= out["base_choice"].eq("MATCH") & out["p_cal"].lt(min_auto_match)
    if max_auto_nonmatch is not None:
        cap |= out["base_choice"].eq("NONMATCH") & out["p_cal"].gt(max_auto_nonmatch)
    priority = pd.Series("nenhuma", index=out.index, dtype="string")
    priority = priority.mask(out["evr"].gt(0), "evr").mask(cap, "trava")
    priority = priority.mask(force, "force").mask(deterministic, "nenhuma")
    status = pd.Series("não pediu", index=out.index, dtype="string")
    status = status.mask(out["review_requested"], "de fora")
    status = status.mask(out["action"].eq("LLM_REVIEW"), "revisado")
    status = status.mask(deterministic, "guardrail")
    return pd.DataFrame({"prioridade": priority, "status": status}, index=out.index)


NOME_PESOS_LOADER = (0.5, 0.25, 0.25)
MACD_JANELAS_LOADER = {"very_close_days": 7, "close_days": 30, "far_years": 5.0}


def _feature_num(df, col):
    raw = df[col] if col in df else pd.Series(0.0, index=df.index)
    return pd.to_numeric(raw, errors="coerce").fillna(0.0).astype("float64")


def nome_score(df, *, w_qtd, w_prim, w_ult) -> pd.Series:
    """Refaz a receita do nome; pesos livres e clipping final em [0, 1]."""
    return (
        w_qtd * _feature_num(df, "NOME qtd frag iguais")
        + w_prim * _feature_num(df, "NOME prim frag igual")
        + w_ult * _feature_num(df, "NOME ult frag igual")
    ).clip(0.0, 1.0).rename("nome_score_total")


def macd_flags(df, *, very_close_days, close_days, far_years) -> pd.DataFrame:
    """Refaz as features MACD a partir das datas já interpretadas pelo loader."""
    r, c = df["R_DTNASC_dt"], df["C_DTNASC_dt"]
    days_abs = (c - r).dt.days.astype("float64").abs()
    years_abs = (days_abs / 365.25).astype("float64")
    capped = years_abs.clip(upper=20.0)
    out = pd.DataFrame(index=df.index)
    out["macd_nasc_diff_capped"] = capped
    for part in ("year", "month", "day"):
        out[f"macd_nasc_{part}_match"] = (
            getattr(r.dt, part) == getattr(c.dt, part)
        ).fillna(False).astype("int8")
    out["macd_nasc_partial_overlap"] = sum(
        out[f"macd_nasc_{part}_match"].astype("float64")
        for part in ("year", "month", "day")
    ) / 3.0
    out["macd_nasc_close"] = (days_abs <= close_days).fillna(False).astype("int8")
    out["macd_nasc_very_close"] = (
        days_abs <= very_close_days
    ).fillna(False).astype("int8")
    perfect = (
        _feature_num(df, "NOME qtd frag iguais").ge(0.95)
        & _feature_num(df, "NOME prim frag igual").ge(1.0)
        & _feature_num(df, "NOME ult frag igual").ge(1.0)
    )
    far = years_abs.notna() & years_abs.ge(far_years)
    out["macd_nome_perf_x_date_far"] = (perfect & far).astype("int8")
    out["macd_nome_perf_x_year_diff"] = perfect.astype("float64") * capped.fillna(0.0)
    return out


def auc_separacao(x, y) -> float:
    """Probabilidade de ordenar um positivo acima de um negativo; empate vale metade."""
    return float(roc_auc_score(y, x))


def assign_bands_with_edges(nota, edges, names, *, cfg=None) -> pd.Series:
    """Atribui bandas com fronteiras internas estritamente crescentes.

    Usa os limites externos de cfg (ou da YAML do projeto); fora deles fica NA.
    """
    edges = np.asarray(edges, dtype=float)
    if edges.ndim != 1 or not np.isfinite(edges).all() or (np.diff(edges) <= 0).any():
        raise ValueError("As fronteiras devem ser finitas e estritamente crescentes.")
    if len(names) != len(edges) + 1 or len(set(names)) != len(names):
        raise ValueError("É necessário um nome único para cada banda.")
    if cfg is None:
        from gzcmd_record_linkage.config import load_config

        cfg = load_config(Path(__file__).resolve().parents[1]
                          / "src/gzcmd_record_linkage/gzcmd_v3_config.yaml")
    first, last = cfg.bands.definitions[0], cfg.bands.definitions[-1]
    limits = [first.min, *edges, last.max]
    definitions = tuple(
        BandDefinition(name, limits[i], limits[i + 1],
                       last.inclusive_max if i == len(names) - 1 else False)
        for i, name in enumerate(names)
    )
    series = nota if isinstance(nota, pd.Series) else pd.Series(nota)
    return BandAssigner(definitions).assign(series)


def band_transitions(old, new, order) -> pd.DataFrame:
    """Conta transições banda antiga (linhas) para nova (colunas), incluindo zeros."""
    old, new = np.asarray(old), np.asarray(new)
    if old.shape != new.shape:
        raise ValueError("As duas atribuições de bandas devem ter o mesmo tamanho.")
    table = pd.crosstab(pd.Series(old, name="anterior"), pd.Series(new, name="nova"))
    return table.reindex(index=order, columns=order, fill_value=0)


def guardrail_diff(df, base_params, new_params) -> pd.DataFrame:
    """Compara regra e motivo por linha, tratando dois valores ausentes como iguais.

    Parâmetros podem ser dicionários ou ``GuardrailsParams`` da configuração.
    """
    base = apply_guardrails(df, **(
        dict(base_params) if isinstance(base_params, Mapping) else asdict(base_params)
    ))
    new = apply_guardrails(df, **(
        dict(new_params) if isinstance(new_params, Mapping) else asdict(new_params)
    ))
    return pd.DataFrame({
        "guardrail_base": base.guardrail, "guardrail_novo": new.guardrail,
        "motivo_base": base.reason, "motivo_novo": new.reason,
        "mudou_regra": base.guardrail.fillna("").ne(new.guardrail.fillna("")),
        "mudou_motivo": base.reason.fillna("").ne(new.reason.fillna("")),
    }, index=df.index)


def _model_probability(model, nota):
    return _sigmoid(model.slope * np.asarray(nota, dtype=float) + model.intercept)


def _model_metrics(model, nota, y, *, p_true=None, n_bins=10):
    p = _model_probability(model, nota)
    return {
        "a": model.slope, "b": model.intercept,
        "s*": -model.intercept / model.slope if model.slope else float("nan"),
        "ECE": expected_calibration_error(y, p, n_bins=n_bins),
        "Brier": brier_score(y, p), "log-loss": log_loss(y, p),
        "MAE vs p*": (float(np.mean(np.abs(p - np.asarray(p_true))))
                      if p_true is not None else float("nan")),
    }


def plot_reliability(y_true, p_pred, *, n_bins=10, ax=None, titulo="Calibração"):
    """Diagrama de confiabilidade; bolinhas maiores representam mais pares.

    ``ax`` permite compor a figura sem criar ou exibir outra janela. O resumo
    contém as mesmas faixas usadas pelo cálculo de ECE.
    """
    import matplotlib.pyplot as plt

    points = reliability_points(y_true, p_pred, n_bins=n_bins)
    ece = expected_calibration_error(y_true, p_pred, n_bins=n_bins)
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4), layout="constrained")
    ax.plot([0, 1], [0, 1], linestyle="--", color="0.5", label="Calibração perfeita")
    ax.scatter(points["conf"], points["acc"],
               s=20 + 180 * points["n"] / points["n"].max(), label="Faixas observadas")
    ax.set(
        title=(f"{titulo}\nECE={fmt_br(ece)}; "
               f"Brier={fmt_br(brier_score(y_true, p_pred))}; "
               f"log-loss={fmt_br(log_loss(y_true, p_pred))}"),
        xlabel="Probabilidade média prevista", ylabel="Fração observada de matches",
        xlim=(0, 1), ylim=(0, 1),
    )
    ax.legend(fontsize="small")
    return ax.figure, points


def plot_platt_explorer(
    nota, y, model, *, ref_model=None, p_true=None, n_bins=10,
    inclinacao=None, intercepto=None,
):
    """Compara a curva escolhida, a referência aprendida e o boletim no teste.

    Sem controles explícitos, usa a inclinação e o intercepto de ``model``.
    ``nota``, ``y`` e ``p_true`` devem ser alinhados ao mesmo conjunto de teste.
    """
    import matplotlib.pyplot as plt

    x, y = _as_pair(nota, y)
    reference = model if ref_model is None else ref_model
    chosen = PlattModel(
        slope=model.slope if inclinacao is None else inclinacao,
        intercept=model.intercept if intercepto is None else intercepto,
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    grid = np.linspace(x.min(), x.max(), 250)
    axes[0].plot(grid, _model_probability(chosen, grid), label="Curva escolhida")
    axes[0].plot(grid, _model_probability(reference, grid), "--",
                 label="Ajuste aprendido")
    if p_true is not None:
        order = np.argsort(x)
        axes[0].plot(x[order], np.asarray(p_true)[order], ":", label="Verdade p*")
    # Transformação afim apenas para reutilizar as mesmas gavetas do helper.
    scaled = (x - x.min()) / max(float(np.ptp(x)), np.finfo(float).eps)
    empirical = reliability_points(y, scaled, n_bins=n_bins)
    centers = x.min() + empirical["conf"] * np.ptp(x)
    axes[0].scatter(centers, empirical["acc"],
                    s=20 + 150 * empirical["n"] / empirical["n"].max(),
                    label="Frequência observada")
    metrics = _model_metrics(chosen, x, y, p_true=p_true, n_bins=n_bins)
    if chosen.slope:
        axes[0].axvline(metrics["s*"], color="0.5", linestyle="-.",
                        label="Nota de 50% (s*)")
    axes[0].set(title="Platt: a = inclinação; b = intercepto",
                xlabel="Nota final", ylabel="Probabilidade de match", ylim=(0, 1))
    axes[0].legend(fontsize="small")
    plot_reliability(y, _model_probability(chosen, x), n_bins=n_bins, ax=axes[1])
    return fig, pd.DataFrame([metrics], index=["Curva escolhida"])


def _policy_plot_frame(df, cfg):
    """Completa uma cópia dos dados; preserva a calibração recebida pelo notebook."""
    from gzcmd_record_linkage.calibration import fit_platt

    frame = df.copy()
    nota = frame["nota_final"] if "nota_final" in frame else frame["nota final"]
    if "p_cal" not in frame:
        # Conveniência estática para dados recém-carregados, não avaliação held-out.
        frame["p_cal"] = _model_probability(fit_platt(nota, frame["TARGET"]), nota)
    if "band" not in frame:
        frame["band"] = BandAssigner.from_config(cfg).assign(nota)
    if "guardrail" not in frame:
        guard = apply_guardrails(frame, **asdict(cfg.guardrails))
        frame["guardrail"], frame["guardrail_reason"] = guard.guardrail, guard.reason
    return frame


def _action_counts(base, new):
    actions = ["MATCH", "NONMATCH", "LLM_REVIEW"]
    return pd.DataFrame({
        "Configuração": base["action"].value_counts().reindex(actions, fill_value=0),
        "Ajuste": new["action"].value_counts().reindex(actions, fill_value=0),
    })


def _grouped_bars(ax, table):
    """Barras sem pandas.plot, mantendo apresentação e eixos explícitos."""
    positions = np.arange(len(table))
    width = 0.8 / len(table.columns)
    for i, col in enumerate(table.columns):
        offset = (i - (len(table.columns) - 1) / 2) * width
        ax.bar(positions + offset, table[col], width=width, label=str(col))
    ax.set_xticks(positions, [str(label) for label in table.index], rotation=20)
    ax.legend(fontsize="small")


def plot_costs_explorer(
    df, cfg, *, mode="vigilancia", c_fp=None, c_fn=None, c_llm=None, band="grey_mid",
):
    """Retas de perda e decisões reais, com orçamento novo em cada comparação."""
    import matplotlib.pyplot as plt

    frame = _policy_plot_frame(df, cfg)
    baseline = build_engine(cfg, mode)
    engine = build_engine(cfg, mode, c_fp=c_fp, c_fn=c_fn, c_llm=c_llm)
    rates = cfg.llm_review.error_rates_by_band[band]
    costs = dict(c_fp=engine.costs.false_positive, c_fn=engine.costs.false_negative,
                 c_llm=engine.costs.llm_review, e_fp=rates["e_fp"], e_fn=rates["e_fn"])
    p = np.linspace(0, 1, 251)
    losses = expected_losses(p, **costs)
    limits = decision_thresholds(**costs)
    counts = _action_counts(baseline.triage(frame), engine.triage(frame))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for col, label in [("loss_match", "Dizer match"), ("loss_non", "Dizer não-match"),
                       ("loss_llm", "Pedir revisão")]:
        axes[0].plot(p, losses[col], label=label)
    if np.isfinite(limits["llm_inf"]):
        axes[0].axvspan(limits["llm_inf"], limits["llm_sup"], alpha=0.15,
                       color="green", label="Revisão mais barata")
    axes[0].axvline(limits["virada"], linestyle="--", color="black",
                    label="Ponto de virada")
    for value, label in [(engine.min_auto_match, "Trava de match"),
                         (engine.max_auto_nonmatch, "Trava de não-match")]:
        if value is not None:
            axes[0].axvline(value, linestyle=":", label=label)
    axes[0].set(title=f"Perdas esperadas: {mode}, banda {band}",
                xlabel="Probabilidade calibrada", ylabel="Custo esperado")
    axes[0].legend(fontsize="small")
    _grouped_bars(axes[1], counts.rename(index={
        "MATCH": "Match", "NONMATCH": "Não-match", "LLM_REVIEW": "Revisão",
    }))
    axes[1].set(title="Ações do motor real", xlabel="Ação",
                ylabel="Quantidade de pares")
    counts.attrs["limiares"] = limits
    counts.attrs["custos"] = costs
    return fig, counts


def plot_anchor_comparison(
    nota_all, nota_test, y_test, sup_model, fit=None, spec=None, *,
    p_true_all=None, p_true_test=None,
):
    """Demonstra âncoras sem rótulos e avalia as duas curvas no mesmo teste.

    Passe ``fit`` calculado com os controles para explorar balanceamento e mínimos.
    Se omitido, o ajuste usa os defaults de ``fit_anchor_platt`` e o supervisionado
    como fallback. Cortes cruzados produzem uma mensagem, nunca uma exceção.
    """
    import matplotlib.pyplot as plt

    if spec is None:
        return None, {"mensagem": "Informe as âncoras da configuração."}
    if spec.neg_cut >= spec.pos_cut:
        return None, {"mensagem": "O corte negativo deve ser menor que o positivo."}
    x = np.asarray(nota_all, dtype=float)
    try:
        pos, neg = anchor_masks(x, spec)
    except ValueError as exc:
        return None, {"mensagem": str(exc)}
    if fit is None:
        fit = fit_anchor_platt(x, spec, fallback=sup_model)
    frozen = fit.status in {"congelado", "não convergiu"}
    plotted_model = sup_model if frozen else fit.model
    if plotted_model is None:
        return None, {
            "mensagem": (f"Ajuste {fit.status}: "
                         "sem curva ajustada e sem modelo anterior."),
        }
    fig = plt.figure(figsize=(12, 7), layout="constrained")
    grid_spec = fig.add_gridspec(2, 2)
    curve = fig.add_subplot(grid_spec[:, 0])
    grid = np.linspace(x.min(), x.max(), 250)
    curve.plot(grid, _model_probability(sup_model, grid), "--", label="Supervisionado")
    curve.plot(grid, _model_probability(plotted_model, grid),
               label="Platt com gabarito (fallback)" if frozen else "Ancorado")
    if p_true_all is not None:
        order = np.argsort(x)
        curve.plot(x[order], np.asarray(p_true_all)[order], ":", label="Verdade p*")
    # As máscaras também respeitam operadores de âncoras diferentes dos defaults.
    grid_pos, grid_neg = anchor_masks(grid, spec)
    for mask, grid_mask, target, label, color in [
        (pos, grid_pos, spec.pos_target, "Âncora positiva", "green"),
        (neg, grid_neg, spec.neg_target, "Âncora negativa", "red"),
    ]:
        curve.fill_between(grid, 0, 1, where=grid_mask, alpha=0.1,
                           color=color, label=label)
        if mask.any():
            center = float(x[mask].mean())
            curve.scatter([center], [target], marker="s", color=color)
            if p_true_all is not None:
                truth = float(np.asarray(p_true_all)[mask].mean())
                curve.annotate(f"{label}: p* médio = {fmt_br(truth)}",
                               (center, truth), xytext=(0, 12),
                               textcoords="offset points",
                               ha="center", fontsize="small")
    message = ({
        "congelado": ("Congelado: poucas âncoras; "
                      "a curva mostrada é o Platt com gabarito."),
        "não convergiu": "Não convergiu: a curva mostrada é o Platt com gabarito.",
    }.get(fit.status, ""))
    title = f"Âncoras: ajuste {fit.status}" + (f"\n{message}" if message else "")
    curve.set(title=title, xlabel="Nota final",
              ylabel="Probabilidade de match", ylim=(-0.03, 1.1))
    curve.legend(fontsize="small")
    rows = []
    anchor_label = "Ancorado (fallback: Platt com gabarito)" if frozen else "Ancorado"
    models = [("Supervisionado", sup_model), (anchor_label, plotted_model)]
    for i, (label, model) in enumerate(models):
        ax = fig.add_subplot(grid_spec[i, 1])
        plot_reliability(y_test, _model_probability(model, nota_test),
                         ax=ax, titulo=label)
        rows.append({
            "modelo": label, "n_pos": fit.n_pos, "mínimo positivo": spec.pos_min_count,
            "n_neg": fit.n_neg, "mínimo negativo": spec.neg_min_count,
            "status": fit.status if i else "supervisionado",
            **_model_metrics(model, nota_test, y_test, p_true=p_true_test),
        })
    summary = pd.DataFrame(rows).set_index("modelo")
    summary["mensagem"] = message
    return fig, summary


def plot_bands_guardrails(df, cfg, *, edges=None, guard_params=None, mode="vigilancia"):
    """Mostra fronteiras, transições e efeito das regras após nova triagem."""
    import matplotlib.pyplot as plt

    names = [band.name for band in cfg.bands.definitions]
    old_edges = [band.max for band in cfg.bands.definitions[:-1]]
    edges = old_edges if edges is None else edges
    if not np.isfinite(edges).all() or (np.diff(edges) <= 0).any():
        formatted = "; ".join(fmt_br(edge) for edge in edges)
        message = f"As fronteiras precisam ser crescentes: {formatted}."
        fig, ax = plt.subplots(figsize=(10, 3), layout="constrained")
        ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
        ax.set(title="Fronteiras inválidas", xlabel="Nota final", ylabel="Diagnóstico")
        return fig, pd.DataFrame([{"mensagem": message}])
    params = asdict(cfg.guardrails)
    if guard_params is not None:
        params.update(dict(guard_params) if isinstance(guard_params, Mapping)
                      else asdict(guard_params))
    frame = _policy_plot_frame(df, cfg)
    nota = frame["nota_final"] if "nota_final" in frame else frame["nota final"]
    old = assign_bands_with_edges(nota, old_edges, names, cfg=cfg)
    new = assign_bands_with_edges(nota, edges, names, cfg=cfg)
    transitions = band_transitions(old, new, names)
    diff = guardrail_diff(frame, cfg.guardrails, params)
    base, changed = frame.copy(), frame.copy()
    base["band"], changed["band"] = old, new
    base["guardrail"] = diff["guardrail_base"]
    changed["guardrail"] = diff["guardrail_novo"]
    base["guardrail_reason"] = diff["motivo_base"]
    changed["guardrail_reason"] = diff["motivo_novo"]
    before = build_engine(cfg, mode).triage(base)
    after = build_engine(cfg, mode).triage(changed)
    summary = pd.DataFrame([{
        "pares": len(frame), "mudaram de banda": int(old.ne(new).sum()),
        "mudaram de regra": int(diff["mudou_regra"].sum()),
        "mudaram de ação": int(before["action"].ne(after["action"]).sum()),
        "modo": mode,
    }])
    counts = pd.DataFrame({
        "Configuração": diff["guardrail_base"].fillna("Sem regra").value_counts(),
        "Ajuste": diff["guardrail_novo"].fillna("Sem regra").value_counts(),
    }).fillna(0).astype(int)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), layout="constrained")
    axes[0].hist(nota, bins=25, alpha=0.5)
    for i, edge in enumerate(old_edges):
        axes[0].axvline(edge, color="0.3", linestyle="--",
                       label="Fronteiras anteriores" if i == 0 else None)
    for i, edge in enumerate(edges):
        axes[0].axvline(edge, color="tab:orange",
                        label="Fronteiras novas" if i == 0 else None)
    axes[0].set(title="Bandas da nota", xlabel="Nota final",
                ylabel="Quantidade de pares")
    axes[0].legend(fontsize="small")
    axes[1].imshow(transitions.to_numpy(), cmap="Blues")
    for (row, col), count in np.ndenumerate(transitions.to_numpy()):
        color = "white" if count > transitions.to_numpy().max() / 2 else "black"
        axes[1].text(col, row, str(count), ha="center", va="center", color=color)
    axes[1].set_xticks(range(len(names)), names, rotation=35)
    axes[1].set_yticks(range(len(names)), names)
    axes[1].set(title="Transições de banda: contagens", xlabel="Banda nova",
                ylabel="Banda anterior")
    _grouped_bars(axes[2], counts.rename(index={
        "ALWAYS_MATCH": "Sempre match", "ALWAYS_NONMATCH": "Sempre não-match",
        "FORCE_REVIEW": "Pedido prioritário (FORCE_REVIEW)",
    }))
    axes[2].set(title="Guardrails antes e depois", xlabel="Regra",
                ylabel="Quantidade de pares")
    summary.attrs["transições"] = transitions
    summary.attrs["guardrails"] = counts
    return fig, summary


def plot_feature_explorer(
    df, *, w_qtd=NOME_PESOS_LOADER[0], w_prim=NOME_PESOS_LOADER[1],
    w_ult=NOME_PESOS_LOADER[2], very_close_days=MACD_JANELAS_LOADER["very_close_days"],
    close_days=MACD_JANELAS_LOADER["close_days"],
    far_years=MACD_JANELAS_LOADER["far_years"],
):
    """Receita do nome, taxas MACD e lacunas de dias, incluindo diferenças zero."""
    import matplotlib.pyplot as plt

    score = nome_score(df, w_qtd=w_qtd, w_prim=w_prim, w_ult=w_ult)
    default = nome_score(df, **dict(zip(
        ["w_qtd", "w_prim", "w_ult"], NOME_PESOS_LOADER, strict=True,
    )))
    flags = macd_flags(df, very_close_days=very_close_days,
                       close_days=close_days, far_years=far_years)
    flag_labels = {"macd_nasc_very_close": "Data muito próxima",
                   "macd_nasc_close": "Data próxima",
                   "macd_nome_perf_x_date_far": "Nome perfeito e data distante"}
    rates = flags[list(flag_labels)].groupby(df["TARGET"]).mean()
    summary = rates.rename(columns=flag_labels)
    summary["AUC novo"] = auc_separacao(score, df["TARGET"])
    summary["AUC padrão"] = auc_separacao(default, df["TARGET"])
    days = (df["C_DTNASC_dt"] - df["R_DTNASC_dt"]).dt.days.abs()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), layout="constrained")
    bins = np.geomspace(1, max(float(days.max()) + 1, 2), 30)
    for target, label in [(0, "Não-match"), (1, "Match")]:
        mask = df["TARGET"].eq(target)
        axes[0].hist(score[mask], bins=np.linspace(0, 1, 21), alpha=0.5, label=label)
        axes[2].hist(days[mask].dropna() + 1, bins=bins, alpha=0.5, label=label)
    axes[0].set(title=(f"Nome: AUC novo={fmt_br(summary['AUC novo'].iloc[0])}; "
                       f"padrão={fmt_br(summary['AUC padrão'].iloc[0])}"),
                xlabel="Nota do nome", ylabel="Quantidade de pares")
    axes[0].legend(fontsize="small")
    _grouped_bars(axes[1], rates.rename(columns=flag_labels).T.rename(columns={
        0: "Não-match", 1: "Match",
    }))
    axes[1].set(title="Taxas dos indicadores MACD", xlabel="Indicador",
                ylabel="Fração de pares")
    for value, label in [(very_close_days, "Janela muito próxima"),
                         (close_days, "Janela próxima"),
                         (far_years * 365.25, "Corte de data distante")]:
        axes[2].axvline(value + 1, linestyle="--", label=label)
    axes[2].set(xscale="log", title="Lacunas nas datas de nascimento",
                xlabel="Diferença absoluta em dias + 1 (escala log)",
                ylabel="Quantidade de pares")
    axes[2].legend(fontsize="small")
    return fig, summary


def plot_budget_explorer(df, cfg, *, mode="vigilancia", llm_max=None):
    """Fila real por prioridade; cores mostram quem recebeu vaga de revisão."""
    import matplotlib.pyplot as plt

    engine = build_engine(cfg, mode, llm_max=llm_max)
    out = engine.triage(_policy_plot_frame(df, cfg))
    labels = review_status(out, min_auto_match=engine.min_auto_match,
                           max_auto_nonmatch=engine.max_auto_nonmatch)
    summary = out[["p_cal", "evr", "action", "review_requested"]].join(labels)
    counts = pd.crosstab(labels["prioridade"], labels["status"]).reindex(
        index=["force", "trava", "evr"], columns=["revisado", "de fora"], fill_value=0,
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), layout="constrained")
    colors = {"revisado": "tab:green", "de fora": "tab:red",
              "não pediu": "0.65", "guardrail": "tab:purple"}
    descriptions = {"de fora": "Pediu mas ficou de fora", "revisado": "Revisado",
                    "não pediu": "Não pediu", "guardrail": "Guardrail"}
    priorities = {"force": ("X", "Pedido prioritário (FORCE_REVIEW)"),
                  "trava": ("^", "Trava"),
                  "evr": ("o", "EVR"), "nenhuma": ("o", "Sem prioridade")}
    for status, color in colors.items():
        for priority, (marker, label) in priorities.items():
            mask = labels["status"].eq(status) & labels["prioridade"].eq(priority)
            if mask.any():
                axes[0].scatter(out.loc[mask, "p_cal"], out.loc[mask, "evr"],
                                color=color, marker=marker, s=25, alpha=0.7,
                                label=f"{descriptions[status]}: {label}")
    axes[0].axhline(0, color="black", linestyle="--", label="EVR zero")
    # A fronteira respeita primeiro a prioridade, depois o EVR, não só o EVR global.
    ranked = summary.assign(ordem=summary["prioridade"].map({
        "force": 0, "trava": 1, "evr": 2, "nenhuma": 3,
    })).sort_values(["ordem", "evr"], ascending=[True, False], kind="stable")
    selected = ranked.loc[ranked["status"].eq("revisado"), "evr"]
    excluded = ranked.loc[ranked["status"].eq("de fora"), "evr"]
    last = fmt_br(selected.iloc[-1]) if len(selected) else "nenhum"
    first = fmt_br(excluded.iloc[0]) if len(excluded) else "nenhum"
    axes[0].set(title=f"Último EVR revisado: {last}; primeiro de fora: {first}",
                xlabel="Probabilidade calibrada",
                ylabel="Economia esperada da revisão (EVR)")
    axes[0].legend(fontsize="x-small")
    _grouped_bars(axes[1], counts.rename(
        index={"force": "FORCE_REVIEW", "trava": "Trava", "evr": "EVR"},
        columns={"revisado": "Revisados", "de fora": "De fora"},
    ))
    config_max = build_engine(cfg, mode).budget.llm_max
    axes[1].set(title=(f"Vagas de revisão: {engine.budget.llm_max}; "
                       f"configuração: {config_max}"),
                xlabel="Prioridade da fila", ylabel="Quantidade de pares")
    summary.attrs["orçamento configuração"] = config_max
    summary.attrs["orçamento explorado"] = engine.budget.llm_max
    return fig, summary


def plot_anchor_windows(history):
    """Desenha a saída de ``simulate_anchor_windows``, sem refazer o ajuste."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
    for ax, parameter, title in [(axes[0], "a", "Inclinação (a)"),
                                 (axes[1], "b", "Intercepto (b)")]:
        ax.plot(history["janela"], history[f"{parameter}_bruto"], "--o",
                label="Ajuste bruto")
        ax.plot(history["janela"], history[f"{parameter}_usado"], "-o",
                label="Valor usado")
        ax.set(title=title, xlabel="Janela simulada", ylabel=title)
    axes[2].plot(history["janela"], history["s_estrela"], "-o", label="Nota de 50%")
    axes[2].set(title="Ponto de indiferença (s*)", xlabel="Janela simulada",
                ylabel="Nota de 50%")
    frozen = history.loc[history["status"].eq("congelado"), "janela"]
    for ax in axes:
        for i, window in enumerate(frozen):
            ax.axvspan(window - 0.4, window + 0.4, color="tab:orange", alpha=0.2,
                       label="Janela congelada" if i == 0 else None)
        ax.legend(fontsize="small")
    return fig, history.copy()


@dataclass(frozen=True)
class Deslizante:
    """Descrição de slider; limites logarítmicos são expoentes de base 10."""

    valor: float
    minimo: float
    maximo: float
    passo: float
    descricao: str
    inteiro: bool = False
    log: bool = False


@dataclass(frozen=True)
class Escolha:
    """Descrição de uma escolha entre opções."""

    opcoes: Sequence
    valor: object
    descricao: str


@dataclass(frozen=True)
class Caixa:
    """Descrição de uma caixa de seleção."""

    valor: bool
    descricao: str


def interagir(func, controles, *, fixos=None, presets=None, preset_de=None) -> bool:
    """Renderiza uma vez ao criar e a cada mudança, agrupando valores de presets.

    ``presets`` mapeia o valor de ``preset_de`` para valores dos outros controles.
    Falha inicial retorna False com diagnóstico; NBH_STRICT_WIDGETS=1 relança
    erros do callback. Ausência de ipywidgets sempre permite fallback estático.
    """
    callback_error = None
    try:
        import ipywidgets as widgets
        from IPython.display import display

        if "google.colab" in sys.modules:
            try:
                from google.colab import output

                output.enable_custom_widget_manager()
            except Exception as exc:
                print(f"[modo não interativo: gerenciador do Colab: {exc}]")
                return False

        controls = {}
        for name, spec in controles.items():
            if isinstance(spec, Deslizante):
                slider = (widgets.FloatLogSlider if spec.log else
                          widgets.IntSlider if spec.inteiro else widgets.FloatSlider)
                controls[name] = slider(
                    value=spec.valor, min=spec.minimo, max=spec.maximo,
                    step=spec.passo, description=spec.descricao,
                    continuous_update=False,
                )
            elif isinstance(spec, Escolha):
                controls[name] = widgets.Dropdown(
                    options=spec.opcoes, value=spec.valor, description=spec.descricao,
                )
            elif isinstance(spec, Caixa):
                controls[name] = widgets.Checkbox(
                    value=spec.valor, description=spec.descricao,
                )
            else:
                raise TypeError(f"Controle não suportado: {name}.")
        if presets is not None and preset_de not in controls:
            raise ValueError("preset_de deve nomear um controle existente.")
        result = widgets.Output()
        paused = False

        def render():
            nonlocal callback_error
            callback_error = None
            with result:
                result.clear_output(wait=True)
                try:
                    func(**(fixos or {}), **{
                        name: control.value for name, control in controls.items()
                    })
                except Exception as exc:
                    callback_error = exc
            if callback_error is not None:
                with result:
                    print(f"[falha no controle: {callback_error}]")
            # Relançar fora de Output, que pode capturar exceções no kernel.
            if (callback_error is not None
                    and os.environ.get("NBH_STRICT_WIDGETS") == "1"):
                raise callback_error

        def changed(change):
            nonlocal paused
            if paused:
                return
            if presets is not None and change["owner"] is controls[preset_de]:
                paused = True
                try:
                    for name, value in presets[change["new"]].items():
                        controls[name].value = value
                finally:
                    paused = False
            render()

        for control in controls.values():
            control.observe(changed, names="value")
        render()
        if callback_error is not None:
            print(f"[modo não interativo: falha inicial: {callback_error}]")
            return False
        display(widgets.VBox([*controls.values(), result]))
        return True
    except Exception as exc:
        if callback_error is not None and os.environ.get("NBH_STRICT_WIDGETS") == "1":
            raise
        print(f"[modo não interativo: {exc}]")
        return False


def fmt_br(x, nd=3):
    """Formata um número com vírgula decimal e nd casas, sem alterar seu valor."""
    return f"{x:.{nd}f}".replace(".", ",")


def ece_noise_floor(p_true, *, n_bins=10, n_sim=2000, seed=0) -> dict:
    """Estima o piso amostral de ECE de p* com novos rótulos Bernoulli.

    Devolve média e percentis 2,5/97,5 em ``mean``, ``ci_low`` e ``ci_high``.
    """
    _, p = _as_pair(p_true, p_true)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("p_true deve ser finito e estar em [0, 1].")
    if n_sim < 1:
        raise ValueError("n_sim deve ser positivo.")
    rng = np.random.default_rng(seed)
    values = [expected_calibration_error(rng.binomial(1, p), p, n_bins=n_bins)
              for _ in range(n_sim)]
    lo, hi = np.quantile(values, [0.025, 0.975])
    return {"mean": float(np.mean(values)), "ci_low": float(lo), "ci_high": float(hi)}


def paired_bootstrap_diff(y, p_a, p_b, *, metric, n_boot=2000, seed=0) -> dict:
    """Diferença A menos B e IC percentil de 95%, sorteando linhas em conjunto.

    Aceita auc, brier, logloss e ece. Para AUC, sorteios de classe única são
    descartados; alvos originais precisam conter as duas classes.
    """
    metrics = {"auc": roc_auc_score, "brier": brier_score,
               "logloss": log_loss, "ece": expected_calibration_error}
    if metric not in metrics or n_boot < 1:
        raise ValueError("Métrica inválida ou n_boot não positivo.")
    y, a = _as_pair(y, p_a)
    _, b = _as_pair(y, p_b)
    if metric == "auc" and len(np.unique(y)) < 2:
        raise ValueError("AUC exige as duas classes nos alvos originais.")
    score = metrics[metric]
    point = float(score(y, a) - score(y, b))
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        ix = rng.integers(0, len(y), size=len(y))
        if metric == "auc" and len(np.unique(y[ix])) < 2:
            continue
        values.append(float(score(y[ix], a[ix]) - score(y[ix], b[ix])))
    if not values:
        raise ValueError("Nenhum sorteio válido para estimar o intervalo.")
    lo, hi = np.quantile(values, [0.025, 0.975])
    return {"diff": point, "ci_low": float(lo), "ci_high": float(hi)}


def custo_esperado_por_par(res, cfg) -> pd.DataFrame:
    """Acrescenta custo_esperado_por_par aos resultados de evaluate_v3_dataframe.

    Usa suas colunas exp_fp, exp_fn, review_selected, n_test e mode; os preços
    vêm exclusivamente de cfg.decision_policy.modes para cada linha.
    """
    out = res.copy()
    values = []
    for row in out.itertuples(index=False):
        policy = cfg.decision_policy.modes[row.mode]
        values.append((row.exp_fp * policy.false_positive_cost
                       + row.exp_fn * policy.false_negative_cost
                       + row.review_selected * policy.llm_review_cost) / row.n_test)
    out["custo_esperado_por_par"] = values
    return out
