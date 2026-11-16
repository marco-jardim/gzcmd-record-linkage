"""Medição, com o LLM real, das taxas de erro do revisor usado pela fila.

Experimental. Mede o Claude Haiku (prompt v4, dossiê sem ``model_outputs`` e sem
``features.nota_final``) por faixa de nota sobre o extrato da tese, numa amostra
estratificada: todos os pares verdadeiros e até ``--per-band`` não pares por
faixa. **Não é a avaliação da fila**: é só a medição do revisor.

Chamadas reais, em paralelo, com cache em disco (``data/llm_cache/``, fora do
versionamento) e teto de revisões na API. A chave é lida de
``ANTHROPIC_API_KEY_ALT`` no ``.env`` e nunca é impressa nem gravada. O dossiê só
tem escores numéricos (mesmo protocolo da tese); nada em nível de par vai para
arquivos versionados: o relatório tem apenas agregados.

Uso (a partir da raiz do repositório)::

    python notebooks/llm_real_eval.py pilot --n 20 --workers 8
    python notebooks/llm_real_eval.py calibrate --workers 24 --max-api-reviews 1200
    python notebooks/llm_real_eval.py report
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import sys
import threading
import time
from collections import Counter
from collections.abc import Callable, Collection, Mapping, Sequence
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gzcmd_record_linkage.anthropic_transport import AnthropicReviewer
from gzcmd_record_linkage.arranjo_corte import THESIS_PARTITION, below_cutoff
from gzcmd_record_linkage.calibration import get_nota_series
from gzcmd_record_linkage.llm_dossier import DossierView, build_dossier, make_pair_id
from gzcmd_record_linkage.llm_review import AgentResponse, LLMCallError, ReviewResult
from gzcmd_record_linkage.loader import LoadConfig, load_comparador_csv
from gzcmd_record_linkage.self_calibrating_queue import (
    EXPLORE_ARM,
    OracleReviewer,
    SelfCalibratingConfig,
    SelfCalibratingSession,
)

__all__ = [
    "BAND_NAMES",
    "CachedParallelReviewer",
    "MissingApiKeyError",
    "ReviewAbort",
    "ReviewBudgetExceeded",
    "ReviewStopped",
    "cache_key",
    "load_api_key",
    "main",
    "select_sample",
    "wilson",
]

ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / ".env"
KEY_VAR = "ANTHROPIC_API_KEY_ALT"
DEFAULT_INPUT = ROOT.parent / "phd-research" / "data" / "COMPARADORSEMIDENT.csv"
DEFAULT_CACHE_DIR = ROOT / "data" / "llm_cache"
DEFAULT_MODEL = "claude-haiku-5-5"
PROMPT_VERSION = "v4"
REMOVE_FIELDS: tuple[str, ...] = ("features.nota_final",)
FORBIDDEN_KEYS: frozenset[str] = frozenset({"nota_final", "model_outputs"})
SEED = 20261115

# Faixas de nota do perfil v3.1: [a, b).
BANDS: tuple[tuple[str, float], ...] = (
    ("low", -math.inf),
    ("grey_low", 5.0),
    ("grey_mid", 6.0),
    ("grey_high", 7.0),
    ("near_high", 8.0),
    ("high", 9.0),
)
BAND_NAMES: tuple[str, ...] = tuple(b[0] for b in BANDS)
_EDGES = np.array([b[1] for b in BANDS[1:]], dtype=float)
SIDES: tuple[str, ...] = ("acima", "abaixo")

# Taxas sintéticas da simulação (docs/sim_fila_autocalibravel.md): (fp, fn, unsure),
# Laplace sobre a vigilância da tese, nota visível; ``low`` herdou ``grey_low``.
SYNTHETIC_RATES: dict[str, tuple[float, float, float]] = {
    "low": (0.084, 0.750, 0.082),
    "grey_low": (0.084, 0.750, 0.082),
    "grey_mid": (0.027, 0.067, 0.060),
    "grey_high": (0.029, 0.111, 0.047),
    "near_high": (0.045, 0.306, 0.080),
    "high": (0.500, 0.250, 0.167),
}
THESIS_VPP = (4, 25, 0.160, 0.064, 0.347)  # (VP, aceitos, VPP, IC inf, IC sup)


class MissingApiKeyError(RuntimeError):
    """A chave da API não está no ambiente."""


class ReviewStopped(RuntimeError):
    """Base das paradas limpas do lote."""


class ReviewBudgetExceeded(ReviewStopped):
    """O teto ``max_api_reviews`` de revisões na API foi atingido."""


class ReviewAbort(ReviewStopped):
    """Erros demais no lote (ex.: autenticação, modelo inexistente)."""


# ---------------------------------------------------------------------------
# Chave
# ---------------------------------------------------------------------------


def load_api_key(env_path: Path | str = ENV_PATH) -> str:
    """Lê ``ANTHROPIC_API_KEY_ALT`` do ``.env``/ambiente. Nunca imprime a chave."""
    from dotenv import load_dotenv  # type: ignore[import-untyped]

    load_dotenv(Path(env_path))
    key = os.environ.get(KEY_VAR, "").strip()
    if not key:
        raise MissingApiKeyError(
            f"{KEY_VAR} ausente: defina-a no .env (não é impressa nem gravada)."
        )
    return key


# ---------------------------------------------------------------------------
# Revisor instrumentado (uso por chamada, sem alterar a biblioteca)
# ---------------------------------------------------------------------------


class _RecordingMessages:
    def __init__(self, inner: Any, log: list[dict[str, Any]]) -> None:
        self._inner = inner
        self._log = log

    def create(self, **kwargs: Any) -> Any:
        try:
            response = self._inner.create(**kwargs)
        except Exception as exc:
            # Só o tipo e o status HTTP: texto do provedor não é guardado.
            self._log.append(
                {
                    "error": type(exc).__name__,
                    "status": getattr(exc, "status_code", None),
                }
            )
            raise
        usage = response.usage
        self._log.append(
            {
                "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
                "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
                "cache_creation_input_tokens": int(
                    getattr(usage, "cache_creation_input_tokens", 0) or 0
                ),
                "cache_read_input_tokens": int(
                    getattr(usage, "cache_read_input_tokens", 0) or 0
                ),
            }
        )
        return response


class _RecordingClient:
    def __init__(self, client: Any, log: list[dict[str, Any]]) -> None:
        self.messages = _RecordingMessages(client.messages, log)


@dataclass
class InstrumentedReviewer(AnthropicReviewer):
    """``AnthropicReviewer`` que registra o uso (tokens, cache de prompt, HTTP)."""

    usage_log: list[dict[str, Any]] = field(
        default_factory=list, init=False, repr=False
    )

    def _get_client(self) -> Any:
        return _RecordingClient(super()._get_client(), self.usage_log)


# ---------------------------------------------------------------------------
# Cache e paralelismo
# ---------------------------------------------------------------------------


def cache_key(
    model: str,
    prompt_version: str,
    remove_fields: Sequence[str],
    dossier_json: str,
) -> str:
    """sha256 de (modelo, versão do prompt, campos removidos, JSON do dossiê)."""
    payload = json.dumps(
        [model, prompt_version, sorted(remove_fields), dossier_json],
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _has_forbidden_key(node: Any) -> bool:
    if isinstance(node, Mapping):
        return any(
            k in FORBIDDEN_KEYS or _has_forbidden_key(v) for k, v in node.items()
        )
    if isinstance(node, list):
        return any(_has_forbidden_key(v) for v in node)
    return False


def result_from_dict(data: Mapping[str, Any]) -> ReviewResult:
    """Reconstrói o ``ReviewResult`` a partir de ``dataclasses.asdict``."""
    d = dict(data)
    agent_a = AgentResponse(**d.pop("agent_a"))
    agent_b = AgentResponse(**d.pop("agent_b"))
    arb = d.pop("arbiter", None)
    arbiter = AgentResponse(**arb) if arb else None
    return ReviewResult(agent_a=agent_a, agent_b=agent_b, arbiter=arbiter, **d)


class CachedParallelReviewer:
    """``PairReviewer`` com cache em disco, chamadas em paralelo e teto de gasto.

    O cache é um JSONL por (modelo, versão do prompt) em ``cache_dir``; a chave de
    cada linha é o ``cache_key`` do dossiê. Um ``LLMCallError`` é contado e
    registrado em ``errors.jsonl`` (só tipos e status), nunca cacheado como veredito.
    ``max_api_reviews`` limita as revisões de par enviadas à API por instância
    (erros contam); ao estourar, ``ReviewBudgetExceeded`` para tudo de forma limpa
    e o que já foi gravado fica no cache.
    """

    def __init__(
        self,
        *,
        cache_dir: Path | str = DEFAULT_CACHE_DIR,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        prompt_version: str = PROMPT_VERSION,
        remove_fields: Sequence[str] = REMOVE_FIELDS,
        max_api_reviews: int = 1200,
        max_retries: int = 4,
        retry_delay_s: float = 3.0,
        reviewer_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.model = model
        self.prompt_version = prompt_version
        self.remove_fields = tuple(remove_fields)
        self.max_api_reviews = int(max_api_reviews)
        self._api_key = api_key
        self._max_retries = max_retries
        self._retry_delay_s = retry_delay_s
        self._factory = reviewer_factory or self._default_factory
        self._lock = threading.Lock()
        self._local = threading.local()
        self._api_reviews = 0
        self.errors: Counter[str] = Counter()
        self.http_errors: Counter[str] = Counter()
        self.cache_hits = 0
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        safe_model = model.replace("/", "_").replace(":", "_")
        self.path = self.cache_dir / f"{safe_model}__{prompt_version}.jsonl"
        self.error_path = self.cache_dir / "errors.jsonl"
        self._cache: dict[str, dict[str, Any]] = self._load()
        self._pids: set[str] = {r["pair_id"] for r in self._cache.values()}

    def __repr__(self) -> str:  # sem a chave
        return (
            f"CachedParallelReviewer(model={self.model!r}, "
            f"prompt_version={self.prompt_version!r}, cached={len(self._cache)})"
        )

    # -- infraestrutura ---------------------------------------------------

    def _default_factory(self) -> InstrumentedReviewer:
        if not self._api_key:
            raise MissingApiKeyError("chave da API não informada")
        return InstrumentedReviewer(
            api_key=self._api_key,
            prompt_version=self.prompt_version,
            model=self.model,
            max_retries=self._max_retries,
            retry_delay_s=self._retry_delay_s,
        )

    def _load(self) -> dict[str, dict[str, Any]]:
        cache: dict[str, dict[str, Any]] = {}
        if not self.path.exists():
            return cache
        with self.path.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:  # linha final truncada
                    continue
                cache[rec["key"]] = rec
        return cache

    def _thread_reviewer(self) -> Any:
        rev = getattr(self._local, "reviewer", None)
        if rev is None:
            rev = self._factory()
            self._local.reviewer = rev
        return rev

    def _append(self, path: Path, record: Mapping[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    # -- API pública --------------------------------------------------------

    def key_for(self, dossier: DossierView) -> str:
        payload = dossier.to_dict()
        if _has_forbidden_key(payload):
            raise ValueError("dossiê contém campo proibido (nota_final/model_outputs)")
        return cache_key(
            self.model,
            self.prompt_version,
            self.remove_fields,
            dossier.to_json(sort_keys=True),
        )

    def __len__(self) -> int:
        return len(self._cache)

    @property
    def api_reviews(self) -> int:
        return self._api_reviews

    def has_pair(self, pair_id: str) -> bool:
        """True se já há veredito cacheado para o ``pair_id``."""
        with self._lock:
            return pair_id in self._pids

    def records_by_pair(self) -> dict[str, dict[str, Any]]:
        """Todos os registros do cache, por ``pair_id`` (cópia rasa)."""
        with self._lock:
            return {r["pair_id"]: r for r in self._cache.values()}

    def record_for(self, dossier: DossierView) -> dict[str, Any] | None:
        """Registro cacheado do dossiê (ou ``None``)."""
        with self._lock:
            return self._cache.get(self.key_for(dossier))

    def review(self, dossier: DossierView) -> ReviewResult:
        """Resultado do cache ou, se faltar, de uma chamada real (e grava)."""
        key = self.key_for(dossier)
        with self._lock:
            rec = self._cache.get(key)
            if rec is not None:
                self.cache_hits += 1
        if rec is not None:
            return result_from_dict(rec["result"])
        return self._review_api(dossier, key)

    def prefetch(
        self,
        dossiers: Sequence[DossierView],
        max_workers: int = 8,
        *,
        max_errors: int | None = None,
    ) -> dict[str, int]:
        """Busca em paralelo os dossiês que faltam no cache.

        Levanta ``ReviewBudgetExceeded`` (teto) ou ``ReviewAbort`` (mais de
        ``max_errors`` erros) depois de cancelar o que não começou; o que já foi
        gravado permanece no cache. Erros de revisão (``LLMCallError``) são
        contados e não interrompem o lote.
        """
        todo: dict[str, DossierView] = {}
        for d in dossiers:
            key = self.key_for(d)
            with self._lock:
                cached = key in self._cache
            if not cached:
                todo.setdefault(key, d)
        summary = {"requested": len(todo), "done": 0, "errors": 0}
        stop: ReviewStopped | None = None
        with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as pool:
            futures = {pool.submit(self._review_api, d, k): k for k, d in todo.items()}
            for fut in as_completed(futures):
                try:
                    fut.result()
                    summary["done"] += 1
                except CancelledError:  # não começou: foi cancelado pela parada
                    continue
                except ReviewBudgetExceeded as exc:
                    stop = stop or exc
                except LLMCallError:
                    summary["errors"] += 1
                    if max_errors is not None and summary["errors"] > max_errors:
                        stop = stop or ReviewAbort(
                            f"mais de {max_errors} erros de revisão no lote"
                        )
                if stop is not None:
                    for pending in futures:
                        pending.cancel()
        if stop is not None:
            raise stop
        return summary

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "api_reviews": self._api_reviews,
                "cache_hits": self.cache_hits,
                "cached": len(self._cache),
                "errors": dict(self.errors),
                "http_errors": dict(self.http_errors),
            }

    # -- núcleo -------------------------------------------------------------

    def _reserve(self) -> None:
        with self._lock:
            if self._api_reviews >= self.max_api_reviews:
                raise ReviewBudgetExceeded(
                    f"teto de {self.max_api_reviews} revisões na API atingido"
                )
            self._api_reviews += 1

    def _review_api(self, dossier: DossierView, key: str) -> ReviewResult:
        self._reserve()
        reviewer = self._thread_reviewer()
        log: list[dict[str, Any]] | None = getattr(reviewer, "usage_log", None)
        if log is not None:
            log.clear()
        try:
            result = reviewer.review(dossier)
        except (LLMCallError, ValueError, KeyError) as exc:
            events = list(log) if log is not None else []
            self._record_error(dossier, exc, events)
            if isinstance(exc, LLMCallError):
                raise
            raise LLMCallError(
                f"resposta inválida ({type(exc).__name__})",
                model=self.model,
                pair_id=dossier.pair_id,
                attempts=1,
            ) from exc
        events = list(log) if log is not None else []
        usage = Counter[str]()
        http: Counter[str] = Counter()
        calls = 0
        for ev in events:
            if "error" in ev:
                http[str(ev.get("status") or ev["error"])] += 1
            else:
                calls += 1
                for name, value in ev.items():
                    usage[name] += value
        tokens_p = result.agent_a.tokens_prompt + result.agent_b.tokens_prompt
        tokens_c = result.agent_a.tokens_completion + result.agent_b.tokens_completion
        if result.arbiter is not None:
            tokens_p += result.arbiter.tokens_prompt
            tokens_c += result.arbiter.tokens_completion
        record = {
            "key": key,
            "pair_id": dossier.pair_id,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "remove_fields": list(self.remove_fields),
            "summary": result.to_dict(),
            "result": dataclasses.asdict(result),
            "api_calls": calls if log is not None else None,
            "usage": dict(usage),
            "http_errors": dict(http),
            "tokens_prompt": tokens_p,
            "tokens_completion": tokens_c,
            "latency_s": result.total_latency_s,
        }
        with self._lock:
            self._cache[key] = record
            self._pids.add(record["pair_id"])
            self.http_errors.update(http)
            self._append(self.path, record)
        return result

    def _record_error(
        self,
        dossier: DossierView,
        exc: BaseException,
        events: list[dict[str, Any]],
    ) -> None:
        http = Counter(
            str(e.get("status") or e["error"]) for e in events if "error" in e
        )
        with self._lock:
            self.errors[type(exc).__name__] += 1
            self.http_errors.update(http)
            self._append(
                self.error_path,
                {
                    "ts": time.time(),
                    "pair_id": dossier.pair_id,
                    "error": type(exc).__name__,
                    "http": dict(http),
                    "model": self.model,
                },
            )


# ---------------------------------------------------------------------------
# Dados e amostra
# ---------------------------------------------------------------------------


@dataclass
class Extract:
    """Extrato sem ``PAR``/``TARGET`` e vetores de apoio (verdade só para medir)."""

    df: pd.DataFrame
    nota: np.ndarray
    band: np.ndarray
    truth: np.ndarray
    below: np.ndarray
    pair_ids: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=object))


def band_of_nota(nota: np.ndarray) -> np.ndarray:
    """Índice da faixa (0..5) de cada nota, com limites [a, b)."""
    return np.searchsorted(_EDGES, nota, side="right")


def load_extract(path: Path | str, *, check_partition: bool = True) -> Extract:
    raw = load_comparador_csv(Path(path), cfg=LoadConfig(require_par=False))
    if "SCORE" not in raw.columns:
        legacy = [c for c in raw.columns if str(c).upper().startswith("SCORE,")]
        if legacy:  # cabeçalho dBase: ``SCORE,C,15,0``
            raw = raw.assign(SCORE=pd.to_numeric(raw[legacy[0]], errors="coerce"))
    par = pd.to_numeric(raw["PAR"], errors="coerce")
    truth = par.isin([1, 2]).to_numpy()
    below = np.asarray(
        below_cutoff(raw, expected=THESIS_PARTITION if check_partition else None),
        dtype=bool,
    )
    df = raw.drop(columns=[c for c in ("PAR", "TARGET") if c in raw.columns])
    df = df.reset_index(drop=True)
    nota = pd.to_numeric(get_nota_series(df), errors="coerce").to_numpy(dtype=float)
    pair_ids = np.array(
        [
            make_pair_id(c, r, p)
            for c, r, p in zip(df["COMPREC"], df["REFREC"], df["PASSO"], strict=True)
        ],
        dtype=object,
    )
    return Extract(df, nota, band_of_nota(nota), truth, below, pair_ids)


def make_view(row: pd.Series) -> DossierView:
    """Dossiê da variante da fila: sem ``model_outputs`` e sem ``nota_final``."""
    return DossierView(
        build_dossier(row), remove_model_outputs=True, remove_fields=REMOVE_FIELDS
    )


def select_sample(
    truth: np.ndarray,
    band: np.ndarray,
    valid: np.ndarray,
    *,
    seed: int = SEED,
    per_band: int = 100,
) -> np.ndarray:
    """Posições da amostra: todos os pares + até ``per_band`` não pares por faixa.

    A ordem devolvida é embaralhada (um teto parcial não enviesa as faixas).
    """
    rng = np.random.default_rng(seed)
    parts = [np.flatnonzero(truth & valid)]
    for b in range(len(BAND_NAMES)):
        neg = np.flatnonzero(~truth & valid & (band == b))
        if len(neg) > per_band:
            neg = np.sort(rng.choice(neg, size=per_band, replace=False))
        parts.append(neg)
    idx = np.concatenate(parts)
    return rng.permutation(idx)


def select_pilot(
    sample: np.ndarray,
    truth: np.ndarray,
    band: np.ndarray,
    n: int,
    *,
    seed: int = SEED + 1,
) -> np.ndarray:
    """Subconjunto da amostra: ~2 não pares por faixa e o resto de pares."""
    rng = np.random.default_rng(seed)
    pos_pool = sample[truth[sample]]
    chosen: list[np.ndarray] = []
    n_neg = 0
    for b in range(len(BAND_NAMES)):
        pool = sample[~truth[sample] & (band[sample] == b)]
        take = pool[: min(2, len(pool), n - n_neg)]
        chosen.append(take)
        n_neg += len(take)
    n_pos = max(0, n - n_neg)
    chosen.append(rng.choice(pos_pool, size=min(n_pos, len(pos_pool)), replace=False))
    return rng.permutation(np.concatenate(chosen))


# ---------------------------------------------------------------------------
# Estatística
# ---------------------------------------------------------------------------


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float, float]:
    """Proporção e IC de Wilson 95% (``nan`` se ``n == 0``)."""
    if n <= 0:
        return (math.nan, math.nan, math.nan)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (p, max(0.0, centre - half), min(1.0, centre + half))


@dataclass
class Cell:
    """Contagens de uma célula (faixa × lado) da amostra."""

    pos: int = 0  # pares na amostra (censo do extrato)
    neg: int = 0  # não pares na amostra
    neg_pop: int = 0  # não pares no extrato
    tp: int = 0
    fn: int = 0
    unsure_pos: int = 0
    fp: int = 0
    tn: int = 0
    unsure_neg: int = 0
    arbiter: int = 0
    missing: int = 0  # sem veredito no cache (pendentes ou com erro)

    def add(self, other: Cell) -> Cell:
        values = (
            getattr(self, f.name) + getattr(other, f.name)
            for f in dataclasses.fields(self)
        )
        return Cell(*values)

    @property
    def pos_r(self) -> int:
        """Pares com veredito."""
        return self.tp + self.fn + self.unsure_pos

    @property
    def neg_r(self) -> int:
        """Não pares com veredito."""
        return self.fp + self.tn + self.unsure_neg

    @property
    def decisions(self) -> int:
        return self.pos_r + self.neg_r


def vpp_population(
    cells: Sequence[Cell], *, draws: int = 20000, seed: int = 0
) -> tuple[float, float, float]:
    """VPP do LLM como aceitador reponderado para o extrato, com IC (Jeffreys MC).

    O ponto é a mediana a posteriori (com 0 FP observado, o plug-in daria 100%).

    A amostra tem todos os pares (censo: VP é exato) e só uma fração dos não
    pares; o VPP bruto da amostra superestima. Aqui o FP do extrato é
    ``neg_pop`` × taxa de FP da célula (Beta de Jeffreys na incerteza da taxa).
    """
    rng = np.random.default_rng(seed)
    fp_tot = np.zeros(draws)
    tp = sum(c.tp for c in cells)
    for c in cells:
        if c.neg_pop > 0:
            if c.neg_r == 0:
                return (math.nan, math.nan, math.nan)
            fp_tot += c.neg_pop * rng.beta(c.fp + 0.5, c.neg_r - c.fp + 0.5, draws)
    if tp == 0 and not fp_tot.any():
        return (math.nan, math.nan, math.nan)
    vpp = tp / (tp + fp_tot)
    lo, mid, hi = np.percentile(vpp, [2.5, 50.0, 97.5])
    return (float(mid), float(lo), float(hi))


# ---------------------------------------------------------------------------
# Execução
# ---------------------------------------------------------------------------


def _views(ex: Extract, idx: np.ndarray) -> list[DossierView]:
    return [make_view(ex.df.iloc[int(i)]) for i in idx]


def _log_run(cache_dir: Path, entry: Mapping[str, Any]) -> None:
    with (cache_dir / "runs.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


def _sample_of(ex: Extract, seed: int, per_band: int) -> np.ndarray:
    valid = ~np.isnan(ex.nota)
    return select_sample(ex.truth, ex.band, valid, seed=seed, per_band=per_band)


def _print_sample_sizes(ex: Extract, sample: np.ndarray) -> None:
    t = ex.truth[sample]
    print(
        f"amostra: {len(sample)} pares (verdadeiros {int(t.sum())}: "
        f"{int((t & ~ex.below[sample]).sum())} acima, "
        f"{int((t & ex.below[sample]).sum())} abaixo do corte; "
        f"não pares {int((~t).sum())})"
    )
    for b, name in enumerate(BAND_NAMES):
        in_b = ex.band[sample] == b
        print(
            f"  {name}: pares {int((t & in_b).sum())}, "
            f"não pares {int((~t & in_b).sum())}"
        )


def run_batch(
    ex: Extract,
    idx: np.ndarray,
    *,
    cmd: str,
    workers: int,
    max_api_reviews: int,
    cache_dir: Path,
    model: str,
    max_errors: int,
) -> CachedParallelReviewer:
    reviewer = CachedParallelReviewer(
        cache_dir=cache_dir,
        api_key=load_api_key(),
        model=model,
        max_api_reviews=max_api_reviews,
    )
    views = _views(ex, idx)
    before = len(reviewer)
    t0 = time.monotonic()
    status = "ok"
    summary: dict[str, int] = {}
    try:
        summary = reviewer.prefetch(views, workers, max_errors=max_errors)
    except ReviewStopped as exc:
        status = type(exc).__name__
        print(f"PARADA: {exc}", file=sys.stderr)
    elapsed = time.monotonic() - t0
    stats = reviewer.stats()
    _log_run(
        cache_dir,
        {
            "cmd": cmd,
            "workers": workers,
            "n": len(idx),
            "cached_before": before,
            "elapsed_s": round(elapsed, 1),
            "status": status,
            "api_reviews": stats["api_reviews"],
            "errors": stats["errors"],
            "http_errors": stats["http_errors"],
            "model": model,
        },
    )
    print(
        f"{cmd}: {status}; {summary or ''} revisões na API {stats['api_reviews']}, "
        f"erros {stats['errors']}, HTTP {stats['http_errors']}, {elapsed:.1f}s"
    )
    return reviewer


def _records(
    reviewer: CachedParallelReviewer, views: Sequence[DossierView]
) -> list[dict[str, Any] | None]:
    return [reviewer.record_for(v) for v in views]


def pilot_summary(recs: Sequence[dict[str, Any] | None]) -> dict[str, Any]:
    got = [r for r in recs if r is not None]
    if not got:
        return {"n": 0}
    calls = [r["api_calls"] for r in got if r.get("api_calls") is not None]
    usage = Counter[str]()
    for r in got:
        usage.update(r.get("usage", {}))
    n_calls = sum(calls) or 1
    lat = np.array([r["latency_s"] for r in got], dtype=float)
    return {
        "n": len(got),
        "calls_per_pair": float(np.mean(calls)) if calls else math.nan,
        "arbiter_frac": float(
            np.mean([r["summary"]["protocol"] == "arbiter" for r in got])
        ),
        "input_per_call": usage["input_tokens"] / n_calls,
        "output_per_call": usage["output_tokens"] / n_calls,
        "cache_create_per_call": usage["cache_creation_input_tokens"] / n_calls,
        "cache_read_per_call": usage["cache_read_input_tokens"] / n_calls,
        "latency_mean_s": float(lat.mean()),
        "latency_p95_s": float(np.percentile(lat, 95)),
        "decisions": dict(Counter(r["summary"]["decision"] for r in got)),
        "http_errors": dict(
            sum((Counter(r.get("http_errors", {})) for r in got), Counter())
        ),
    }


def cmd_pilot(args: argparse.Namespace) -> int:
    ex = load_extract(args.input)
    sample = _sample_of(ex, args.seed, args.per_band)
    _print_sample_sizes(ex, sample)
    idx = select_pilot(sample, ex.truth, ex.band, args.n)
    reviewer = run_batch(
        ex,
        idx,
        cmd="pilot",
        workers=args.workers,
        max_api_reviews=args.max_api_reviews,
        cache_dir=Path(args.cache_dir),
        model=args.model,
        max_errors=args.max_errors,
    )
    summary = pilot_summary(_records(reviewer, _views(ex, idx)))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    ex = load_extract(args.input)
    sample = _sample_of(ex, args.seed, args.per_band)
    _print_sample_sizes(ex, sample)
    run_batch(
        ex,
        sample,
        cmd="calibrate",
        workers=args.workers,
        max_api_reviews=args.max_api_reviews,
        cache_dir=Path(args.cache_dir),
        model=args.model,
        max_errors=args.max_errors,
    )
    return 0


# ---------------------------------------------------------------------------
# Relatório
# ---------------------------------------------------------------------------


def build_cells(
    ex: Extract,
    sample: np.ndarray,
    recs: Sequence[dict[str, Any] | None],
) -> dict[tuple[str, str], Cell]:
    """Contagens por (faixa, lado); ``recs`` alinhado a ``sample``."""
    cells: dict[tuple[str, str], Cell] = {
        (b, s): Cell() for b in BAND_NAMES for s in SIDES
    }
    for b, name in enumerate(BAND_NAMES):
        for s, is_below in (("acima", False), ("abaixo", True)):
            sel = (ex.band == b) & (ex.below == is_below) & ~np.isnan(ex.nota)
            cells[(name, s)].neg_pop = int((sel & ~ex.truth).sum())
    for i, rec in zip(sample, recs, strict=True):
        i = int(i)
        c = cells[(BAND_NAMES[int(ex.band[i])], "abaixo" if ex.below[i] else "acima")]
        truth = bool(ex.truth[i])
        if truth:
            c.pos += 1
        else:
            c.neg += 1
        if rec is None:
            c.missing += 1
            continue
        decision = rec["summary"]["decision"]
        if truth:
            c.tp += decision == "MATCH"
            c.fn += decision == "NONMATCH"
            c.unsure_pos += decision == "UNSURE"
        else:
            c.fp += decision == "MATCH"
            c.tn += decision == "NONMATCH"
            c.unsure_neg += decision == "UNSURE"
        c.arbiter += rec["summary"]["protocol"] == "arbiter"
    return cells


def _fmt(p: float, lo: float | None = None, hi: float | None = None) -> str:
    if math.isnan(p):
        return "n/d"
    s = f"{100 * p:.1f}%".replace(".", ",")
    if lo is None or hi is None or math.isnan(lo):
        return s
    return f"{s} [{100 * lo:.1f}–{100 * hi:.1f}]".replace(".", ",")


def _rate(r: Mapping[str, Any], name: str) -> str:
    """Taxa com IC de uma linha agregada (``name`` = prefixo das colunas)."""
    return _fmt(r[f"{name}_taxa"], r[f"{name}_ic_inf"], r[f"{name}_ic_sup"])


def _vpp(r: Mapping[str, Any], name: str) -> str:
    return _fmt(r[name], r[f"{name}_ic_inf"], r[f"{name}_ic_sup"])


def aggregate_rows(cells: Mapping[tuple[str, str], Cell]) -> list[dict[str, Any]]:
    """Linhas agregadas: faixas × lados, mais totais (com e sem ``low``)."""
    groups: list[tuple[str, str, list[tuple[str, str]]]] = []
    for name in BAND_NAMES:
        groups.append((name, "todos", [(name, s) for s in SIDES]))
        for s in SIDES:
            groups.append((name, s, [(name, s)]))
    nolow = [b for b in BAND_NAMES if b != "low"]
    for label, bands in (("total", list(BAND_NAMES)), ("total_sem_low", nolow)):
        groups.append((label, "todos", [(b, s) for b in bands for s in SIDES]))
        for s in SIDES:
            groups.append((label, s, [(b, s) for b in bands]))
    rows: list[dict[str, Any]] = []
    for band, side, keys in groups:
        parts = [cells[k] for k in keys]
        c = Cell()
        for p in parts:
            c = c.add(p)
        n_dec = c.decisions
        fp = wilson(c.fp, c.neg_r)
        fn = wilson(c.fn, c.pos_r)
        un = wilson(c.unsure_pos + c.unsure_neg, n_dec)
        arb = wilson(c.arbiter, n_dec)
        vpp_raw = wilson(c.tp, c.tp + c.fp)
        vpp_pop = vpp_population(parts)
        rows.append(
            {
                "faixa": band,
                "lado": side,
                "n_pares": c.pos,
                "n_nao_pares": c.neg,
                "n_nao_pares_extrato": c.neg_pop,
                "vp": c.tp,
                "fn": c.fn,
                "unsure_pares": c.unsure_pos,
                "fp": c.fp,
                "vn": c.tn,
                "unsure_nao_pares": c.unsure_neg,
                "pendentes": c.missing,
                "fp_taxa": fp[0],
                "fp_ic_inf": fp[1],
                "fp_ic_sup": fp[2],
                "fn_taxa": fn[0],
                "fn_ic_inf": fn[1],
                "fn_ic_sup": fn[2],
                "unsure_taxa": un[0],
                "unsure_ic_inf": un[1],
                "unsure_ic_sup": un[2],
                "arbitro_taxa": arb[0],
                "arbitro_ic_inf": arb[1],
                "arbitro_ic_sup": arb[2],
                "vpp_amostra": vpp_raw[0],
                "vpp_amostra_ic_inf": vpp_raw[1],
                "vpp_amostra_ic_sup": vpp_raw[2],
                "vpp_extrato": vpp_pop[0],
                "vpp_extrato_ic_inf": vpp_pop[1],
                "vpp_extrato_ic_sup": vpp_pop[2],
            }
        )
    return rows


def _usage_totals(recs: Sequence[dict[str, Any] | None]) -> dict[str, int]:
    tot: Counter[str] = Counter()
    for r in recs:
        if r is None:
            continue
        tot["pares"] += 1
        tot["chamadas"] += int(r.get("api_calls") or 0)
        tot["arbitros"] += r["summary"]["protocol"] == "arbiter"
        tot["tokens_entrada"] += int(r["tokens_prompt"])
        tot["tokens_saida"] += int(r["tokens_completion"])
        u = r.get("usage", {})
        tot["cache_escrita"] += int(u.get("cache_creation_input_tokens", 0))
        tot["cache_leitura"] += int(u.get("cache_read_input_tokens", 0))
    return dict(tot)


def _ledger(cache_dir: Path) -> dict[str, Any]:
    path = cache_dir / "runs.jsonl"
    out: dict[str, Any] = {
        "relogio_s": 0.0,
        "execucoes": 0,
        "http": Counter(),
        "erros": 0,
    }
    err = cache_dir / "errors.jsonl"
    if err.exists():
        out["erros"] = len(err.read_text(encoding="utf-8").splitlines())
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        out["relogio_s"] += float(e.get("elapsed_s", 0))
        out["execucoes"] += 1
        out["http"].update(e.get("http_errors", {}))
    return out


def render_markdown(
    rows: Sequence[Mapping[str, Any]],
    usage: Mapping[str, int],
    ledger: Mapping[str, Any],
    sizes: Mapping[str, Any],
    model: str,
) -> str:
    by = {(r["faixa"], r["lado"]): r for r in rows}
    out: list[str] = []
    w = out.append
    w("# Revisor LLM real (Claude Haiku, sem a nota) — taxas por faixa\n")
    w(
        "> **Aviso:** medição com LLM real numa amostra estratificada; **não é "
        "avaliação da fila**. O dossiê tem só escores numéricos (mesmo protocolo da "
        "tese), sem `model_outputs` e sem `features.nota_final`; a nota está oculta "
        "ao LLM. Apenas agregados; nada em nível de par.\n"
    )
    w("## Protocolo\n")
    w(
        f"- Modelo `{model}`, prompt `v4`, Agente A + Agente B e árbitro só na "
        "divergência, `max_tokens=2500`, sem temperatura.\n"
        f"- Amostra (semente {sizes['seed']}): **todos** os pares verdadeiros "
        f"({sizes['pos']}: {sizes['pos_acima']} acima e {sizes['pos_abaixo']} abaixo "
        f"do corte) e até {sizes['per_band']} não pares por faixa ({sizes['neg']} "
        f"no total); {sizes['n']} pares.\n"
        "- Faixas pela nota v3.1: low [0,5), grey_low [5,6), grey_mid [6,7), "
        "grey_high [7,8), near_high [8,9), high [9,∞). Lado do corte: `SCORE` "
        "contra o corte da tese por `PASSO`.\n"
        "- Verdade: PAR ∈ {1, 2}. FP = MATCH entre não pares; FN = NONMATCH "
        "entre pares (UNSURE fica nos denominadores); UNSURE e árbitro sobre as "
        "decisões. IC de Wilson 95%.\n"
        "- VPP do LLM como aceitador (MATCH = aceita): **VPP (amostra)** = "
        "VP/(VP+FP) na amostra, com IC de Wilson, superestimado porque só uma "
        "fração dos não pares entrou; **VPP (extrato)** reescala os não pares da "
        "célula (faixa × lado) para o total do extrato (mediana a posteriori e "
        "IC de Jeffreys por simulação, 20 mil sorteios).\n"
    )
    w("## Taxas por faixa e lado do corte\n")
    w(
        "| faixa | lado | pares | não pares | FP | FP % [IC] | FN | FN % [IC] "
        "| UNSURE % [IC] | árbitro % | VPP amostra % [IC] | VPP extrato % [IC] |"
    )
    w("|---|---|--:|--:|--:|---|--:|---|---|---|---|---|")
    for r in rows:
        w(
            f"| {r['faixa']} | {r['lado']} | {r['n_pares']} | {r['n_nao_pares']} "
            f"| {r['fp']} | {_rate(r, 'fp')} "
            f"| {r['fn']} | {_rate(r, 'fn')} "
            f"| {_rate(r, 'unsure')} "
            f"| {_fmt(r['arbitro_taxa'])} "
            f"| {_vpp(r, 'vpp_amostra')} "
            f"| {_vpp(r, 'vpp_extrato')} |"
        )
    w("")
    w("## Comparação com as taxas sintéticas da simulação\n")
    w(
        "Taxas sintéticas: Laplace sobre a vigilância da tese (nota **visível**); "
        "`low` herdou `grey_low` (nunca medido). Medido: faixa inteira (os dois "
        "lados), nota **oculta**.\n"
    )
    w(
        "| faixa | FP sint. | FP medido [IC] | FN sint. | FN medido [IC] "
        "| UNSURE sint. | UNSURE medido [IC] |"
    )
    w("|---|--:|---|--:|---|--:|---|")
    for name in BAND_NAMES:
        r = by[(name, "todos")]
        s = SYNTHETIC_RATES[name]
        w(
            f"| {name} | {_fmt(s[0])} | {_rate(r, 'fp')} "
            f"| {_fmt(s[1])} | {_rate(r, 'fn')} "
            f"| {_fmt(s[2])} | {_rate(r, 'unsure')} |"
        )
    w("")
    tv = THESIS_VPP
    t = by[("total", "abaixo")]
    ts = by[("total_sem_low", "abaixo")]
    w("## Comparação com a tese\n")
    w(
        f"- Tese (nota **visível**): VPP {_fmt(tv[2])}, {tv[0]}/{tv[1]}, "
        f"IC {_fmt(tv[3])}–{_fmt(tv[4])}.\n"
        f"- Aqui (nota **oculta**), abaixo do corte, VPP reescalado ao extrato: "
        f"{_vpp(t, 'vpp_extrato')} "
        f"com todas as faixas e "
        f"{_vpp(ts, 'vpp_extrato')} "
        "sem a faixa `low`. Populações e seleção diferem (a tese mediu pares "
        "escolhidos pela fila, não uma amostra estratificada), então a comparação "
        "é de ordem de grandeza, não de igualdade.\n"
    )
    w("## Custo da medição\n")

    def n(key: str) -> str:
        return f"{usage.get(key, 0):,}".replace(",", ".")

    minimum = 2 * usage.get("pares", 0) + usage.get("arbitros", 0)
    extra = usage.get("chamadas", 0) - minimum
    w(
        f"- Pares revisados: {n('pares')} (árbitro em {n('arbitros')}); chamadas "
        f"à API: {n('chamadas')} ({extra} a mais que o mínimo A+B(+árbitro), por "
        f"nova tentativa de resposta truncada ou inválida); erros de revisão "
        f"registrados (par sem veredito na tentativa): {ledger['erros']}.\n"
        f"- Tokens de entrada (sem cache): {n('tokens_entrada')}; de saída: "
        f"{n('tokens_saida')}; escrita no cache de prompt: {n('cache_escrita')}; "
        f"leitura do cache de prompt: {n('cache_leitura')}.\n"
        f"- Tempo de relógio (soma das execuções, piloto incluído): "
        f"{ledger['relogio_s']:.0f} s em {ledger['execucoes']} execuções; "
        f"erros HTTP (429 etc.): {dict(ledger['http']) or 'nenhum'}.\n"
    )
    return "\n".join(out) + "\n"


def cmd_report(args: argparse.Namespace) -> int:
    ex = load_extract(args.input)
    sample = _sample_of(ex, args.seed, args.per_band)
    cache_dir = Path(args.cache_dir)
    reviewer = CachedParallelReviewer(
        cache_dir=cache_dir, api_key="report-only", model=args.model
    )
    recs = _records(reviewer, _views(ex, sample))
    cells = build_cells(ex, sample, recs)
    rows = aggregate_rows(cells)
    pos_s = ex.truth[sample]
    sizes = {
        "seed": args.seed,
        "per_band": args.per_band,
        "n": len(sample),
        "pos": int(pos_s.sum()),
        "neg": int((~pos_s).sum()),
        "pos_acima": int((pos_s & ~ex.below[sample]).sum()),
        "pos_abaixo": int((pos_s & ex.below[sample]).sum()),
    }
    missing = sum(r is None for r in recs)
    usage = _usage_totals(recs)
    ledger = _ledger(cache_dir)
    md = render_markdown(rows, usage, ledger, sizes, args.model)
    out_md, out_csv = Path(args.out_md), Path(args.out_csv)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md, encoding="utf-8")
    pd.DataFrame(rows).to_csv(out_csv, index=False, float_format="%.6f")
    print(f"relatório: {out_md} e {out_csv}; pares sem veredito no cache: {missing}")
    return 0


# ---------------------------------------------------------------------------
# Fixture de decisões e FixtureReviewer
# ---------------------------------------------------------------------------

FIXTURE_DIR = ROOT / "data" / "llm_fixtures"
FIXTURE_STEM = "haiku-5-5_v4_sem-nota"
AGENTS: tuple[str, ...] = ("agent_a", "agent_b", "arbiter")


class FixtureMissingError(LookupError):
    """Par ausente do fixture com ``on_missing="error"``."""


def fixture_row(rec: Mapping[str, Any]) -> dict[str, Any]:
    """Linha do fixture a partir de um registro do cache: só decisões e números.

    Sem ``raw_json``, ``evidence_summary``, ``quality_flags`` nem texto livre.
    ``reason_codes`` são códigos de um conjunto fechado do protocolo.
    """
    r = rec["result"]
    row: dict[str, Any] = {
        "pair_id": rec["pair_id"],
        "model": rec["model"],
        "prompt_version": rec["prompt_version"],
        "removed_fields": list(rec["remove_fields"]),
        "decision": r["decision"],
        "protocol": r["protocol"],
        "confidence": r["confidence"],
        "reason_codes": list(r["reason_codes"]),
        "total_latency_s": r["total_latency_s"],
        "tokens_prompt": rec["tokens_prompt"],
        "tokens_completion": rec["tokens_completion"],
    }
    for name in AGENTS:
        a = r.get(name)
        row[f"{name}_decision"] = a["decision"] if a else None
        row[f"{name}_confidence"] = a["confidence"] if a else None
        row[f"{name}_reason_codes"] = list(a["reason_codes"]) if a else None
        row[f"{name}_tokens_prompt"] = a["tokens_prompt"] if a else None
        row[f"{name}_tokens_completion"] = a["tokens_completion"] if a else None
        row[f"{name}_latency_s"] = a["latency_s"] if a else None
    return row


FIXTURE_KEYS: frozenset[str] = frozenset(
    fixture_row(
        {
            "pair_id": "",
            "model": "",
            "prompt_version": "",
            "remove_fields": [],
            "tokens_prompt": 0,
            "tokens_completion": 0,
            "result": {
                "decision": "",
                "protocol": "",
                "confidence": 0,
                "reason_codes": [],
                "total_latency_s": 0,
            },
        }
    )
)


def result_from_fixture_row(row: Mapping[str, Any]) -> ReviewResult:
    """``ReviewResult`` de uma linha do fixture (campos de texto livre vazios)."""

    def agent(name: str) -> AgentResponse | None:
        if row.get(f"{name}_decision") is None:
            return None
        return AgentResponse(
            decision=row[f"{name}_decision"],
            confidence=row[f"{name}_confidence"],
            reason_codes=list(row[f"{name}_reason_codes"]),
            evidence_summary={},
            quality_flags={},
            raw_json={},
            latency_s=row[f"{name}_latency_s"],
            tokens_prompt=row[f"{name}_tokens_prompt"],
            tokens_completion=row[f"{name}_tokens_completion"],
        )

    agent_a, agent_b = agent("agent_a"), agent("agent_b")
    if agent_a is None or agent_b is None:
        raise ValueError("linha de fixture sem agent_a/agent_b")
    return ReviewResult(
        pair_id=row["pair_id"],
        decision=row["decision"],
        confidence=row["confidence"],
        reason_codes=list(row["reason_codes"]),
        evidence_summary={},
        quality_flags={},
        protocol=row["protocol"],
        agent_a=agent_a,
        agent_b=agent_b,
        arbiter=agent("arbiter"),
        total_latency_s=row["total_latency_s"],
    )


def export_fixture(
    records: Sequence[Mapping[str, Any]],
    out_dir: Path | str,
    *,
    stem: str = FIXTURE_STEM,
    origin: Mapping[str, Any] | None = None,
    rates_note: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Grava o JSONL do fixture (ordenado por ``pair_id``) e o manifesto."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = sorted((fixture_row(r) for r in records), key=lambda x: x["pair_id"])
    models = sorted({r["model"] for r in rows})
    prompts = sorted({r["prompt_version"] for r in rows})
    removed = sorted({tuple(r["removed_fields"]) for r in rows})
    if len(models) > 1 or len(prompts) > 1 or len(removed) > 1:
        raise ValueError("registros de mais de uma configuração no mesmo fixture")
    body = "".join(
        json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows
    )
    path = out / f"{stem}.jsonl"
    path.write_bytes(body.encode("utf-8"))
    manifest: dict[str, Any] = {
        "model": models[0] if models else None,
        "prompt_version": prompts[0] if prompts else None,
        "removed_fields": list(removed[0]) if removed else [],
        "date": time.strftime("%Y-%m-%d"),
        "n_pairs": len(rows),
        "jsonl": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "contents": "decisions, confidences, reason codes and token counts; "
        "no raw_json, evidence_summary or free text",
        "origin": dict(origin or {}),
    }
    if rates_note:
        manifest["rates_note"] = rates_note
    (out / f"{stem}.manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return path, manifest


def load_rates_csv(path: Path | str) -> dict[str, tuple[float, float, float]]:
    """Taxas ``band,fp,fn,unsure`` de um CSV."""
    df = pd.read_csv(path)
    missing = {"band", "fp", "fn", "unsure"} - set(df.columns)
    if missing:
        raise ValueError(f"CSV de taxas sem colunas {sorted(missing)}")
    return {
        str(r.band): (float(r.fp), float(r.fn), float(r.unsure))
        for r in df.itertuples(index=False)
    }


class FixtureReviewer:
    """``PairReviewer`` que reaplica as decisões reais gravadas no fixture.

    Cada par presente no JSONL volta como o ``ReviewResult`` original (decisão,
    confiança, protocolo, decisões/confianças/códigos dos agentes e tokens; os
    campos de texto livre ficam vazios). Política para par ausente:

    * ``on_missing="error"`` (padrão): levanta :class:`FixtureMissingError`.
    * ``on_missing="rates"``: usa as taxas por faixa (``rates``: ``band ->
      (fp, fn, unsure)``, ver ``load_rates_csv``) por meio de um
      ``OracleReviewer`` determinístico por ``pair_id``. Precisa de ``truth``
      (``pair_id -> bool``) e ``band_by_pair`` (``pair_id -> nome da faixa``);
      ``disagreement_rate`` (padrão 0) controla quantos pares vão ao árbitro.
    """

    def __init__(
        self,
        path: Path | str,
        *,
        on_missing: str = "error",
        rates: Mapping[str, tuple[float, float, float]] | None = None,
        truth: Mapping[str, bool] | None = None,
        band_by_pair: Mapping[str, str] | None = None,
        seed: int = 0,
        disagreement_rate: float = 0.0,
    ) -> None:
        if on_missing not in ("error", "rates"):
            raise ValueError("on_missing deve ser 'error' ou 'rates'")
        if on_missing == "rates" and (
            rates is None or truth is None or band_by_pair is None
        ):
            raise ValueError("on_missing='rates' exige rates, truth e band_by_pair")
        self.on_missing = on_missing
        self.n_replayed = 0
        self.n_missing = 0
        self._rows: dict[str, dict[str, Any]] = {}
        with Path(path).open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    row = json.loads(line)
                    self._rows[row["pair_id"]] = row
        self._band = band_by_pair
        self._oracles: dict[str, Any] = {}
        if on_missing == "rates":
            assert rates is not None and truth is not None
            self._oracles = {
                band: OracleReviewer(
                    truth,
                    fp_rate=fp,
                    fn_rate=fn,
                    unsure_rate=unsure,
                    disagreement_rate=disagreement_rate,
                    seed=seed,
                )
                for band, (fp, fn, unsure) in rates.items()
            }

    def __len__(self) -> int:
        return len(self._rows)

    def __contains__(self, pair_id: object) -> bool:
        return pair_id in self._rows

    def review(self, dossier: DossierView) -> ReviewResult:
        row = self._rows.get(dossier.pair_id)
        if row is not None:
            self.n_replayed += 1
            return result_from_fixture_row(row)
        self.n_missing += 1
        if self.on_missing == "error":
            raise FixtureMissingError(f"par ausente do fixture: {dossier.pair_id}")
        assert self._band is not None
        band = self._band[dossier.pair_id]
        oracle = self._oracles[band]
        result: ReviewResult = oracle.review(dossier)
        oracle.seen_payloads.clear()
        return result


def calibration_cells(
    ex: Extract, reviewer: CachedParallelReviewer, seed: int, per_band: int
) -> tuple[np.ndarray, dict[tuple[str, str], Cell]]:
    """Amostra de calibração e suas contagens, a partir do cache (sem API)."""
    sample = _sample_of(ex, seed, per_band)
    by_pair = reviewer.records_by_pair()
    recs = [by_pair.get(str(ex.pair_ids[int(i)])) for i in sample]
    return sample, build_cells(ex, sample, recs)


def measured_rates(
    cells: Mapping[tuple[str, str], Cell],
) -> dict[str, tuple[float, float, float]]:
    """Taxas (fp, fn, unsure) por faixa, sem Laplace, dos dois lados juntos."""
    out: dict[str, tuple[float, float, float]] = {}
    for name in BAND_NAMES:
        c = Cell()
        for s in SIDES:
            c = c.add(cells[(name, s)])
        fp = c.fp / c.neg_r if c.neg_r else math.nan
        fn = c.fn / c.pos_r if c.pos_r else math.nan
        un = (c.unsure_pos + c.unsure_neg) / c.decisions if c.decisions else math.nan
        out[name] = (fp, fn, un)
    return out


RATES_NOTE = (
    "Rates are raw proportions (no Laplace smoothing) measured on the "
    "calibration sample, both sides of the cutoff together, nota hidden. "
    "'low' has only 3 true pairs (FN is 2/3) and 'high' only 2 non-pairs "
    "(FP is 0/2): treat those two bands as unmeasured. All the non-pairs of the "
    "sample lie below the cutoff."
)


def cmd_export_fixture(args: argparse.Namespace) -> int:
    ex = load_extract(args.input)
    cache_dir = Path(args.cache_dir)
    reviewer = CachedParallelReviewer(
        cache_dir=cache_dir, api_key="export-only", model=args.model
    )
    sample, cells = calibration_cells(ex, reviewer, args.seed, args.per_band)
    sample_pids = {str(ex.pair_ids[int(i)]) for i in sample}
    records = list(reviewer.records_by_pair().values())
    n_cal = sum(r["pair_id"] in sample_pids for r in records)
    pool = np.flatnonzero(ex.below)
    top = pool[base_order(ex.nota[pool], ex.pair_ids[pool])][: max(RERANK_WINDOWS)]
    cached = {r["pair_id"] for r in records}
    rerank_pids = {str(ex.pair_ids[int(i)]) for i in top} & cached
    n_rerank = len(rerank_pids - sample_pids)
    origin = {
        "precedence": "calibration_sample > rerank_test > queue_runs_and_baselines",
        "calibration_sample": {
            "seed": args.seed,
            "per_band": args.per_band,
            "n_pairs": n_cal,
        },
        "rerank_test": {
            "windows": list(RERANK_WINDOWS),
            "pool": "pairs below the cutoff, ordered by decreasing nota",
            "n_pairs": n_rerank,
        },
        "queue_runs_and_baselines": {"n_pairs": len(records) - n_cal - n_rerank},
    }
    out_dir = Path(args.fixture_dir)
    path, manifest = export_fixture(
        records, out_dir, origin=origin, rates_note=RATES_NOTE
    )
    rates = measured_rates(cells)
    rates_path = out_dir / f"{FIXTURE_STEM}.rates.csv"
    pd.DataFrame(
        [(b, *rates[b]) for b in BAND_NAMES], columns=["band", "fp", "fn", "unsure"]
    ).to_csv(rates_path, index=False, float_format="%.6f")
    print(f"fixture: {path} ({manifest['n_pairs']} pares)")
    print(f"sha256: {manifest['sha256']}")
    print(f"taxas: {rates_path}")
    return 0


# ---------------------------------------------------------------------------
# Fila ponta a ponta com o LLM real
# ---------------------------------------------------------------------------

QUEUE_DIR_NAME = "queue"
GLOBAL_CAP = 3000
BASELINE_HUMAN_BUDGETS: tuple[int, ...] = (50, 100, 200, 400)


class RunTimeLimit(ReviewStopped):
    """Tempo máximo desta execução atingido (retomada gratuita pelo cache)."""


class BudgetLedger:
    """Teto global de NOVAS revisões de par na API, somando todas as execuções."""

    def __init__(self, path: Path, cap: int = GLOBAL_CAP) -> None:
        self.path = path
        self.cap = cap
        self.used = 0
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            self.used = int(data.get("used", 0))
            self.cap = int(data.get("cap", cap))
        self._base = self.used

    @property
    def remaining(self) -> int:
        return max(0, self.cap - self.used)

    def sync(self, process_reviews: int) -> None:
        """Registra o total desta execução (``reviewer.api_reviews``)."""
        self.used = self._base + int(process_reviews)
        self.path.write_text(
            json.dumps({"cap": self.cap, "used": self.used}), encoding="utf-8"
        )


def queue_config(negative_labels: str, seed: int = SEED) -> SelfCalibratingConfig:
    """Configuração PADRÃO atual da fila (humano, composites, 50 por rodada)."""
    return SelfCalibratingConfig(
        seed=seed,
        negative_labels=negative_labels,  # type: ignore[arg-type]
    )


def _cached_prefix(reviewer: CachedParallelReviewer, ids: Sequence[str]) -> int:
    n = 0
    for pid in ids:
        if not reviewer.has_pair(pid):
            break
        n += 1
    return n


def lookahead(
    session: SelfCalibratingSession,
    reviewer: CachedParallelReviewer,
    view_of: Callable[[str], DossierView],
    *,
    workers: int,
    k_hot: int,
    k_cold: int,
    k_low: int,
    k_explore: int,
    hot: Collection[str] = (),
    ledger: BudgetLedger | None = None,
) -> int:
    """Pré-busca em paralelo os próximos pares de cada braço (``peek_upcoming``).

    Um braço cujo trecho já cacheado é menor que ``k_low`` é reabastecido até
    ``k_hot`` pares se for "quente" (``hot``: já consumido na rodada), ``k_cold``
    se for frio e ``k_explore`` para a exploração; os demais não gastam nada.
    Devolve quantos pares foram pedidos à API.
    """
    peek = session.peek_upcoming(max(k_hot, k_cold, k_explore))
    need: dict[str, None] = {}
    for label, ids in peek.items():
        arm_size = k_hot if label in hot else k_cold
        size = k_explore if label == EXPLORE_ARM else arm_size
        if _cached_prefix(reviewer, ids) >= min(k_low, size, len(ids)):
            continue
        for pid in ids[:size]:
            if not reviewer.has_pair(pid):
                need[pid] = None
    if not need:
        return 0
    try:
        reviewer.prefetch([view_of(p) for p in need], workers, max_errors=25)
    finally:
        if ledger is not None:
            ledger.sync(reviewer.api_reviews)
    return len(need)


def _recal_summary(
    ev: Mapping[str, Mapping[str, Any]], prior: tuple[float, list[float]] | None
) -> dict[str, Any] | None:
    rec = ev.get("recalibration")
    if rec is None:
        return None
    hold = ev.get("holdout_built", {})
    out: dict[str, Any] = {
        "status": rec["status"],
        "n_labeled": rec["n_labeled"],
        "n_pos": rec["n_pos"],
        "n_neg": rec["n_neg"],
        "kish_effective_n": hold.get("effective_n"),
        "intercept": rec["intercept"],
    }
    if prior is not None and rec.get("coef"):
        delta = np.asarray(rec["coef"], dtype=float) - np.asarray(prior[1], dtype=float)
        out["coef_l2_vs_prior"] = float(np.linalg.norm(delta))
        out["coef_maxabs_vs_prior"] = float(np.abs(delta).max())
        out["intercept_vs_prior"] = float(rec["intercept"] - prior[0])
    return out


def summarize_queue(
    session: SelfCalibratingSession,
    ex: Extract,
    reviewer: CachedParallelReviewer,
    prior: tuple[float, list[float]] | None,
) -> dict[str, Any]:
    """Agregados por rodada e do run (sem nenhum identificador de par)."""
    res = session.results()
    pid_idx = {str(p): i for i, p in enumerate(ex.pair_ids)}
    idx = res["pair_id"].map(pid_idx).to_numpy(dtype=int)
    res = res.assign(truth=ex.truth[idx], below=ex.below[idx])
    by_round: dict[int, dict[str, Mapping[str, Any]]] = {}
    for e in session.events:
        by_round.setdefault(e.round, {})[e.kind] = e.payload
    rounds: list[dict[str, Any]] = []
    cum = Counter[str]()
    for rnd, g in res.groupby("round"):
        ev = by_round.get(int(rnd), {})
        human = g["human_decision_source"].eq("human")
        verified = human & g["human_decision"].eq("MATCH") & g["truth"]
        llm_match = g["llm_decision"].eq("MATCH")
        row: dict[str, Any] = {
            "round": int(rnd),
            "llm_reviews": len(g),
            "llm_match": int(llm_match.sum()),
            "llm_nonmatch": int(g["llm_decision"].eq("NONMATCH").sum()),
            "llm_unsure": int(g["llm_decision"].eq("UNSURE").sum()),
            "llm_error": int(g["llm_decision"].eq("ERROR").sum()),
            "llm_match_true": int((llm_match & g["truth"]).sum()),
            "gate_items": int(ev.get("gate_opened", {}).get("n_items", 0)),
            "over_budget": int(g["human_decision_source"].eq("over_budget").sum()),
            "human_decisions": int(human.sum()),
            "verified_true": int(verified.sum()),
            "fp_caught": int(
                (human & llm_match & g["human_decision"].eq("NONMATCH")).sum()
            ),
            "below_verified": int((verified & g["below"]).sum()),
            "audit_matches": int((g["audited"] & verified).sum()),
            "stop_reason": ev.get("round_stopped", {}).get("reason"),
            "complete": "recalibration" in ev,
            "recalibration": _recal_summary(ev, prior),
        }
        for key in (
            "llm_reviews",
            "human_decisions",
            "verified_true",
            "below_verified",
        ):
            cum[key] += row[key]
        row["cum_llm_reviews"] = cum["llm_reviews"]
        row["cum_human_decisions"] = cum["human_decisions"]
        row["cum_verified_true"] = cum["verified_true"]
        row["cum_below_verified"] = cum["below_verified"]
        rounds.append(row)
    by_pair = reviewer.records_by_pair()
    use = Counter[str]()
    for pid in res["pair_id"]:
        rec = by_pair.get(pid)
        if rec is None:
            continue
        use["calls"] += int(rec.get("api_calls") or 0)
        use["tokens_prompt"] += int(rec["tokens_prompt"])
        use["tokens_completion"] += int(rec["tokens_completion"])
        u = rec.get("usage", {})
        use["cache_write"] += int(u.get("cache_creation_input_tokens", 0))
        use["cache_read"] += int(u.get("cache_read_input_tokens", 0))
        use["arbiter"] += rec["summary"]["protocol"] == "arbiter"
    summary = session.summary()
    return {
        "finish_reason": summary["finish_reason"],
        "finished": summary["finished"],
        "n_reviewed": summary["n_reviewed"],
        "rounds": rounds,
        "totals": {
            "llm_reviews": int(len(res)),
            "llm_match": int(res["llm_decision"].eq("MATCH").sum()),
            "human_decisions": int(cum["human_decisions"]),
            "over_budget": int(sum(r["over_budget"] for r in rounds)),
            "verified_true": int(cum["verified_true"]),
            "fp_caught": int(sum(r["fp_caught"] for r in rounds)),
            "below_verified": int(cum["below_verified"]),
            "below_total": int((ex.truth & ex.below).sum()),
            "true_total": int(ex.truth.sum()),
        },
        "usage": dict(use),
    }


def drive_queue(
    ex: Extract,
    reviewer: CachedParallelReviewer,
    *,
    negative_labels: str,
    workers: int,
    k_hot: int,
    k_cold: int,
    k_low: int,
    k_explore: int,
    ledger: BudgetLedger,
    max_minutes: float | None,
    out_dir: Path,
) -> dict[str, Any]:
    """Roda a sessão até o fim (ou até o teto/limite), com humano perfeito."""
    cfg = queue_config(negative_labels)
    session = SelfCalibratingSession(ex.df, reviewer, cfg)
    pid_idx = {str(p): i for i, p in enumerate(ex.pair_ids)}

    def view_of(pid: str) -> DossierView:
        return DossierView(
            build_dossier(
                ex.df.iloc[pid_idx[pid]], pair_id=pid, include_macd=cfg.include_macd
            ),
            remove_model_outputs=True,
            remove_fields=cfg.remove_fields,
        )

    deadline = None if max_minutes is None else time.monotonic() + 60 * max_minutes
    status, note = "finished", None
    prior: tuple[float, list[float]] | None = None
    last = "start"
    arm_use: Counter[str] = Counter()
    try:
        while not session.finished:
            if deadline is not None and time.monotonic() > deadline:
                raise RunTimeLimit(f"limite de {max_minutes} min desta execução")
            if session.awaiting_human:
                decisions = {
                    it["pair_id"]: (
                        "MATCH" if ex.truth[pid_idx[it["pair_id"]]] else "NONMATCH"
                    )
                    for it in session.gate_items
                }
                session.submit_human_decisions(decisions)  # type: ignore[arg-type]
                continue
            if session.phase == "reviewing" and last in (
                "ordering",
                "reordering",
                "final_verdict",
                "review_error",
            ):
                lookahead(
                    session,
                    reviewer,
                    view_of,
                    workers=workers,
                    k_hot=k_hot,
                    k_cold=k_cold,
                    k_low=k_low,
                    k_explore=k_explore,
                    hot={label for label, n in arm_use.items() if n >= 2},
                    ledger=ledger,
                )
            event = session.step()
            last = event.kind
            if last == "pair_presented":
                arm_use[event.payload["band_label"]] += 1
            elif last == "reordering":
                arm_use.clear()
            model = session.current_model
            if prior is None and model is not None:
                prior = (float(model.intercept), [float(c) for c in model.coef])
    except ReviewStopped as exc:
        status, note = type(exc).__name__, str(exc)
    finally:
        ledger.sync(reviewer.api_reviews)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"queue_{negative_labels}"
    if status != "finished":
        (out_dir / f"{tag}.state.json").write_text(
            json.dumps(session.to_state()), encoding="utf-8"
        )
    session.results().to_csv(out_dir / f"{tag}.results.csv", index=False)
    summary = summarize_queue(session, ex, reviewer, prior)
    summary.update(
        {
            "policy": negative_labels,
            "status": status,
            "stop_note": note,
            "phase": session.phase,
            "config": {
                "seed": cfg.seed,
                "field_set": cfg.field_set,
                "mode": cfg.mode,
                "human_budget_per_round": cfg.human_budget_per_round,
                "max_reviews": cfg.max_reviews,
                "max_rounds": cfg.max_rounds,
                "negative_labels": cfg.negative_labels,
            },
        }
    )
    return summary


def cmd_queue_run(args: argparse.Namespace) -> int:
    ex = load_extract(args.input)
    cache_dir = Path(args.cache_dir)
    out_dir = cache_dir / QUEUE_DIR_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    ledger = BudgetLedger(out_dir / "budget.json", args.global_cap)
    reviewer = CachedParallelReviewer(
        cache_dir=cache_dir,
        api_key=load_api_key(),
        model=args.model,
        max_api_reviews=ledger.remaining,
        max_retries=args.max_retries,
        retry_delay_s=args.retry_delay,
    )
    before = len(reviewer)
    t0 = time.monotonic()
    summary = drive_queue(
        ex,
        reviewer,
        negative_labels=args.negative_labels,
        workers=args.workers,
        k_hot=args.k_hot,
        k_cold=args.k_cold,
        k_low=args.k_low,
        k_explore=args.k_explore,
        ledger=ledger,
        max_minutes=args.max_minutes,
        out_dir=out_dir,
    )
    elapsed = time.monotonic() - t0
    stats = reviewer.stats()
    (out_dir / f"queue_{args.negative_labels}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    _log_run(
        cache_dir,
        {
            "cmd": f"queue-{args.negative_labels}",
            "workers": args.workers,
            "cached_before": before,
            "elapsed_s": round(elapsed, 1),
            "status": summary["status"],
            "api_reviews": stats["api_reviews"],
            "errors": stats["errors"],
            "http_errors": stats["http_errors"],
            "model": args.model,
        },
    )
    t = summary["totals"]
    print(
        f"queue-run {args.negative_labels}: {summary['status']} "
        f"({summary['finish_reason']}); LLM {t['llm_reviews']}, humano "
        f"{t['human_decisions']}, verdadeiros {t['verified_true']}, abaixo "
        f"{t['below_verified']}/{t['below_total']}; API nesta execução "
        f"{stats['api_reviews']} (global {ledger.used}/{ledger.cap}); "
        f"erros {stats['errors']}, HTTP {stats['http_errors']}; {elapsed:.0f}s"
    )
    if summary["stop_note"]:
        print(f"parada: {summary['stop_note']}")
    return 0


def baseline_order(ex: Extract) -> np.ndarray:
    """Pool ``all`` pela nota decrescente (empate por ``pair_id``)."""
    frame = pd.DataFrame({"neg": -np.nan_to_num(ex.nota, nan=-1e9), "pid": ex.pair_ids})
    return frame.sort_values(["neg", "pid"], kind="mergesort").index.to_numpy()


def cmd_baseline(args: argparse.Namespace) -> int:
    ex = load_extract(args.input)
    cache_dir = Path(args.cache_dir)
    out_dir = cache_dir / QUEUE_DIR_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    ledger = BudgetLedger(out_dir / "budget.json", args.global_cap)
    n_llm = int(args.n_llm)
    if n_llm <= 0:
        counts = [
            json.loads(p.read_text(encoding="utf-8"))["totals"]["llm_reviews"]
            for p in sorted(out_dir.glob("queue_*.json"))
            if not p.name.endswith(".state.json")
        ]
        n_llm = max(counts, default=0)
    if n_llm <= 0:
        raise SystemExit("informe --n-llm (ou rode o queue-run antes)")
    reviewer = CachedParallelReviewer(
        cache_dir=cache_dir,
        api_key=load_api_key(),
        model=args.model,
        max_api_reviews=ledger.remaining,
        max_retries=args.max_retries,
        retry_delay_s=args.retry_delay,
    )
    order = baseline_order(ex)
    top = order[:n_llm]
    views = [make_view(ex.df.iloc[int(i)]) for i in top]
    before = len(reviewer)
    t0 = time.monotonic()
    deadline = None if args.max_minutes is None else t0 + 60 * args.max_minutes
    status, note = "finished", None
    chunk = max(1, 4 * args.workers)
    try:
        for start in range(0, len(views), chunk):
            if deadline is not None and time.monotonic() > deadline:
                raise RunTimeLimit(f"limite de {args.max_minutes} min desta execução")
            reviewer.prefetch(views[start : start + chunk], args.workers, max_errors=25)
            ledger.sync(reviewer.api_reviews)
    except ReviewStopped as exc:
        status, note = type(exc).__name__, str(exc)
    finally:
        ledger.sync(reviewer.api_reviews)
    elapsed = time.monotonic() - t0
    by_pair = reviewer.records_by_pair()
    match: list[int] = []
    unsure: list[int] = []
    for i in top:
        rec = by_pair.get(str(ex.pair_ids[int(i)]))
        if rec is None:
            break  # só o prefixo contíguo com veredito entra na base
        decision = rec["summary"]["decision"]
        match.append(int(decision == "MATCH"))
        unsure.append(int(decision == "UNSURE"))
    n_eff = len(match)
    h0_n = max(2500, n_llm)
    h0 = order[:h0_n]
    payload = {
        "n_requested": n_llm,
        "n_effective": n_eff,
        "status": status if n_eff == n_llm or status != "finished" else "partial",
        "stop_note": note,
        "h1": {
            "match": match,
            "unsure": unsure,
            "truth": ex.truth[top[:n_eff]].astype(int).tolist(),
            "below": ex.below[top[:n_eff]].astype(int).tolist(),
        },
        "h0": {
            "truth": ex.truth[h0].astype(int).tolist(),
            "below": ex.below[h0].astype(int).tolist(),
        },
        "pool": {
            "true_total": int(ex.truth.sum()),
            "below_total": int((ex.truth & ex.below).sum()),
            "n_pairs": int(len(ex.df)),
        },
    }
    (out_dir / "baseline.json").write_text(json.dumps(payload), encoding="utf-8")
    stats = reviewer.stats()
    _log_run(
        cache_dir,
        {
            "cmd": "baseline-h1",
            "workers": args.workers,
            "cached_before": before,
            "elapsed_s": round(elapsed, 1),
            "status": payload["status"],
            "api_reviews": stats["api_reviews"],
            "errors": stats["errors"],
            "http_errors": stats["http_errors"],
            "model": args.model,
        },
    )
    print(
        f"baseline H1: {payload['status']}, LLM {n_eff}/{n_llm}; API nesta execução "
        f"{stats['api_reviews']} (global {ledger.used}/{ledger.cap}); "
        f"erros {stats['errors']}, HTTP {stats['http_errors']}; {elapsed:.0f}s"
    )
    if note:
        print(f"parada: {note}")
    return 0


# ---------------------------------------------------------------------------
# Relatório da fila ponta a ponta
# ---------------------------------------------------------------------------


def _first_reach(cum: np.ndarray, target: int) -> int | None:
    """Menor posição (1-based) em que o acumulado atinge ``target``."""
    hit = np.flatnonzero(cum >= target)
    return int(hit[0]) + 1 if hit.size else None


def h1_at_human(base: Mapping[str, Any], budget: int) -> dict[str, Any]:
    """H1 com ``budget`` decisões humanas (o humano confirma só os MATCH)."""
    m = np.asarray(base["h1"]["match"], dtype=int)
    t = np.asarray(base["h1"]["truth"], dtype=int)
    b = np.asarray(base["h1"]["below"], dtype=int)
    pos = _first_reach(np.cumsum(m), budget)
    truncated = pos is None
    r = len(m) if truncated else int(pos)
    sel = m[:r] == 1
    return {
        "human": int(sel.sum()),
        "llm_reviews": r,
        "verified": int((sel & (t[:r] == 1)).sum()),
        "below": int((sel & (t[:r] == 1) & (b[:r] == 1)).sum()),
        "truncated": truncated,
    }


def h1_at_llm(base: Mapping[str, Any], n_llm: int) -> dict[str, Any]:
    """H1 depois de ``n_llm`` revisões do LLM (humano confirma todo MATCH)."""
    m = np.asarray(base["h1"]["match"], dtype=int)[:n_llm]
    t = np.asarray(base["h1"]["truth"], dtype=int)[:n_llm]
    b = np.asarray(base["h1"]["below"], dtype=int)[:n_llm]
    sel = m == 1
    return {
        "human": int(sel.sum()),
        "llm_reviews": len(m),
        "verified": int((sel & (t == 1)).sum()),
        "below": int((sel & (t == 1) & (b == 1)).sum()),
        "truncated": len(m) < n_llm,
    }


def h0_at_human(base: Mapping[str, Any], budget: int) -> dict[str, int]:
    """H0: humano sozinho, ``budget`` decisões na ordem da nota."""
    t = np.asarray(base["h0"]["truth"], dtype=int)[:budget]
    b = np.asarray(base["h0"]["below"], dtype=int)[:budget]
    return {"human": len(t), "verified": int(t.sum()), "below": int((t & b).sum())}


def _ledger_by_cmd(cache_dir: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    path = cache_dir / "runs.jsonl"
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        d = out.setdefault(
            e["cmd"],
            {
                "elapsed_s": 0.0,
                "runs": 0,
                "api": 0,
                "http": Counter(),
                "workers": set(),
            },
        )
        d["elapsed_s"] += float(e.get("elapsed_s", 0))
        d["runs"] += 1
        d["api"] += int(e.get("api_reviews", 0))
        d["http"].update(e.get("http_errors", {}))
        d["workers"].add(e.get("workers"))
    return out


def _n(x: Any) -> str:
    return "n/d" if x is None else f"{x:,}".replace(",", ".")


def _f(x: float | None, nd: int = 2) -> str:
    return "n/d" if x is None else f"{x:.{nd}f}".replace(".", ",")


def queue_report_rows(
    runs: Mapping[str, Mapping[str, Any]], base: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Linhas longas (tabela, politica, ponto, metrica, valor) do CSV."""
    rows: list[dict[str, Any]] = []

    def add(table: str, policy: str, point: Any, metric: str, value: Any) -> None:
        rows.append(
            {
                "tabela": table,
                "politica": policy,
                "ponto": point,
                "metrica": metric,
                "valor": value,
            }
        )

    for policy, run in runs.items():
        for r in run["rounds"]:
            for key, val in r.items():
                if key == "recalibration" and val:
                    for k2, v2 in val.items():
                        add("rodadas", policy, r["round"], f"recal_{k2}", v2)
                elif key != "recalibration":
                    add("rodadas", policy, r["round"], key, val)
        for key, val in run["totals"].items():
            add("totais", policy, "run", key, val)
        add("totais", policy, "run", "finish_reason", run["finish_reason"])
        add("totais", policy, "run", "status", run["status"])
    for policy, run in runs.items():
        for r in run["rounds"]:
            n_h, n_l = r["cum_human_decisions"], r["cum_llm_reviews"]
            h1h, h1l = h1_at_human(base, n_h), h1_at_llm(base, n_l)
            h0 = h0_at_human(base, n_h)
            for name, val in (
                ("fila_verificados", r["cum_verified_true"]),
                ("fila_abaixo", r["cum_below_verified"]),
                ("h1_igual_humano_verificados", h1h["verified"]),
                ("h1_igual_humano_abaixo", h1h["below"]),
                ("h1_igual_humano_revisoes_llm", h1h["llm_reviews"]),
                ("h1_igual_llm_verificados", h1l["verified"]),
                ("h1_igual_llm_abaixo", h1l["below"]),
                ("h1_igual_llm_decisoes_humanas", h1l["human"]),
                ("h0_igual_humano_verificados", h0["verified"]),
                ("h0_igual_humano_abaixo", h0["below"]),
            ):
                add(
                    "comparacao_rodadas",
                    policy,
                    f"r{r['round']}:h{n_h}:l{n_l}",
                    name,
                    val,
                )
    for b_ in BASELINE_HUMAN_BUDGETS:
        h1, h0 = h1_at_human(base, b_), h0_at_human(base, b_)
        for name, val in (
            ("h1_verificados", h1["verified"]),
            ("h1_abaixo", h1["below"]),
            ("h1_revisoes_llm", h1["llm_reviews"]),
            ("h0_verificados", h0["verified"]),
            ("h0_abaixo", h0["below"]),
        ):
            add("bases_orcamento_humano", "base", b_, name, val)
    return rows


def queue_discussion(
    runs: Mapping[str, Mapping[str, Any]], base: Mapping[str, Any]
) -> list[str]:
    """Pontos da discussão, todos com números das tabelas."""
    out: list[str] = []
    for policy, run in runs.items():
        t, rounds = run["totals"], run["rounds"]
        last = rounds[-1]
        h = t["human_decisions"]
        h1h, h0 = h1_at_human(base, h), h0_at_human(base, h)
        h1l = h1_at_llm(base, t["llm_reviews"])
        out.append(
            f"- `{policy}`: {t['verified_true']} verdadeiros verificados "
            f"({t['below_verified']} de {t['below_total']} abaixo do corte) com "
            f"{h} decisões humanas e {_n(t['llm_reviews'])} revisões do LLM. Com as "
            f"mesmas {h} decisões, H1 verifica {h1h['verified']} ({h1h['below']} "
            f"abaixo) e H0 verifica {h0['verified']} ({h0['below']} abaixo); com as "
            f"mesmas {_n(t['llm_reviews'])} revisões do LLM, H1 verifica "
            f"{h1l['verified']} ({h1l['below']} abaixo) gastando {h1l['human']} "
            "decisões humanas."
        )
        out.append(
            f"- `{policy}`: {len(rounds)} rodadas de no máximo "
            f"{run['config']['max_rounds']}; a última ({last['round']}) parou por "
            f"`{last['stop_reason']}` e trouxe {last['verified_true']} verdadeiros "
            f"novos; motivo de parada do run: `{run['finish_reason']}` "
            f"(`max_reviews`={run['config']['max_reviews']} não foi atingido: "
            f"{_n(t['llm_reviews'])} revisões). O LLM marcou {t['llm_match']} MATCH; "
            f"{t['over_budget']} candidatos do portão (MATCH, UNSURE ou auditoria) "
            "ficaram sem decisão humana por falta de orçamento "
            f"({run['config']['human_budget_per_round']} por rodada)."
        )
    if len(runs) == 2:
        a, b = (runs[k] for k in runs)
        na, nb = list(runs)
        ra, rb = a["rounds"][0]["recalibration"], b["rounds"][0]["recalibration"]
        la, lb = a["rounds"][-1]["recalibration"], b["rounds"][-1]["recalibration"]
        out.append(
            f"- Políticas de negativos: na rodada 1, `{na}` recalibra com "
            f"{ra['n_labeled']} rótulos ({ra['n_pos']} pos, {ra['n_neg']} neg; "
            f"Kish {_f(ra['kish_effective_n'], 1)}; status `{ra['status']}`) e "
            f"`{nb}` com {rb['n_labeled']} ({rb['n_pos']} pos, {rb['n_neg']} neg; "
            f"status `{rb['status']}`). Ao fim, ‖Δcoef‖₂ vs prior é "
            f"{_f(la.get('coef_l2_vs_prior'))} em `{na}` e "
            f"{_f(lb.get('coef_l2_vs_prior'))} em `{nb}`; verdadeiros verificados: "
            f"{a['totals']['verified_true']} e {b['totals']['verified_true']}; "
            f"abaixo do corte: {a['totals']['below_verified']} e "
            f"{b['totals']['below_verified']}."
        )
    return out


def render_queue_markdown(
    runs: Mapping[str, Mapping[str, Any]],
    base: Mapping[str, Any],
    ledger: Mapping[str, Mapping[str, Any]],
    budget: Mapping[str, Any],
    model: str,
) -> str:
    out: list[str] = []
    w = out.append
    w("# Fila auto-calibrável ponta a ponta com o Haiku real (humano simulado)\n")
    w(
        "> **Aviso:** uma única execução com LLM real (temperatura padrão; o "
        "veredito estocástico fica congelado no fixture); humano simulado "
        "**perfeito** (decide todo item do portão pelo PAR); método experimental. "
        "Não é avaliação final da fila. Apenas agregados; nada em nível de par.\n"
    )
    pool = base["pool"]
    cfg = next(iter(runs.values()))["config"]
    w("## Protocolo\n")
    w(
        f"- Pool `all` do extrato ({_n(pool['n_pairs'])} pares; {pool['true_total']} "
        f"verdadeiros, {pool['below_total']} deles abaixo do corte). A sessão não vê "
        "`PAR` nem `TARGET`.\n"
        f"- Sessão com a configuração padrão: modo `{cfg['mode']}`, `field_set` "
        f"`{cfg['field_set']}`, {cfg['human_budget_per_round']} decisões humanas "
        f"por rodada, `max_reviews`={cfg['max_reviews']}, `max_rounds`="
        f"{cfg['max_rounds']}, semente {cfg['seed']}.\n"
        f"- Revisor: `{model}`, prompt `v4`, sem `model_outputs` e sem "
        "`features.nota_final`; cache em disco e pré-busca paralela dos próximos "
        "pares de cada braço (`peek_upcoming`).\n"
        "- Rótulos da recalibração: decisões humanas (verificadas) mais, conforme "
        "a política, os NONMATCH do LLM (`llm_nonmatch`) ou nenhum negativo sem "
        "verificação (`verified_only`).\n"
        "- H1: ordem da nota decrescente, LLM revisa, humano confirma só os MATCH. "
        "H0: humano sozinho na ordem da nota.\n"
    )
    for policy, run in runs.items():
        t = run["totals"]
        w(f"## Rodadas — `{policy}`\n")
        w(
            f"Status: **{run['status']}** (motivo de parada do run: "
            f"`{run['finish_reason']}`); {_n(t['llm_reviews'])} revisões do LLM.\n"
        )
        if run.get("stop_note"):
            w(f"Parada antecipada: {run['stop_note']}\n")
        w(
            "| rodada | revisões LLM | MATCH LLM | decisões humanas | over_budget "
            "| verdadeiros verificados | FP pegos | abaixo do corte (de "
            f"{t['below_total']}) | parada da rodada |"
        )
        w("|--:|--:|--:|--:|--:|--:|--:|--:|---|")
        for r in run["rounds"]:
            w(
                f"| {r['round']} | {r['llm_reviews']} | {r['llm_match']} "
                f"| {r['human_decisions']} | {r['over_budget']} "
                f"| {r['verified_true']} | {r['fp_caught']} "
                f"| {r['below_verified']} | {r['stop_reason'] or 'incompleta'} |"
            )
        w(
            f"| **total** | {t['llm_reviews']} | {t['llm_match']} "
            f"| {t['human_decisions']} | {t['over_budget']} | {t['verified_true']} "
            f"| {t['fp_caught']} | {t['below_verified']} | {run['finish_reason']} |\n"
        )
        w("Recalibração ao fim de cada rodada (rótulos verificados por humano):\n")
        w(
            "| rodada | status | rotulados | pos | neg | Kish | ‖Δcoef‖₂ vs prior "
            "| máx abs Δcoef | Δintercepto |"
        )
        w("|--:|---|--:|--:|--:|--:|--:|--:|--:|")
        for r in run["rounds"]:
            rc = r["recalibration"]
            if not rc:
                w(f"| {r['round']} | (sem recalibração) | | | | | | | |")
                continue
            w(
                f"| {r['round']} | {rc['status']} | {rc['n_labeled']} | {rc['n_pos']} "
                f"| {rc['n_neg']} | {_f(rc['kish_effective_n'], 1)} "
                f"| {_f(rc.get('coef_l2_vs_prior'))} "
                f"| {_f(rc.get('coef_maxabs_vs_prior'))} "
                f"| {_f(rc.get('intercept_vs_prior'))} |"
            )
        w("")
    w("## Comparação com H1 e H0\n")
    w(
        "**A decisões humanas iguais** (a fila decide até "
        f"{cfg['human_budget_per_round']} itens por rodada; os pontos são os "
        "acumulados ao fim de cada rodada; H1 gasta uma decisão por MATCH do LLM):\n"
    )
    w(
        "| política | rodada | decisões humanas | fila: verificados | fila: abaixo "
        "| H1: verificados | H1: abaixo | H1: revisões LLM | H0: verificados "
        "| H0: abaixo |"
    )
    w("|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    for policy, run in runs.items():
        for r in run["rounds"]:
            h = r["cum_human_decisions"]
            h1, h0 = h1_at_human(base, h), h0_at_human(base, h)
            star = "*" if h1["truncated"] else ""
            w(
                f"| {policy} | {r['round']} | {h} | {r['cum_verified_true']} "
                f"| {r['cum_below_verified']} | {h1['verified']}{star} "
                f"| {h1['below']}{star} | {h1['llm_reviews']} | {h0['verified']} "
                f"| {h0['below']} |"
            )
    w("")
    w(
        "**Orçamentos de referência das bases** (50/100/200/400 decisões "
        "humanas; a fila, com 50 por rodada e no máximo 5 rodadas, não passa de 250):\n"
    )
    w(
        "| decisões humanas | H1: verificados | H1: abaixo | H1: revisões LLM "
        "| H0: verificados | H0: abaixo |"
    )
    w("|--:|--:|--:|--:|--:|--:|")
    for b_ in BASELINE_HUMAN_BUDGETS:
        h1, h0 = h1_at_human(base, b_), h0_at_human(base, b_)
        star = "*" if h1["truncated"] else ""
        w(
            f"| {b_} | {h1['verified']}{star} | {h1['below']}{star} "
            f"| {h1['llm_reviews']} | {h0['verified']} | {h0['below']} |"
        )
    w("")
    w(
        f"H1 foi calculada sobre {_n(base['n_effective'])} revisões do LLM "
        f"(pedidas: {_n(base['n_requested'])}; status `{base['status']}`). "
        "`*` = a base não chegou a esse nº de decisões humanas dentro das revisões "
        "disponíveis (valor ao fim delas).\n"
    )
    w("**A revisões de LLM iguais** (o humano confirma todo MATCH do H1):\n")
    w(
        "| política | rodada | revisões LLM | fila: verificados | fila: abaixo "
        "| fila: decisões humanas | H1: verificados | H1: abaixo "
        "| H1: decisões humanas |"
    )
    w("|---|--:|--:|--:|--:|--:|--:|--:|--:|")
    for policy, run in runs.items():
        for r in run["rounds"]:
            n_l = r["cum_llm_reviews"]
            h1 = h1_at_llm(base, n_l)
            star = "*" if h1["truncated"] else ""
            w(
                f"| {policy} | {r['round']} | {n_l} | {r['cum_verified_true']} "
                f"| {r['cum_below_verified']} | {r['cum_human_decisions']} "
                f"| {h1['verified']}{star} | {h1['below']}{star} "
                f"| {h1['human']}{star} |"
            )
    w("")
    if len(runs) == 2:
        a, b = (runs[k]["totals"] for k in runs)
        names = list(runs)
        w("## Políticas de negativos\n")
        w("| métrica | " + " | ".join(f"`{n}`" for n in names) + " |")
        w("|---|--:|--:|")
        for label, key in (
            ("revisões do LLM", "llm_reviews"),
            ("decisões humanas", "human_decisions"),
            ("verdadeiros verificados", "verified_true"),
            ("abaixo do corte verificados", "below_verified"),
            ("FP do LLM pegos pelo humano", "fp_caught"),
            ("over_budget", "over_budget"),
        ):
            w(f"| {label} | {a[key]} | {b[key]} |")
        w(
            "| motivo de parada | "
            + " | ".join(f"`{runs[n]['finish_reason']}`" for n in names)
            + " |\n"
        )
    w("## Discussão\n")
    for line in queue_discussion(runs, base):
        w(line)
    w("")
    w("## Custo\n")
    w(
        "| etapa | pares com veredito | chamadas | tokens entrada | tokens saída "
        "| escrita cache | leitura cache | revisões novas na API | relógio (s) "
        "| workers | HTTP 429/erros |"
    )
    w("|---|--:|--:|--:|--:|--:|--:|--:|--:|---|---|")
    for policy, run in runs.items():
        u = run["usage"]
        led = ledger.get(f"queue-{policy}", {})
        w(
            f"| fila `{policy}` | {_n(run['totals']['llm_reviews'])} "
            f"| {_n(u.get('calls'))} | {_n(u.get('tokens_prompt'))} "
            f"| {_n(u.get('tokens_completion'))} | {_n(u.get('cache_write'))} "
            f"| {_n(u.get('cache_read'))} | {_n(led.get('api'))} "
            f"| {_n(round(led.get('elapsed_s', 0)))} "
            f"| {', '.join(str(x) for x in sorted(led.get('workers', [])))} "
            f"| {dict(led.get('http', {})) or 'nenhum'} |"
        )
    led = ledger.get("baseline-h1", {})
    w(
        f"| base H1 | {_n(base['n_effective'])} | | | | | | {_n(led.get('api'))} "
        f"| {_n(round(led.get('elapsed_s', 0)))} "
        f"| {', '.join(str(x) for x in sorted(led.get('workers', [])))} "
        f"| {dict(led.get('http', {})) or 'nenhum'} |\n"
    )
    w(
        "A primeira execução da fila `llm_nonmatch` usou 32 workers e teve as "
        "respostas 429 e os 2 erros de revisão registrados acima; as demais usaram "
        "24 workers, sem 429.\n"
    )
    w(
        f"Teto global de novas revisões de par na API: {budget['used']} usadas de "
        f"{budget['cap']}. As execuções são retomáveis: refazer a fila reaplica o "
        "que está no cache sem custo, então o relógio soma só o tempo útil de cada "
        "execução.\n"
    )
    return "\n".join(out) + "\n"


def cmd_queue_report(args: argparse.Namespace) -> int:
    cache_dir = Path(args.cache_dir)
    qdir = cache_dir / QUEUE_DIR_NAME
    runs: dict[str, dict[str, Any]] = {}
    for policy in ("llm_nonmatch", "verified_only"):
        path = qdir / f"queue_{policy}.json"
        if path.exists():
            runs[policy] = json.loads(path.read_text(encoding="utf-8"))
    if not runs:
        raise SystemExit("nenhum queue_*.json em data/llm_cache/queue/")
    base = json.loads((qdir / "baseline.json").read_text(encoding="utf-8"))
    budget = json.loads((qdir / "budget.json").read_text(encoding="utf-8"))
    md = render_queue_markdown(
        runs, base, _ledger_by_cmd(cache_dir), budget, args.model
    )
    out_md, out_csv = Path(args.out_md), Path(args.out_csv)
    out_md.write_text(md, encoding="utf-8")
    pd.DataFrame(queue_report_rows(runs, base)).to_csv(out_csv, index=False)
    print(f"relatório: {out_md} e {out_csv}")
    return 0


# ---------------------------------------------------------------------------
# Teste de reordenação (desempate) com o LLM abaixo do corte
# ---------------------------------------------------------------------------

STRATEGIES: tuple[str, ...] = ("N", "L1", "L2", "L3")
GROUP_OF: dict[str, int] = {"MATCH": 0, "UNSURE": 1, "NONMATCH": 2}
RERANK_WINDOWS: tuple[int, ...] = (500, 1000, 2000, 3000)
RERANK_BUDGETS: tuple[int, ...] = (25, 50, 100, 200, 400, 800, 1600)
BOOTSTRAP_BUDGETS: tuple[int, ...] = (50, 100, 200, 400)
BLOCK_WIDTH = 0.5
RERANK_CAP = 2500
BOOTSTRAP_REPS = 2000
CURVE_STEP = 10


def base_order(nota: np.ndarray, pair_ids: np.ndarray) -> np.ndarray:
    """Ordem N: nota decrescente, empate por ``pair_id`` (índices do vetor dado)."""
    frame = pd.DataFrame({"neg": -np.asarray(nota, dtype=float), "pid": pair_ids})
    return frame.sort_values(["neg", "pid"], kind="mergesort").index.to_numpy()


def score_block(nota: np.ndarray, width: float = BLOCK_WIDTH) -> np.ndarray:
    """Bloco de nota de largura ``width`` (índice crescente com a nota)."""
    return np.floor(np.asarray(nota, dtype=float) / width).astype(np.int64)


def window_order(
    strategy: str,
    rank: np.ndarray,
    group: np.ndarray,
    conf: np.ndarray,
    block: np.ndarray,
) -> np.ndarray:
    """Posições (índices dos vetores da janela) na ordem da estratégia.

    ``rank`` = posição na ordem N; ``group`` = 0 MATCH, 1 UNSURE, 2 NONMATCH;
    ``conf`` = confiança do LLM; ``block`` = bloco de nota (``score_block``).
    Só reordena a janela: nada é descartado.
    """
    if strategy == "N":
        return np.lexsort((rank,))
    if strategy == "L1":
        return np.lexsort((rank, group))
    if strategy == "L2":
        return np.lexsort((rank, group, -block))
    if strategy == "L3":
        return np.lexsort((rank, np.where(group == 0, -conf, 0.0), group))
    raise ValueError(f"estratégia desconhecida: {strategy!r}")


def full_order(
    strategy: str,
    n: int,
    group: np.ndarray,
    conf: np.ndarray,
    block: np.ndarray,
) -> np.ndarray:
    """Permutação de ``range(n)`` (posições na ordem N).

    Os vetores têm o tamanho da janela ``W``; as posições ``W..n-1`` (fora da
    janela) seguem na ordem da nota, depois da janela.
    """
    w = len(group)
    if w > n:
        raise ValueError("janela maior que o pool")
    head = window_order(strategy, np.arange(w), group, conf, block)
    return np.concatenate([head, np.arange(w, n)]).astype(np.int64)


def found_curve(order: np.ndarray, truth_n: np.ndarray, max_b: int) -> np.ndarray:
    """Verdadeiros acumulados após 1..``max_b`` decisões, nesta ordem."""
    return np.cumsum(np.asarray(truth_n, dtype=int)[order[:max_b]])


def found_at(order: np.ndarray, truth_n: np.ndarray, budget: int) -> int:
    return int(np.asarray(truth_n, dtype=int)[order[:budget]].sum())


def bootstrap_diff(
    strategy: str,
    truth_w: np.ndarray,
    group: np.ndarray,
    conf: np.ndarray,
    block: np.ndarray,
    budgets: Sequence[int],
    *,
    reps: int = BOOTSTRAP_REPS,
    seed: int = SEED,
) -> dict[int, dict[str, float]]:
    """Bootstrap pareado da diferença (estratégia − N) de verdadeiros achados.

    Reamostra os pares da janela com reposição, estratificado por verdade (mesmos
    tamanhos), e aplica N e a estratégia à MESMA amostra. Com ``B >= W`` os dois
    revisam o mesmo conjunto, então a diferença é 0. IC percentil de 95%.
    """
    truth = np.asarray(truth_w, dtype=bool)
    w = len(truth)
    pos, neg = np.flatnonzero(truth), np.flatnonzero(~truth)
    rng = np.random.default_rng(seed)
    out: dict[int, dict[str, float]] = {}
    rank = np.arange(w)
    observed_n = window_order("N", rank, group, conf, block)
    observed_s = window_order(strategy, rank, group, conf, block)
    diffs = {b: np.zeros(reps) for b in budgets}
    for r in range(reps):
        idx = np.concatenate(
            [
                rng.choice(pos, size=pos.size, replace=True) if pos.size else pos,
                rng.choice(neg, size=neg.size, replace=True) if neg.size else neg,
            ]
        )
        rk, gp, cf, bl, tr = rank[idx], group[idx], conf[idx], block[idx], truth[idx]
        order_n = window_order("N", rk, gp, cf, bl)
        order_s = window_order(strategy, rk, gp, cf, bl)
        cum_n = np.cumsum(tr[order_n].astype(int))
        cum_s = np.cumsum(tr[order_s].astype(int))
        for b in budgets:
            k = min(b, w) - 1
            diffs[b][r] = (cum_s[k] - cum_n[k]) if b < w else 0.0
    for b in budgets:
        k = min(b, w)
        obs = int(truth[observed_s[:k]].sum() - truth[observed_n[:k]].sum())
        lo, hi = np.percentile(diffs[b], [2.5, 97.5])
        out[b] = {
            "observed": float(obs if b < w else 0),
            "boot_mean": float(diffs[b].mean()),
            "lo": float(lo),
            "hi": float(hi),
        }
    return out


def window_arrays(
    recs: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Grupo (0/1/2), confiança e chamadas reais por par da janela."""
    group = np.array([GROUP_OF[r["summary"]["decision"]] for r in recs], dtype=int)
    conf = np.array([float(r["summary"]["confidence"]) for r in recs], dtype=float)
    calls = np.array([int(r.get("api_calls") or 0) for r in recs], dtype=int)
    return group, conf, calls


def rerank_results(
    nota_n: np.ndarray,
    truth_n: np.ndarray,
    recs: Sequence[Mapping[str, Any]],
    windows: Sequence[int],
    budgets: Sequence[int] = RERANK_BUDGETS,
    *,
    reps: int = BOOTSTRAP_REPS,
    seed: int = SEED,
) -> dict[str, Any]:
    """Todos os agregados do teste, a partir dos vetores na ordem N (puro)."""
    n = len(nota_n)
    max_b = max(budgets)
    out: dict[str, Any] = {
        "n_pool": n,
        "n_true": int(truth_n.sum()),
        "pool_true_total": int(truth_n.sum()),
        "budgets": [int(b) for b in budgets],
        "windows": {},
    }
    for w in windows:
        group, conf, calls = window_arrays(recs[:w])
        block = score_block(nota_n[:w])
        truth_w = truth_n[:w].astype(bool)
        tok_in = sum(int(r["tokens_prompt"]) for r in recs[:w])
        tok_out = sum(int(r["tokens_completion"]) for r in recs[:w])
        res: dict[str, Any] = {
            "llm_reviews": w,
            "api_calls": int(calls.sum()),
            "tokens_prompt": tok_in,
            "tokens_completion": tok_out,
            "true_in_window": int(truth_w.sum()),
            "true_by_decision": {
                name: int((truth_w & (group == g)).sum())
                for name, g in GROUP_OF.items()
            },
            "nontrue_by_decision": {
                name: int((~truth_w & (group == g)).sum())
                for name, g in GROUP_OF.items()
            },
            "arbiter": sum(r["summary"]["protocol"] == "arbiter" for r in recs[:w]),
            "strategies": {},
        }
        quart = np.minimum(4 * np.arange(w) // w, 3)
        res["true_by_quartile"] = [
            {
                name: int((truth_w & (quart == q) & (group == g)).sum())
                for name, g in GROUP_OF.items()
            }
            for q in range(4)
        ]
        base_curve = found_curve(np.arange(n), truth_n, max_b)
        for strat in STRATEGIES:
            order = full_order(strat, n, group, conf, block)
            curve = found_curve(order, truth_n, max_b)
            entry: dict[str, Any] = {
                "found": {int(b): int(curve[b - 1]) for b in budgets},
                "diff": {
                    int(b): int(curve[b - 1] - base_curve[b - 1]) for b in budgets
                },
                "curve": [
                    int(curve[b - 1]) for b in range(CURVE_STEP, max_b + 1, CURVE_STEP)
                ],
            }
            if strat != "N":
                entry["bootstrap"] = {
                    str(b): v
                    for b, v in bootstrap_diff(
                        strat,
                        truth_w,
                        group,
                        conf,
                        block,
                        BOOTSTRAP_BUDGETS,
                        reps=reps,
                        seed=seed,
                    ).items()
                }
            res["strategies"][strat] = entry
        out["windows"][str(w)] = res
    return out


def cmd_rerank_test(args: argparse.Namespace) -> int:
    ex = load_extract(args.input)
    cache_dir = Path(args.cache_dir)
    out_dir = cache_dir / "rerank"
    out_dir.mkdir(parents=True, exist_ok=True)
    pool = np.flatnonzero(ex.below)
    order = pool[base_order(ex.nota[pool], ex.pair_ids[pool])]  # índices no extrato
    windows = [w for w in RERANK_WINDOWS if w <= args.max_window]
    wmax = max(windows)
    ledger = BudgetLedger(out_dir / "budget.json", args.global_cap)
    reviewer = CachedParallelReviewer(
        cache_dir=cache_dir,
        api_key=load_api_key() if not args.no_api else "cache-only",
        model=args.model,
        max_api_reviews=0 if args.no_api else ledger.remaining,
        max_retries=args.max_retries,
        retry_delay_s=args.retry_delay,
    )
    before, t0 = len(reviewer), time.monotonic()
    status, note = "finished", None
    views = [make_view(ex.df.iloc[int(i)]) for i in order[:wmax]]
    if not args.no_api:
        chunk = max(1, 4 * args.workers)
        try:
            for start in range(0, len(views), chunk):
                reviewer.prefetch(
                    views[start : start + chunk], args.workers, max_errors=25
                )
                ledger.sync(reviewer.api_reviews)
            missing = [v for v in views if not reviewer.has_pair(v.pair_id)]
            if missing:  # uma segunda tentativa para os pares com erro
                reviewer.prefetch(missing, args.workers, max_errors=25)
        except ReviewStopped as exc:
            status, note = type(exc).__name__, str(exc)
        finally:
            ledger.sync(reviewer.api_reviews)
    elapsed = time.monotonic() - t0
    by_pair = reviewer.records_by_pair()
    recs: list[dict[str, Any]] = []
    for i in order[:wmax]:
        rec = by_pair.get(str(ex.pair_ids[int(i)]))
        if rec is None:
            break
        recs.append(rec)
    usable = [w for w in windows if w <= len(recs)]
    if not usable:
        raise SystemExit("nenhuma janela completa com veredito")
    nota_n, truth_n = ex.nota[order], ex.truth[order]
    results = rerank_results(
        nota_n, truth_n, recs, usable, reps=args.reps, seed=args.seed
    )
    results.update(
        {
            "status": status,
            "stop_note": note,
            "requested_windows": windows,
            "usable_windows": usable,
            "pool_true_total": int(truth_n.sum()),
        }
    )
    (out_dir / "results.json").write_text(json.dumps(results), encoding="utf-8")
    stats = reviewer.stats()
    if not args.no_api:
        _log_run(
            cache_dir,
            {
                "cmd": "rerank-test",
                "workers": args.workers,
                "cached_before": before,
                "elapsed_s": round(elapsed, 1),
                "status": status,
                "api_reviews": stats["api_reviews"],
                "errors": stats["errors"],
                "http_errors": stats["http_errors"],
                "model": args.model,
            },
        )
    md = render_rerank_markdown(
        results,
        _ledger_by_cmd(cache_dir).get("rerank-test", {}),
        ledger.used,
        ledger.cap,
    )
    Path(args.out_md).write_text(md, encoding="utf-8")
    pd.DataFrame(rerank_csv_rows(results)).to_csv(args.out_csv, index=False)
    print(
        f"rerank-test: {status}; janelas {usable}; API nesta execução "
        f"{stats['api_reviews']} (teto {ledger.used}/{ledger.cap}); erros "
        f"{stats['errors']}, HTTP {stats['http_errors']}; {elapsed:.0f}s"
    )
    if note:
        print(f"parada: {note}")
    return 0


def rerank_csv_rows(results: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(table: str, strat: str, w: Any, b: Any, metric: str, value: Any) -> None:
        rows.append(
            {
                "tabela": table,
                "estrategia": strat,
                "janela": w,
                "B": b,
                "metrica": metric,
                "valor": value,
            }
        )

    for w, res in results["windows"].items():
        for key in ("llm_reviews", "api_calls", "tokens_prompt", "tokens_completion"):
            add("janela", "", w, "", key, res[key])
        add("janela", "", w, "", "true_in_window", res["true_in_window"])
        add("janela", "", w, "", "arbiter", res["arbiter"])
        for name, v in res["true_by_decision"].items():
            add("janela", "", w, "", f"true_llm_{name}", v)
        for name, v in res["nontrue_by_decision"].items():
            add("janela", "", w, "", f"nontrue_llm_{name}", v)
        for q, d in enumerate(res["true_by_quartile"], 1):
            for name, v in d.items():
                add("janela_quartil", "", w, f"Q{q}", f"true_llm_{name}", v)
        for strat, e in res["strategies"].items():
            for b, v in e["found"].items():
                add("encontrados", strat, w, b, "found", v)
                add("encontrados", strat, w, b, "diff_vs_N", e["diff"][b])
                add(
                    "encontrados",
                    strat,
                    w,
                    b,
                    "decisoes_por_verdadeiro",
                    round(int(b) / v, 3) if v else None,
                )
                extra = e["diff"][b]
                add(
                    "custo",
                    strat,
                    w,
                    b,
                    "chamadas_por_par_extra",
                    round(res["api_calls"] / extra, 2)
                    if extra > 0 and strat != "N"
                    else None,
                )
            for i, v in enumerate(e["curve"], 1):
                add("curva", strat, w, i * CURVE_STEP, "found", v)
            for b, v in e.get("bootstrap", {}).items():
                for k2, v2 in v.items():
                    add("bootstrap", strat, w, b, k2, round(v2, 4))
    return rows


def render_rerank_markdown(
    results: Mapping[str, Any],
    ledger: Mapping[str, Any],
    used: int,
    cap: int,
) -> str:
    wins = results["windows"]
    budgets = tuple(results["budgets"])
    total = results["pool_true_total"]
    out: list[str] = []
    w_ = out.append
    w_("# O LLM como desempate na zona cinza (abaixo do corte)\n")
    w_(
        "> **Aviso:** os vereditos do LLM são **uma única realização** (temperatura "
        "padrão); humano simulado **perfeito** (decide pelo PAR todo par que "
        "revisa); o experimento **não mede tempo por decisão**; exploratório, com "
        f"apenas {total} verdadeiros. Apenas agregados; nada em nível de par.\n"
    )
    w_("## Desenho\n")
    w_(
        f"- Pool: os {_n(results['n_pool'])} pares abaixo do corte, {total} "
        "verdadeiros (PAR ∈ {1, 2}, usado só como verdade). Ordem base **N**: nota "
        "decrescente, empate por `pair_id`.\n"
        "- Janela W: os W primeiros de N recebem veredito do Haiku (prompt v4, sem "
        "`model_outputs` e sem `features.nota_final`). O LLM não decide nada e "
        "nenhum par é descartado: cada estratégia é uma permutação do pool e o "
        "humano decide cada par que revisa.\n"
        "- **N**: ordem da nota. **L1**: na janela, MATCH, depois UNSURE, depois "
        "NONMATCH, cada grupo na ordem da nota; depois o resto em N. **L2**: blocos "
        f"de nota de largura {str(BLOCK_WIDTH).replace('.', ',')} (decrescentes); "
        "em cada bloco, MATCH → UNSURE → NONMATCH na ordem da nota; depois o "
        "resto em N. **L3**: como L1, "
        "mas os MATCH por confiança do LLM decrescente e depois pela nota.\n"
        f"- Janelas: {', '.join(str(w) for w in wins)}; orçamentos de decisões "
        f"humanas B: {', '.join(str(b) for b in budgets)}. Com B ≥ W, L e N "
        "revisam o mesmo conjunto (diferença 0 por construção).\n"
        "- Incerteza: bootstrap pareado da diferença L − N "
        f"({BOOTSTRAP_REPS} réplicas, semente {SEED}), reamostrando os pares da "
        "janela com reposição, estratificado por verdade; IC percentil de 95%. "
        "Com 36 positivos, diferenças de 1–3 pares são ruído.\n"
    )
    w_("## O LLM na janela\n")
    w_(
        "| W | verdadeiros na janela | LLM MATCH / UNSURE / NONMATCH entre os "
        "verdadeiros | não pares com MATCH (FP) | não pares com UNSURE "
        "| MATCH totais | árbitro |"
    )
    w_("|--:|--:|---|--:|--:|--:|--:|")
    for w, r in wins.items():
        td, nd = r["true_by_decision"], r["nontrue_by_decision"]
        w_(
            f"| {w} | {r['true_in_window']} | {td['MATCH']} / {td['UNSURE']} / "
            f"{td['NONMATCH']} | {nd['MATCH']} | {nd['UNSURE']} "
            f"| {td['MATCH'] + nd['MATCH']} | {r['arbiter']} |"
        )
    w_("")
    w_("Verdadeiros por quartil de posição na janela (MATCH / UNSURE / NONMATCH):\n")
    w_("| W | Q1 (topo) | Q2 | Q3 | Q4 |")
    w_("|--:|---|---|---|---|")
    for w, r in wins.items():
        cells = [
            f"{d['MATCH']} / {d['UNSURE']} / {d['NONMATCH']}"
            for d in r["true_by_quartile"]
        ]
        w_(f"| {w} | " + " | ".join(cells) + " |")
    w_("")
    w_(f"## Verdadeiros abaixo do corte encontrados (de {total}) a B decisões\n")
    for w, r in wins.items():
        w_(f"**W = {w}** (valor; diferença contra N entre parênteses)\n")
        w_("| estratégia | " + " | ".join(f"B={b}" for b in budgets) + " |")
        w_("|---|" + "--:|" * len(budgets))
        for strat, e in r["strategies"].items():
            cells = []
            for b in budgets:
                d = e["diff"][b]
                cells.append(
                    f"{e['found'][b]}" if strat == "N" else f"{e['found'][b]} ({d:+d})"
                )
            w_(f"| {strat} | " + " | ".join(cells) + " |")
        w_("")
    w_("Decisões humanas por verdadeiro encontrado (B / achados):\n")
    w_("| W | estratégia | " + " | ".join(f"B={b}" for b in budgets) + " |")
    w_("|--:|---|" + "--:|" * len(budgets))
    for w, r in wins.items():
        for strat, e in r["strategies"].items():
            cells = [
                _f(b / e["found"][b], 1) if e["found"][b] else "n/d" for b in budgets
            ]
            w_(f"| {w} | {strat} | " + " | ".join(cells) + " |")
    w_("")
    w_("## Bootstrap pareado da diferença L − N\n")
    w_("| W | estratégia | " + " | ".join(f"B={b}" for b in BOOTSTRAP_BUDGETS) + " |")
    w_("|--:|---|" + "---|" * len(BOOTSTRAP_BUDGETS))
    for w, r in wins.items():
        for strat, e in r["strategies"].items():
            if strat == "N":
                continue
            cells = []
            for b in BOOTSTRAP_BUDGETS:
                v = e["bootstrap"][str(b)]
                cells.append(f"{v['observed']:+.0f} [{v['lo']:+.0f}; {v['hi']:+.0f}]")
            w_(f"| {w} | {strat} | " + " | ".join(cells) + " |")
    w_(
        "\nCada célula: diferença observada [IC 95% do bootstrap]. A curva completa "
        f"(achados × B, de {CURVE_STEP} em {CURVE_STEP}) está no CSV, tabela `curva`.\n"
    )
    w_("## Custo\n")
    w_(
        "| W | revisões do LLM | chamadas à API | tokens de entrada | tokens de saída "
        "| chamadas por par extra (L1 / L2 / L3) em B=100 e B=400 |"
    )
    w_("|--:|--:|--:|--:|--:|---|")
    for w, r in wins.items():
        cells = []
        for b in (100, 400):
            vals = []
            for strat in ("L1", "L2", "L3"):
                d = r["strategies"][strat]["diff"][b]
                vals.append(f"{r['api_calls'] / d:.0f}" if d > 0 else "n/d")
            cells.append(f"B={b}: " + " / ".join(vals))
        w_(
            f"| {w} | {_n(r['llm_reviews'])} | {_n(r['api_calls'])} "
            f"| {_n(r['tokens_prompt'])} | {_n(r['tokens_completion'])} "
            f"| {'; '.join(cells)} |"
        )
    w_(
        "\n`n/d` = a estratégia não achou mais verdadeiros que N nesse B. As "
        "chamadas contam todas as feitas para a janela (incluindo as reaproveitadas "
        "do cache de execuções anteriores, que foram pagas antes).\n"
    )
    w_(
        f"Revisões novas na API neste teste: {_n(ledger.get('api'))} (teto {cap}; "
        f"usadas {used}); tempo de relógio {_n(round(ledger.get('elapsed_s', 0)))} s; "
        f"workers {', '.join(str(x) for x in sorted(ledger.get('workers', [])))}; "
        f"HTTP 429/erros: {dict(ledger.get('http', {})) or 'nenhum'}.\n"
    )
    w_("## Discussão\n")
    for line in rerank_discussion(results):
        w_(line)
    w_("")
    return "\n".join(out) + "\n"


def rerank_discussion(results: Mapping[str, Any]) -> list[str]:
    """Respostas à pergunta, só com números das tabelas."""
    wins = results["windows"]
    total = results["pool_true_total"]
    lines: list[str] = []
    excl = 0
    cells = 0
    best: tuple[int, str, str, int] | None = None
    for w, r in wins.items():
        td = r["true_by_decision"]
        lines.append(
            f"- W={w}: {r['true_in_window']} dos {total} verdadeiros estão na "
            f"janela; o LLM marcou {td['MATCH']} como MATCH, {td['UNSURE']} como "
            f"UNSURE e {td['NONMATCH']} como NONMATCH, e deu MATCH a "
            f"{r['nontrue_by_decision']['MATCH']} não pares."
        )
        for strat in ("L1", "L2", "L3"):
            e = r["strategies"][strat]
            diffs = {b: e["diff"][b] for b in (50, 100, 200, 400)}
            for b in BOOTSTRAP_BUDGETS:
                v = e["bootstrap"][str(b)]
                if b < int(w):
                    cells += 1
                    if v["lo"] > 0 or v["hi"] < 0:
                        excl += 1
                if best is None or e["diff"][b] > best[0]:
                    best = (e["diff"][b], strat, w, b)
            lines.append(
                f"  - {strat}: diferença contra N em B=50/100/200/400 = "
                + "/".join(f"{diffs[b]:+d}" for b in (50, 100, 200, 400))
                + "."
            )
    lines.append("- Faixa das diferenças contra N entre as janelas (mín a máx):")
    for strat in ("L1", "L2", "L3"):
        parts = []
        for b in (x for x in (25, 50, 100, 200, 400, 800) if x in results["budgets"]):
            vals = [r["strategies"][strat]["diff"][b] for r in wins.values()]
            parts.append(f"B={b}: {min(vals):+d} a {max(vals):+d}")
        lines.append(f"  - {strat}: " + "; ".join(parts) + ".")
    same = sum(
        r["strategies"]["L1"]["found"][b] == r["strategies"]["L3"]["found"][b]
        for r in wins.values()
        for b in results["budgets"]
    )
    lines.append(
        f"- L1 e L3 acharam o mesmo número de verdadeiros em {same} de "
        f"{len(wins) * len(results['budgets'])} células (janela × B)."
    )
    costs = [
        (r["api_calls"] / e["diff"][b], strat, w, b)
        for w, r in wins.items()
        for strat, e in r["strategies"].items()
        for b in (50, 100, 200, 400)
        if strat != "N" and e["diff"][b] > 0
    ]
    if costs:
        c = min(costs)
        lines.append(
            f"- Menor custo por par extra entre B=50..400: {c[0]:.0f} chamadas "
            f"({c[1]}, W={c[2]}, B={c[3]})."
        )
    lines.append(
        f"- Em {excl} de {cells} combinações (estratégia, W, B ∈ {{50,100,200,400}} "
        "com B < W) o IC 95% do bootstrap exclui zero."
    )
    if best is not None:
        lines.append(
            f"- Maior diferença observada: {best[0]:+d} verdadeiros ({best[1]}, "
            f"W={best[2]}, B={best[3]}), em {total} positivos: diferenças de 1–3 "
            "pares são ruído."
        )
    lines.append(
        "- Custo (chamadas à API por janela): "
        + ", ".join(f"W={w}: {_n(r['api_calls'])} chamadas" for w, r in wins.items())
        + "; o humano decide o mesmo número B de pares em todas as estratégias."
    )
    return lines


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--input", default=str(DEFAULT_INPUT))
        sp.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
        sp.add_argument("--model", default=DEFAULT_MODEL)
        sp.add_argument("--seed", type=int, default=SEED)
        sp.add_argument("--per-band", type=int, default=100)

    pi = sub.add_parser("pilot", help="piloto com poucos pares")
    common(pi)
    pi.add_argument("--n", type=int, default=20)
    pi.add_argument("--workers", type=int, default=8)
    pi.add_argument("--max-api-reviews", type=int, default=60)
    pi.add_argument("--max-errors", type=int, default=5)
    pi.set_defaults(func=cmd_pilot)

    ca = sub.add_parser("calibrate", help="amostra de calibração completa")
    common(ca)
    ca.add_argument("--workers", type=int, default=24)
    ca.add_argument("--max-api-reviews", type=int, default=1200)
    ca.add_argument("--max-errors", type=int, default=25)
    ca.set_defaults(func=cmd_calibrate)

    re = sub.add_parser("report", help="relatório agregado a partir do cache")
    common(re)
    re.add_argument(
        "--out-md", default=str(ROOT / "docs" / "llm_real_haiku_sem_nota.md")
    )
    re.add_argument(
        "--out-csv", default=str(ROOT / "docs" / "llm_real_haiku_sem_nota.csv")
    )
    re.set_defaults(func=cmd_report)

    qr = sub.add_parser("queue-run", help="fila ponta a ponta com o LLM real")
    common(qr)
    qr.add_argument(
        "--negative-labels", choices=("llm_nonmatch", "verified_only"), required=True
    )
    qr.add_argument("--workers", type=int, default=24)
    qr.add_argument("--k-hot", type=int, default=24)
    qr.add_argument("--k-cold", type=int, default=4)
    qr.add_argument("--k-low", type=int, default=2)
    qr.add_argument("--max-retries", type=int, default=6)
    qr.add_argument("--retry-delay", type=float, default=4.0)
    qr.add_argument("--k-explore", type=int, default=6)
    qr.add_argument("--max-minutes", type=float, default=None)
    qr.add_argument("--global-cap", type=int, default=GLOBAL_CAP)
    qr.set_defaults(func=cmd_queue_run)

    bl = sub.add_parser("baseline", help="bases H1 (LLM real) e H0 sobre a nota")
    common(bl)
    bl.add_argument("--n-llm", type=int, default=0)
    bl.add_argument("--workers", type=int, default=24)
    bl.add_argument("--max-retries", type=int, default=6)
    bl.add_argument("--retry-delay", type=float, default=4.0)
    bl.add_argument("--max-minutes", type=float, default=None)
    bl.add_argument("--global-cap", type=int, default=GLOBAL_CAP)
    bl.set_defaults(func=cmd_baseline)

    ef = sub.add_parser("export-fixture", help="exporta as decisões do cache")
    common(ef)
    ef.add_argument("--fixture-dir", default=str(FIXTURE_DIR))
    ef.set_defaults(func=cmd_export_fixture)

    qp = sub.add_parser("queue-report", help="relatório da fila ponta a ponta")
    common(qp)
    qp.add_argument(
        "--out-md", default=str(ROOT / "docs" / "llm_real_fila_ponta_a_ponta.md")
    )
    qp.add_argument(
        "--out-csv", default=str(ROOT / "docs" / "llm_real_fila_ponta_a_ponta.csv")
    )
    qp.set_defaults(func=cmd_queue_report)

    rr = sub.add_parser("rerank-test", help="LLM como desempate abaixo do corte")
    common(rr)
    rr.add_argument("--workers", type=int, default=24)
    rr.add_argument("--max-retries", type=int, default=6)
    rr.add_argument("--retry-delay", type=float, default=4.0)
    rr.add_argument("--max-window", type=int, default=max(RERANK_WINDOWS))
    rr.add_argument("--reps", type=int, default=BOOTSTRAP_REPS)
    rr.add_argument("--global-cap", type=int, default=RERANK_CAP)
    rr.add_argument("--no-api", action="store_true", help="só o cache, sem chamadas")
    rr.add_argument(
        "--out-md", default=str(ROOT / "docs" / "llm_real_desempate_zona_cinza.md")
    )
    rr.add_argument(
        "--out-csv", default=str(ROOT / "docs" / "llm_real_desempate_zona_cinza.csv")
    )
    rr.set_defaults(func=cmd_rerank_test)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except MissingApiKeyError as exc:
        print(f"erro: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
