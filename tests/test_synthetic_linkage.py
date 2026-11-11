from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import synthetic_linkage as sl
from sklearn.metrics import roc_auc_score

from gzcmd_record_linkage.llm_dossier import build_dossier
from gzcmd_record_linkage.loader import LoadConfig, load_comparador_csv


@pytest.fixture(scope="module")
def base() -> pd.DataFrame:
    return sl.gerar_base()


def test_vinte_e_nove_subescores_distintos():
    assert len(sl.SUBESCORES) == 29
    assert len(set(sl.SUBESCORES)) == 29


def test_determinismo_por_semente():
    primeira = sl.gerar_base(n_obitos=400, seed=7)
    segunda = sl.gerar_base(n_obitos=400, seed=7)
    outra = sl.gerar_base(n_obitos=400, seed=8)
    pd.testing.assert_frame_equal(primeira, segunda)
    assert not primeira.equals(outra)


def test_contrato_de_colunas(base):
    esperadas = {
        "REFREC",
        "COMPREC",
        "PASSO",
        *sl.SUBESCORES,
        "nota final",
        "escore_orl",
        "ponto_de_corte",
        "acima_corte",
        "PAR",
        *sl.COLUNAS_LATENTES,
    }
    assert set(base.columns) == esperadas
    assert set(base["PASSO"]) == set(sl.PASSOS)
    subescores = base[list(sl.SUBESCORES)]
    assert ((subescores >= 0.0) & (subescores <= 1.0)).all().all()
    assert base["nota final"].between(0.0, 10.0).all()
    numericas = [*sl.SUBESCORES, "nota final", "escore_orl"]
    assert not base[numericas].isna().any().any()
    # Só códigos: nenhum texto livre que lembre dado pessoal.
    assert base["REFREC"].str.fullmatch(r"R\d{6}").all()
    assert base["COMPREC"].str.fullmatch(r"C\d{7}").all()
    assert set(base["tipo_par"]) <= set(sl.TIPOS_PAR)


def test_carregador_e_dossie_aceitam_a_base(tmp_path):
    base = sl.gerar_base(n_obitos=300, seed=5)
    caminho = tmp_path / "base_sintetica.csv"
    base.to_csv(caminho, sep=";", decimal=",", index=False, encoding="utf-8")
    carregada = load_comparador_csv(caminho, cfg=LoadConfig())
    assert len(carregada) == len(base)
    esperado = base["PAR"].isin([1, 2]).astype(int).to_numpy()
    np.testing.assert_array_equal(carregada["TARGET"].to_numpy(), esperado)
    for coluna in [*sl.SUBESCORES, "nota final"]:
        assert pd.api.types.is_numeric_dtype(carregada[coluna])
    np.testing.assert_allclose(carregada["nota final"], base["nota final"])
    dossie = build_dossier(carregada.iloc[0])
    assert set(dossie.features.name) == {
        "NOME qtd frag iguais",
        "NOME prim frag igual",
        "NOME ult frag igual",
    }
    assert len(dossie.features.address) == 6
    assert len(dossie.pair_id) == 16


def test_proporcoes_na_ordem_de_grandeza_da_tese(base):
    assert 20_000 <= len(base) <= 40_000
    assert base["PASSO"].value_counts(normalize=True)[11] > 0.6
    verdadeiros = base["PAR"].isin([1, 2])
    taxa_por_passo = verdadeiros.groupby(base["PASSO"]).mean()
    assert taxa_por_passo[11] < taxa_por_passo[9]
    n_par1 = int(base["PAR"].eq(1).sum())
    n_par2 = int(base["PAR"].eq(2).sum())
    assert 0.70 <= n_par1 / (n_par1 + n_par2) <= 0.95
    assert 15 <= n_par2 <= 80
    assert (~base["acima_corte"]).mean() > 0.95
    # A maior parte das notas fica abaixo de 5 (65% no extrato da tese).
    assert 0.40 < base["nota final"].lt(5).mean() < 0.90


def test_semantica_da_referencia_operacional(base):
    acima = base["acima_corte"]
    assert acima.equals(base["escore_orl"] >= base["ponto_de_corte"])
    corte = base["PASSO"].map(sl.ParametrosBase().corte_por_passo)
    np.testing.assert_allclose(base["ponto_de_corte"], corte)
    assert set(base["PAR"]) <= {0, 1, 2}
    # Como na tese: o corte define PAR = 1, inclusive para não pares acima dele.
    assert base["PAR"].eq(1).equals(acima)
    assert not base.loc[base["PAR"].eq(2), "acima_corte"].any()
    assert base.loc[base["PAR"].eq(2), "par_verdadeiro"].eq(1).all()
    # A revisão manual de origem deixa pares verdadeiros abaixo do corte sem PAR.
    perdidos = ~acima & base["par_verdadeiro"].eq(1) & base["PAR"].eq(0)
    assert perdidos.any()


def test_no_maximo_um_par_verdadeiro_por_registro(base):
    assert not base.duplicated(["REFREC", "COMPREC"]).any()
    verdadeiros = base.loc[base["par_verdadeiro"].eq(1)]
    assert verdadeiros["REFREC"].is_unique
    assert verdadeiros["COMPREC"].is_unique
    assert base.groupby("REFREC").size().max() > 1


def test_escore_e_nota_informativos(base):
    verdade = base["par_verdadeiro"]
    assert roc_auc_score(verdade, base["escore_orl"]) > 0.85
    assert roc_auc_score(verdade, base["nota final"]) > 0.85


def test_negativos_dificeis_e_ausencias(base):
    homonimos = base.loc[base["tipo_par"].eq("homonimo")]
    assert len(homonimos) > 0
    assert homonimos["NOME prim frag igual"].mean() > 0.9
    familiares = base.loc[base["tipo_par"].eq("familiar")]
    assert familiares["NOMEMAE prim frag igual"].mean() > 0.5
    # Mãe ausente vira zero em todos os subescores do nome da mãe.
    assert base["NOMEMAE qtd frag iguais"].eq(0).mean() > 0.10


def test_registros_precarios_concentrados_no_passo_11_e_pobres_no_cdr(base):
    assert base.loc[base["par_verdadeiro"].eq(0), "registro_precario"].eq(0).all()
    pares = base.loc[base["par_verdadeiro"].eq(1)]
    por_passo = pares.groupby("PASSO")["registro_precario"].mean()
    assert por_passo[11] > por_passo[9]
    precarios = pares["registro_precario"].eq(1)
    assert precarios.any()
    # O CdR vê pior os registros precários (ausências viram zero).
    nota_precarios = pares.loc[precarios, "nota final"].mean()
    nota_bons = pares.loc[~precarios, "nota final"].mean()
    assert nota_precarios < nota_bons
    assert pares.loc[precarios, "NOMEMAE qtd frag iguais"].eq(0).mean() > 0.5


def test_revisao_manual_de_origem_favorece_escore_alto():
    probs = sl.prob_revisao_manual(np.array([-10.0, -5.0, -1.0]))
    assert np.all(np.diff(probs) > 0)
    grande = sl.gerar_base(n_obitos=20_000, seed=11)
    alvo = grande.loc[~grande["acima_corte"] & grande["par_verdadeiro"].eq(1)]
    distancia = alvo["escore_orl"] - alvo["ponto_de_corte"]
    recuperados = alvo["PAR"].eq(2)
    assert distancia[recuperados].mean() > distancia[~recuperados].mean()


def test_sem_colunas_latentes(base):
    observada = sl.sem_colunas_latentes(base)
    assert not set(sl.COLUNAS_LATENTES) & set(observada.columns)
    assert "PAR" in observada.columns


def test_parametros_invalidos():
    with pytest.raises(ValueError):
        sl.ParametrosBase(prop_homonimos=1.5)
    with pytest.raises(ValueError):
        sl.ParametrosBase(ponto_de_corte=(12.0, 13.0))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        sl.ParametrosBase(prop_precarios=(0.1, 0.1, 1.2))
    with pytest.raises(ValueError):
        sl.gerar_base(n_obitos=0)
