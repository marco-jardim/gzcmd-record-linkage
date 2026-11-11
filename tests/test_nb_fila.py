from __future__ import annotations

import nb_fila as nf
import numpy as np
import pandas as pd
import pytest
import synthetic_linkage as sl

from gzcmd_record_linkage.ranking import review_queue, rrf


@pytest.fixture(scope="module")
def base() -> pd.DataFrame:
    return sl.gerar_base(n_obitos=1_500, seed=3)


@pytest.fixture(scope="module")
def abaixo(base) -> pd.DataFrame:
    # Mantém o índice original (não contíguo) para exercitar o alinhamento.
    selecao = base.loc[~base["acima_corte"]]
    assert selecao["PAR"].eq(2).sum() >= 2, "base de teste sem PAR == 2 suficiente"
    return selecao


def _prob_didatica(df: pd.DataFrame) -> np.ndarray:
    return np.clip((df["nota final"].to_numpy() - 4.0) / 6.0, 0.0, 1.0)


def test_atributos_sao_os_30_do_extrato(base):
    x = nf.atributos(base)
    assert list(x.columns) == [*sl.SUBESCORES, "nota final"]
    assert x.shape == (len(base), 30)
    proibidas = {"REFREC", "COMPREC", "PASSO", "PAR", "escore_orl", "par_verdadeiro"}
    assert not proibidas & set(x.columns)


def test_particoes_aninhadas_sem_vazamento_de_obito(base):
    y = nf.rotulo(base)
    grupos = base["REFREC"].to_numpy()
    particoes = nf.particoes_aninhadas(y, grupos, seed=42)
    assert len(particoes) == 5
    testes = np.concatenate([parte.teste for parte in particoes])
    np.testing.assert_array_equal(np.sort(testes), np.arange(len(base)))
    for parte in particoes:
        assert np.intersect1d(grupos[parte.treino], grupos[parte.teste]).size == 0
        assert len(parte.internas) == 5
        previstos = np.concatenate([previsao for _, previsao in parte.internas])
        np.testing.assert_array_equal(np.sort(previstos), np.sort(parte.treino))
        for ajuste, previsao in parte.internas:
            assert np.isin(ajuste, parte.treino).all()
            assert np.intersect1d(grupos[ajuste], grupos[previsao]).size == 0


def test_prob_calibrada_oof_agrupada_sem_vazamento_e_reprodutivel(base):
    registros = []
    p = nf.prob_calibrada_oof(
        base,
        n_estimators=10,
        n_jobs=1,
        auditoria=lambda *args: registros.append(args),
    )
    assert p.index.equals(base.index)
    assert p.between(0.0, 1.0).all()
    assert len(registros) == 5 * 5 + 5
    grupos = base["REFREC"].to_numpy()
    for _, _, ajuste, previsao in registros:
        assert np.intersect1d(grupos[ajuste], grupos[previsao]).size == 0
    de_novo = nf.prob_calibrada_oof(base, n_estimators=10, n_jobs=1)
    pd.testing.assert_series_equal(p, de_novo)


def test_prob_por_linha_mistura_obitos_entre_treino_e_teste(base):
    registros = []
    p = nf.prob_por_linha(
        base,
        n_estimators=10,
        n_jobs=1,
        auditoria=lambda *args: registros.append(args),
    )
    assert p.between(0.0, 1.0).all()
    grupos = base["REFREC"].to_numpy()
    externas = [registro for registro in registros if registro[1] == "externa"]
    assert len(externas) == 5
    assert any(
        np.intersect1d(grupos[treino], grupos[teste]).size > 0
        for _, _, treino, teste in externas
    )


def test_curva_de_esforco_em_fila_de_brinquedo():
    curva = nf.curva_esforco([2, 0, 3, 1], [1, 0, 1, 0])
    assert curva.tolist() == [1, 2, 2, 2]
    assert curva.index.tolist() == [1, 2, 3, 4]
    assert nf.revisoes_para_meta(curva, 1) == 1
    assert nf.revisoes_para_meta(curva, 2) == 2
    assert nf.recuperados_com_carga(curva, 0) == 0
    assert nf.recuperados_com_carga(curva, 1) == 1
    assert nf.recuperados_com_carga(curva, 3) == 2
    assert nf.recuperados_com_carga(curva, 10) == 2
    with pytest.raises(ValueError):
        nf.revisoes_para_meta(curva, 3)
    with pytest.raises(ValueError):
        nf.curva_esforco([0, 0, 1, 2], [1, 0, 1, 0])


def test_curvas_monotonas_e_completas(abaixo):
    ordens = nf.ordenacoes(abaixo, _prob_didatica(abaixo))
    assert set(ordens) == {"O1", "O2", "O3", "F4"}
    y = nf.rotulo(abaixo)
    for ordem in ordens.values():
        curva = nf.curva_esforco(ordem, y)
        assert (np.diff(curva.to_numpy()) >= 0).all()
        assert curva.iloc[-1] == y.sum()


def test_f4_igual_ao_rrf_do_pacote(abaixo):
    p = np.random.default_rng(0).random(len(abaixo))
    ordens = nf.ordenacoes(abaixo, p)
    quadro = abaixo.assign(p_cal=p, posicao=np.arange(len(abaixo)))
    fila = review_queue(
        quadro, scores=["p_cal", "escore_orl"], method="rrf", k=60, seed=42
    )
    np.testing.assert_array_equal(ordens["F4"], fila["posicao"].to_numpy())
    escore = abaixo["escore_orl"].to_numpy()
    np.testing.assert_array_equal(ordens["O1"], rrf([escore], input_kind="scores"))
    np.testing.assert_array_equal(ordens["O3"], rrf([p], input_kind="scores"))


def test_ordenacoes_alinham_series_da_base_inteira(base, abaixo):
    serie = pd.Series(np.random.default_rng(1).random(len(base)), index=base.index)
    ordens = nf.ordenacoes(abaixo, serie)
    esperado = rrf([serie.loc[abaixo.index].to_numpy()], input_kind="scores")
    np.testing.assert_array_equal(ordens["O3"], esperado)
    with pytest.raises(ValueError):
        nf.ordenacoes(abaixo, np.zeros(len(abaixo) + 1))


def test_regra_hibrida_da_meta():
    assert nf.meta_da_replica(9, 36, 36) == 9
    assert nf.meta_da_replica(9, 36, 50) == 9
    assert nf.meta_da_replica(9, 36, 20) == 5
    assert nf.meta_da_replica(23, 36, 20) == 13
    assert nf.meta_da_replica(36, 36, 20) == 20


def test_bootstrap_reprodutivel_e_coerente(abaixo):
    ordens = nf.ordenacoes(abaixo, np.random.default_rng(0).random(len(abaixo)))
    meta = int(nf.rotulo(abaixo).sum()) // 2
    a = nf.bootstrap_diferenca(abaixo, ordens["O3"], ordens["O1"], meta, n=200)
    b = nf.bootstrap_diferenca(abaixo, ordens["O3"], ordens["O1"], meta, n=200)
    pd.testing.assert_series_equal(a, b)
    mediana = a["diferença mediana (A menos B)"]
    assert a["IC95 inferior"] <= mediana <= a["IC95 superior"]
    assert 0.0 <= a["proporção em que A exige menos"] <= 1.0
    assert 0 < a["réplicas válidas"] <= 200
    igual = nf.bootstrap_diferenca(abaixo, ordens["O1"], ordens["O1"], meta, n=50)
    assert igual["diferença mediana (A menos B)"] == 0
    assert igual["proporção em que A exige menos"] == 0
    with pytest.raises(ValueError):
        nf.bootstrap_diferenca(abaixo, ordens["O3"], ordens["O1"], 10_000, n=10)


def test_revisor_llm_simulado_segue_as_taxas():
    quadro = pd.DataFrame({"band": ["grey_mid"] * 6, "PAR": [2, 2, 2, 0, 0, 0]})
    perfeito = {"grey_mid": {"e_fp": 0.0, "e_fn": 0.0, "e_unsure": 0.0}}
    assert nf.revisor_llm_simulado(quadro, perfeito).tolist() == (
        ["MATCH"] * 3 + ["NONMATCH"] * 3
    )
    sempre_erra = {"grey_mid": {"e_fp": 1.0, "e_fn": 1.0, "e_unsure": 0.0}}
    assert nf.revisor_llm_simulado(quadro, sempre_erra).tolist() == (
        ["NONMATCH"] * 3 + ["MATCH"] * 3
    )
    indeciso = {"grey_mid": {"e_fp": 0.0, "e_fn": 0.0, "e_unsure": 1.0}}
    assert set(nf.revisor_llm_simulado(quadro, indeciso)) == {"UNSURE"}
    pd.testing.assert_series_equal(
        nf.revisor_llm_simulado(quadro, seed=5), nf.revisor_llm_simulado(quadro, seed=5)
    )
    with pytest.raises(ValueError):
        nf.revisor_llm_simulado(quadro.assign(band="faixa_inexistente"))


def test_revisor_llm_simulado_reproduz_as_taxas_em_media():
    n = 20_000
    quadro = pd.DataFrame({"band": ["grey_high"] * n, "PAR": [0] * n})
    respostas = nf.revisor_llm_simulado(quadro, seed=3)
    taxa = nf.TAXAS_LLM_HAIKU_V4["vigilancia"]["grey_high"]
    assert respostas.eq("MATCH").mean() == pytest.approx(taxa["e_fp"], abs=0.01)
    assert respostas.eq("UNSURE").mean() == pytest.approx(taxa["e_unsure"], abs=0.01)


def test_taxas_do_revisor_de_referencia_e_reserva():
    vig = nf.TAXAS_LLM_HAIKU_V4["vigilancia"]
    conf = nf.TAXAS_LLM_HAIKU_V4["confirmacao"]
    faixas = {"low", "grey_low", "grey_mid", "grey_high", "near_high", "high"}
    assert set(vig) == set(conf) == faixas
    decisoes = {
        modo: sum(linha[0] for linha in contagens.values())
        for modo, contagens in nf.CONTAGENS_LLM_HAIKU_V4.items()
    }
    assert decisoes == {"vigilancia": 1774, "confirmacao": 459}
    assert vig["grey_high"]["e_fp"] == pytest.approx(26 / 920)
    assert vig["grey_high"]["e_fn"] == pytest.approx(4 / 43)
    assert vig["grey_high"]["e_unsure"] == pytest.approx(44 / 963)
    assert conf["near_high"]["e_fn"] == pytest.approx(8 / 34)
    # Reserva: denominador vazio volta às proporções fixas do YAML do pacote, e
    # UNSURE sem decisões na faixa usa a proporção do modo.
    assert vig["high"]["e_fp"] == pytest.approx(0.01)
    assert conf["grey_mid"]["e_fp"] == pytest.approx(0.06)
    assert conf["grey_mid"]["e_unsure"] == pytest.approx(26 / 459)
    assert vig["low"]["e_unsure"] == pytest.approx(96 / 1774)


@pytest.mark.parametrize("modo", ["vigilancia", "confirmacao"])
def test_politica_v31_retorna_so_as_quatro_acoes(abaixo, modo):
    decisoes, resumo = nf.politica_v31(abaixo, _prob_didatica(abaixo), modo)
    assert decisoes.index.equals(abaixo.index)
    assert set(decisoes["action"]) <= set(nf.ACOES_V31)
    contagens = [
        "aceitos automaticamente (MATCH)",
        "rejeitados automaticamente (NONMATCH)",
        "enviados ao humano (DEFERRED)",
        "enviados ao LLM (LLM_REVIEW)",
    ]
    assert sum(int(resumo[chave]) for chave in contagens) == len(abaixo)
    assert resumo["enviados ao LLM (LLM_REVIEW)"] <= resumo["orçamento do LLM"]
    no_llm = decisoes["action"].eq("LLM_REVIEW")
    respostas = decisoes.loc[no_llm, "resposta_llm"]
    assert respostas.isin(["MATCH", "NONMATCH", "UNSURE"]).all()
    assert decisoes.loc[~no_llm, "resposta_llm"].isna().all()
    baixa = decisoes["nota final"] < 5
    assert decisoes.loc[baixa, "action"].eq("NONMATCH").all()
    assert not decisoes["destino"].eq("sem destino").any()
    with pytest.raises(ValueError):
        nf.politica_v31(abaixo, _prob_didatica(abaixo), "modo_inexistente")


def test_mesma_carga_compara_filas_e_politica(abaixo):
    p = _prob_didatica(abaixo)
    ordens = nf.ordenacoes(abaixo, p)
    decisoes, _ = nf.politica_v31(abaixo, p, "vigilancia")
    tabela = nf.mesma_carga(abaixo, {"O3": ordens["O3"], "F4": ordens["F4"]}, decisoes)
    assert tabela["estratégia"].tolist() == [
        "v3.1, LLM aceita o MATCH",
        "v3.1, MATCH do LLM ao humano",
        "fila O3",
        "fila F4",
    ]
    variante = tabela.iloc[1]
    filas = tabela.iloc[2:]
    assert filas["revisões humanas"].eq(variante["revisões humanas"]).all()
    assert filas["falsos positivos"].eq(0).all()
    total = int(nf.rotulo(abaixo).sum())
    assert tabela["pares verdadeiros recuperados"].le(total).all()
    assert variante["revisões humanas"] >= tabela.iloc[0]["revisões humanas"]


def test_orcamento_llm_escalado():
    assert nf.orcamento_llm_escalado("vigilancia", 61_485) == 2000
    assert nf.orcamento_llm_escalado("confirmacao", 61_485) == 1000
    assert nf.orcamento_llm_escalado("vigilancia", 10) == 1
