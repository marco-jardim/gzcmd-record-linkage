"""Regressões de erros após a primeira renderização de um controle."""

from contextlib import redirect_stdout
from io import StringIO

import nb_helpers as nbh
import pytest

widgets = pytest.importorskip("ipywidgets")


@pytest.mark.parametrize("strict", [False, True])
def test_erro_apos_mudar_valor_fica_no_output(monkeypatch, capsys, strict):
    # Sem frontend de Jupyter, emulamos sua captura de streams no Output real.
    # A impressão fora de `with result` não entra em outputs e falha este teste.
    class CapturingOutput(widgets.Output):
        def __enter__(self):
            self._stream = StringIO()
            self._redirect = redirect_stdout(self._stream)
            self._redirect.__enter__()
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            self._redirect.__exit__(exc_type, exc_value, traceback)
            text = self._stream.getvalue()
            if text:
                self.append_stdout(text)
            return False

        def clear_output(self, *args, **kwargs):
            self.outputs = ()

    monkeypatch.setattr(widgets, "Output", CapturingOutput)
    shown = []
    monkeypatch.setattr("IPython.display.display", shown.append)
    monkeypatch.setenv("NBH_STRICT_WIDGETS", "1" if strict else "0")

    def callback(valor):
        if valor > 1:
            raise ValueError("mudança inválida")

    assert nbh.interagir(callback, {
        "valor": nbh.Deslizante(1, 0, 3, 1, "Valor", inteiro=True),
    })
    slider, output = shown[0].children
    if strict:
        with pytest.raises(ValueError, match="mudança inválida"):
            slider.value = 2
    else:
        slider.value = 2
    captured = "".join(item.get("text", "") for item in output.outputs)
    assert "[falha no controle: mudança inválida]" in captured
    assert "modo não interativo" not in captured
    assert "mudança inválida" not in capsys.readouterr().out
    for widget in [*shown[0].children, shown[0]]:
        widget.close()
