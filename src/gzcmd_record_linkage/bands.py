from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import BandDefinition
from .config import GZCMDConfig


@dataclass(frozen=True)
class BandAssigner:
    """Assigns band labels based on `nota_final` numeric ranges."""

    definitions: tuple[BandDefinition, ...]

    @classmethod
    def from_config(cls, cfg: GZCMDConfig) -> "BandAssigner":
        return cls(definitions=cfg.bands.definitions)

    def assign(self, series: pd.Series) -> pd.Series:
        s = pd.to_numeric(series, errors="coerce")
        out = pd.Series(pd.NA, index=s.index, dtype="string")

        for bd in self.definitions:
            lo_ok = True if bd.min is None else (s >= bd.min)
            if bd.max is None:
                hi_ok = True
            else:
                hi_ok = (s <= bd.max) if bd.inclusive_max else (s < bd.max)

            cond = out.isna() & lo_ok & hi_ok & s.notna()
            out = out.mask(cond, bd.name)

        return out
