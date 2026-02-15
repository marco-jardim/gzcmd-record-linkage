from __future__ import annotations

import pandas as pd

from gzcmd_record_linkage.bands import BandAssigner
from gzcmd_record_linkage.config import BandDefinition


def test_band_assigner_covers_expected_ranges() -> None:
    assigner = BandAssigner(
        definitions=(
            BandDefinition(name="low", min=0.0, max=5.0, inclusive_max=False),
            BandDefinition(name="mid", min=5.0, max=9.0, inclusive_max=False),
            BandDefinition(name="high", min=9.0, max=None, inclusive_max=True),
        )
    )
    out = assigner.assign(pd.Series([0.0, 4.9, 5.0, 8.99, 9.0]))

    assert out.tolist() == ["low", "low", "mid", "mid", "high"]
