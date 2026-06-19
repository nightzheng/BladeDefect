import numpy as np

from blade_defect.fusion import IdentityFusion, IdentityRegistration


def test_rgb_only_fusion() -> None:
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    assert np.array_equal(IdentityRegistration().register(rgb).image, rgb)
    result = IdentityFusion().fuse(rgb)
    assert result.metadata["thermal_used"] is False
