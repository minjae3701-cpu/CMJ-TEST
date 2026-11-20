"""Run: pytest -q test_timing.py"""
import pytest
from main import resolve_timing

@pytest.mark.parametrize("reported", [0.0, 1000.0, -1.0])
def test_invalid_browser_fps(reported):
    fps, stamps, method = resolve_timing(300, reported, [0.0] * 300, 10.0)
    assert fps == 30.0
    assert stamps[0] == 0
    assert abs(stamps[-1] - 299/30) < 1e-5
    assert "recomputed_timestamps" in method

def test_monotonic_positions():
    fps, stamps, method = resolve_timing(120, 30, [i * 1000/30 for i in range(120)], 4)
    assert fps == 30.0 and abs(stamps[20] - 20/30) < 1e-5
    assert "decoded_pts" in method

def test_inconsistent_metadata():
    fps, stamps, method = resolve_timing(200, 100, [0.0] * 200, 10)
    assert fps == 20 and "metadata mismatch" in method

def test_no_duration_and_bad_fps():
    with pytest.raises(ValueError):
        resolve_timing(50, 0, [0.0] * 50, None)
