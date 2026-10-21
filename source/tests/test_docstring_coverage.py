"""Keep the public API documented: fail when docstring coverage drops."""
import check_docstrings


def test_docstring_coverage_stays_high():
    items = check_docstrings.collect()
    missing = [name for name, _kind, ok in items if not ok]
    assert len(missing) <= 0.02 * len(items), f"undocumented: {missing[:10]}"
