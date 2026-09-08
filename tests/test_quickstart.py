"""Quickstart smoke test: the demo runs and catches the headline fault.

Guards demo rot — if the example breaks, the README lies. Uses the demo's own
defaults (small network, few epochs) so it stays proportional to the suite.
"""

from examples.quickstart import main


def test_quickstart_runs_and_catches_headline():
    summary = main()
    assert summary["headline_status"] in ("QUARANTINED", "ESTIMATED")
    assert "p=" in summary["headline_verdict"]
    assert summary["point"]["n_true"] > 0
    assert summary["point"]["recall"] > 0.0
    assert summary["event_recall"] > 0.5
    assert summary["genuine_extreme_far"] < 0.2
    assert len(summary["ladder"]) == 2
