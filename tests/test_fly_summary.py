"""Tests for the end-of-run termination summary table in trapsim.fly."""

from __future__ import annotations

from trapsim.fly import _print_termination_summary


def _info(reason, splat_object=None):
    return {"reason": reason, "splat_object": splat_object}


class TestTerminationSummary:
    def test_counts_fractions_and_splat_breakdown(self, capsys):
        summaries = {i: _info("trapped") for i in range(3)}
        summaries[10] = _info("splatted", "rod_2_TL")
        summaries[11] = _info("splatted", "rod_2_TL")
        summaries[12] = _info("splatted", "gate_valve")
        summaries[13] = _info("lost")
        summaries[14] = _info("max_time")

        _print_termination_summary(summaries)
        out = capsys.readouterr().out

        assert "Termination summary (8 particles):" in out
        assert "trapped" in out and "3" in out and "37.5%" in out
        assert "splatted" in out and "rod_2_TL" in out and "gate_valve" in out
        # splat breakdown fractions are of ALL particles, not of splats
        assert "25.0%" in out      # 2/8 on rod_2_TL
        # reasons sorted by count: trapped (3) before splatted (3-way tie ok),
        # lost/max_time (1) last
        assert out.index("trapped") < out.index("lost")

    def test_splat_without_label_reports_unknown(self, capsys):
        _print_termination_summary({1: _info("splatted", None)})
        out = capsys.readouterr().out
        assert "(unknown)" in out
        assert "1 particle):" in out

    def test_empty_summaries_prints_nothing(self, capsys):
        _print_termination_summary({})
        assert capsys.readouterr().out == ""
