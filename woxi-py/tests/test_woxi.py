"""Tests for the woxi Python package.

Run with: python -m pytest woxi-py/tests/
Requires the extension to be built first (e.g. `maturin develop`).
"""

import subprocess
import sys

import pytest

import woxi


class TestInterpret:
    def test_arithmetic(self):
        assert woxi.interpret("Plus[1, 2]") == "3"

    def test_symbolic(self):
        assert woxi.interpret("Integrate[x^2, x]") == "x^3/3"

    def test_exact_rational(self):
        assert woxi.interpret("1/3 + 1/6") == "1/2"

    def test_string(self):
        assert woxi.interpret('StringJoin["a", "b"]') == "ab"

    def test_list(self):
        assert woxi.interpret("Map[#^2 &, {1, 2, 3}]") == "{1, 4, 9}"

    def test_suppressed_output_is_null(self):
        assert woxi.interpret("x = 1;") == "Null"
        woxi.clear_state()

    def test_parse_error_raises(self):
        with pytest.raises(woxi.WolframError):
            woxi.interpret("1 +")

    def test_state_persists_across_calls(self):
        woxi.interpret("stateTest = 42;")
        assert woxi.interpret("stateTest + 1") == "43"
        woxi.clear_state()
        assert woxi.interpret("stateTest") == "stateTest"


class TestEvaluate:
    def test_result_and_stdout(self):
        res = woxi.evaluate('Print["hi"]; 1 + 1')
        assert res.result == "2"
        assert res.stdout == "hi\n"

    def test_stdout_not_echoed(self, capfd):
        woxi.evaluate('Print["not echoed"]')
        captured = capfd.readouterr()
        assert "not echoed" not in captured.out

    def test_graphics_is_svg(self):
        res = woxi.evaluate("Plot[Sin[x], {x, 0, 10}]")
        assert res.graphics is not None
        assert "<svg" in res.graphics

    def test_no_graphics_for_plain_result(self):
        assert woxi.evaluate("1 + 1").graphics is None

    def test_warnings(self):
        res = woxi.evaluate("1/0")
        assert res.result == "ComplexInfinity"
        assert any("infy" in w or "Infinite" in w for w in res.warnings)

    def test_repr(self):
        assert 'result="2"' in repr(woxi.evaluate("1 + 1"))


class TestRng:
    def test_seeded_rng_is_reproducible(self):
        woxi.seed_rng(7)
        first = woxi.interpret("RandomInteger[{1, 10^9}]")
        woxi.seed_rng(7)
        second = woxi.interpret("RandomInteger[{1, 10^9}]")
        woxi.unseed_rng()
        assert first == second


class TestCli:
    def _woxi(self, *args, stdin=None):
        return subprocess.run(
            [sys.executable, "-m", "woxi", *args],
            capture_output=True,
            text=True,
            input=stdin,
        )

    def test_eval(self):
        proc = self._woxi("eval", "Plus[1, 2]")
        assert proc.returncode == 0
        assert proc.stdout == "3\n"

    def test_eval_print_goes_to_stdout(self):
        proc = self._woxi("eval", 'Print["side effect"]; 5')
        assert proc.stdout == "side effect\n5\n"

    def test_eval_stdin(self):
        proc = self._woxi("eval", "-", stdin="Times[6, 7]")
        assert proc.stdout == "42\n"

    def test_eval_error_exit_code(self):
        proc = self._woxi("eval", "1 +")
        assert proc.returncode == 1
        assert "Error" in proc.stderr

    def test_run_script(self, tmp_path):
        script = tmp_path / "script.wls"
        script.write_text('#!/usr/bin/env woxi\nPrint["from script"]\n99\n')
        proc = self._woxi("run", str(script))
        assert proc.returncode == 0
        # Print output shows; the final expression value is suppressed.
        assert proc.stdout == "from script\n"

    def test_run_script_command_line(self, tmp_path):
        script = tmp_path / "args.wls"
        script.write_text("Print[Part[$ScriptCommandLine, 2]]\n")
        proc = self._woxi("run", str(script), "hello")
        assert proc.stdout == "hello\n"

    def test_file_shorthand(self, tmp_path):
        script = tmp_path / "direct.wls"
        script.write_text('Print["direct"]\n')
        proc = self._woxi(str(script))
        assert proc.stdout == "direct\n"

    def test_version(self):
        proc = self._woxi("--version")
        assert proc.stdout == f"woxi {woxi.__version__}\n"

    def test_help(self):
        proc = self._woxi("--help")
        assert proc.returncode == 0
        assert "Usage" in proc.stdout
