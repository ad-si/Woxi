"""Tests for the structured expression tree API.

Run with: python -m pytest woxi-py/tests/
Requires the extension to be built first (e.g. `maturin develop`).
"""

from fractions import Fraction

import pytest

import woxi
from woxi import BigReal, Expr, Symbol, to_python, wl


def tree(code: str):
    """The structured result of evaluating `code`."""
    return woxi.evaluate(code).expr


def full_form(value) -> str:
    """Render a Python tree the way Wolfram's FullForm prints it.

    Used to check the tree against the interpreter's own FullForm, which
    is the contract the representation promises.
    """
    if isinstance(value, bool):  # bool before int: bool is an int subclass
        raise AssertionError("a tree never contains a Python bool")
    if isinstance(value, Symbol):
        return value.name
    if isinstance(value, str):
        return f'\\"{value}\\"'
    if isinstance(value, float):
        # FullForm marks machine reals with a trailing backtick and drops
        # the trailing zero of an integral value.
        return f"{int(value)}.`" if value.is_integer() else f"{value}`"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, list):
        return f"List[{', '.join(full_form(v) for v in value)}]"
    if isinstance(value, Expr):
        args = ", ".join(full_form(a) for a in value.args)
        return f"{full_form(value.head)}[{args}]"
    raise AssertionError(f"unexpected node {value!r}")


class TestAtoms:
    @pytest.mark.parametrize(
        ("code", "expected"),
        [
            ("3", 3),
            ("-3", -3),
            ("2^200", 2**200),
            ("3.5", 3.5),
            ('"abc"', "abc"),
            ("x", Symbol("x")),
            ("Pi", Symbol("Pi")),
            ("True", Symbol("True")),
            ("False", Symbol("False")),
            ("Null", Symbol("Null")),
            ("{1, 2, 3}", [1, 2, 3]),
            ("{}", []),
        ],
    )
    def test_atom_maps_to_native_type(self, code, expected):
        assert tree(code) == expected

    def test_big_integer_is_exact(self):
        value = tree("100!")
        assert isinstance(value, int)
        assert str(value) == woxi.evaluate("100!").result

    def test_arbitrary_precision_real(self):
        value = tree("N[Pi, 30]")
        assert isinstance(value, BigReal)
        assert value.digits.startswith("3.14159265358979323846")
        assert value.precision == 30

    def test_nested_list(self):
        assert tree("{{1, 2}, {3}}") == [[1, 2], [3]]


class TestNormalisation:
    """Operator forms report their FullForm heads, not the parser's AST."""

    @pytest.mark.parametrize(
        ("code", "expected"),
        [
            ("Hold[x + y]", "Plus[x, y]"),
            ("Hold[x - y]", "Plus[x, Times[-1, y]]"),
            ("Hold[a/b]", "Times[a, Power[b, -1]]"),
            ("Hold[x^2]", "Power[x, 2]"),
            ("Hold[-x]", "Times[-1, x]"),
            ("Hold[x_]", "Pattern[x, Blank[]]"),
            ("Hold[x_Integer]", "Pattern[x, Blank[Integer]]"),
            ("Hold[_]", "Blank[]"),
            ("Hold[a -> b]", "Rule[a, b]"),
            ("Hold[a :> b]", "RuleDelayed[a, b]"),
            ("Hold[a && b]", "And[a, b]"),
            ("Hold[a == b]", "Equal[a, b]"),
            ("Hold[a <= b < c]", "Inequality[a, LessEqual, b, Less, c]"),
            ("Hold[f /@ l]", "Map[f, l]"),
            ("Hold[{1, 2}]", "List[1, 2]"),
        ],
    )
    def test_operator_form_uses_full_form_head(self, code, expected):
        held = tree(code)
        assert isinstance(held, Expr)
        assert held.head == Symbol("Hold")
        assert full_form(held.args[0]) == expected

    def test_infinity_is_directed_infinity(self):
        # FullForm[Infinity] is DirectedInfinity[1]; the tree agrees.
        assert tree("Infinity") == Expr(Symbol("DirectedInfinity"), [1])

    def test_compound_head_keeps_its_structure(self):
        # A curried call's head is an expression, not a flattened string.
        assert tree("Hold[f[a][b]]").args[0] == Expr(
            Expr(Symbol("f"), [Symbol("a")]), [Symbol("b")]
        )


class TestFullFormAgreement:
    """The tree must say exactly what the interpreter's FullForm says."""

    @pytest.mark.parametrize(
        "code",
        [
            "1/3 + 1/6",
            "2 + 3 I",
            "I",
            "(1 + I)/2",
            "Sqrt[-4]",
            "1.5 + 2.5 I",
            "Solve[x^2 == 2, x]",
            "<|\"a\" -> 1, \"b\" -> 2|>",
            "Integrate[x^2, x]",
            "Series[Exp[x], {x, 0, 3}] // Normal",
            "{1, 2.5, \"s\", x}",
            "D[Sin[x] Cos[x], x]",
            "Expand[(a + b)^3]",
            "Hold[x_ :> x + 1]",
            "Range[5]",
            "Factor[x^2 - 1]",
            "3.0",
            "-7",
        ],
    )
    def test_tree_matches_interpreter_full_form(self, code):
        expected = woxi.interpret(f"ToString[FullForm[{code}]]")
        assert full_form(tree(code)) == expected


class TestToPython:
    def test_rational(self):
        assert to_python(tree("1/3 + 1/6")) == Fraction(1, 2)

    def test_complex(self):
        assert to_python(tree("2 + 3 I")) == complex(2, 3)

    def test_association(self):
        assert to_python(tree('<|"a" -> 1, "b" -> 2|>')) == {"a": 1, "b": 2}

    def test_symbolic_constants(self):
        assert to_python(tree("True")) is True
        assert to_python(tree("False")) is False
        assert to_python(tree("Null")) is None

    def test_recurses_into_lists_and_args(self):
        assert to_python(tree("{1/2, 1/4}")) == [Fraction(1, 2), Fraction(1, 4)]
        converted = to_python(tree("1/2 + x"))
        assert converted.head == Symbol("Plus")
        assert converted.args == [Fraction(1, 2), Symbol("x")]

    def test_leaves_unconvertible_nodes_alone(self):
        # A complex with symbolic parts has no Python counterpart, and an
        # association with an unhashable key has no dict counterpart.
        symbolic = to_python(tree("Hold[Complex[a, b]]").args[0])
        assert symbolic == Expr(Symbol("Complex"), [Symbol("a"), Symbol("b")])
        assoc = to_python(tree("<|{1, 2} -> 3|>"))
        assert isinstance(assoc, Expr)
        assert assoc.head == Symbol("Association")

    def test_output_is_accepted_back(self):
        converted = to_python(tree('<|"a" -> 1/2|>'))
        assert converted == {"a": Fraction(1, 2)}
        assert woxi.evaluate_expr(converted).result == "<|a -> 1/2|>"

    def test_passes_through_plain_values(self):
        assert to_python(3) == 3
        assert to_python("s") == "s"
        assert to_python(Symbol("x")) == Symbol("x")


class TestEvaluateExpr:
    def test_matches_the_text_path(self):
        built = woxi.evaluate_expr(wl.Integrate(wl.Power(wl.x, 2), wl.x))
        assert built.result == woxi.evaluate("Integrate[x^2, x]").result
        assert built.expr == woxi.evaluate("Integrate[x^2, x]").expr

    def test_python_values_need_no_string_building(self):
        data = [1, 2, 3, 4]
        assert woxi.evaluate_expr(wl.Total(data)).result == "10"
        assert woxi.evaluate_expr(wl.Mean(data)).result == "5/2"

    def test_string_argument_is_data_not_code(self):
        # The distinction `evaluate` cannot make: here "1 + 1" is a string.
        assert woxi.evaluate_expr(wl.StringLength("1 + 1")).result == "5"

    def test_captures_stdout(self):
        res = woxi.evaluate_expr(wl.Print("hi"))
        assert res.stdout == "hi\n"

    def test_captures_graphics(self):
        res = woxi.evaluate_expr(wl.Plot(wl.Sin(wl.x), [wl.x, 0, 10]))
        assert res.graphics is not None
        assert "<svg" in res.graphics

    def test_round_trips_a_result(self):
        first = woxi.evaluate("Solve[x^2 == 2, x]")
        assert woxi.evaluate_expr(first.expr).result == first.result

    def test_explicit_list_head_is_a_list(self):
        assert woxi.evaluate_expr(Expr(Symbol("List"), [1, 2])).result == "{1, 2}"

    def test_raises_on_evaluation_error(self):
        with pytest.raises(woxi.WolframError):
            woxi.evaluate_expr(wl.Throw(1))


class TestPythonToExpr:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (3, "3"),
            (2**200, str(2**200)),
            (3.5, "3.5"),
            ("abc", "abc"),
            (True, "True"),
            (False, "False"),
            (None, "Null"),
            ([1, 2], "{1, 2}"),
            ((1, 2), "{1, 2}"),
            (Fraction(1, 2), "1/2"),
            (complex(2, 3), "2. + 3.*I"),
            # `evaluate` renders strings in OutputForm, so the key is bare.
            ({"a": 1}, "<|a -> 1|>"),
            (BigReal("3.25", 20), "3.25`20."),
        ],
    )
    def test_python_value_evaluates(self, value, expected):
        assert woxi.evaluate_expr(value).result == expected

    def test_bool_is_not_an_integer(self):
        # bool subclasses int in Python; True must not become 1.
        assert woxi.evaluate_expr([True, 1]).result == "{True, 1}"

    def test_rejects_unsupported_object(self):
        with pytest.raises(TypeError):
            woxi.evaluate_expr(object())

    def test_rejects_output_only_graphics(self):
        # A pre-rendered plot has no symbolic form left to evaluate.
        graphic = woxi.evaluate("Plot[Sin[x], {x, 0, 10}]").expr
        assert isinstance(graphic, woxi.Graphics)
        with pytest.raises(TypeError):
            woxi.evaluate_expr(graphic)

    def test_rejects_deeply_nested_tree(self):
        deep = Symbol("x")
        for _ in range(6000):
            deep = wl.f(deep)
        with pytest.raises(RecursionError):
            woxi.evaluate_expr(deep)


class TestParseExpr:
    def test_does_not_evaluate(self):
        assert woxi.parse_expr("1 + 1") == Expr(Symbol("Plus"), [1, 1])
        assert woxi.evaluate("1 + 1").expr == 2

    def test_normalises_operators(self):
        assert woxi.parse_expr("a/b") == Expr(
            Symbol("Times"),
            [Symbol("a"), Expr(Symbol("Power"), [Symbol("b"), -1])],
        )

    def test_multiple_statements_are_a_compound_expression(self):
        parsed = woxi.parse_expr("a = 1; a + 1")
        assert parsed.head == Symbol("CompoundExpression")
        assert len(parsed.args) == 2

    def test_raises_on_syntax_error(self):
        with pytest.raises(woxi.WolframError):
            woxi.parse_expr("f[")

    def test_result_can_be_evaluated(self):
        assert woxi.evaluate_expr(woxi.parse_expr("1 + 1")).result == "2"


class TestRenderedOutput:
    """The tree reports the value, not the display pipeline's rendering."""

    def test_symbolic_forms_survive(self):
        assert woxi.evaluate("Grid[{{1, 2}}]").expr == Expr(
            Symbol("Grid"), [[[1, 2]]]
        )
        assert woxi.evaluate("Graphics[{Circle[]}]").expr.head == Symbol(
            "Graphics"
        )

    def test_image_is_an_opaque_placeholder(self):
        image = woxi.evaluate("Image[{{0, 1}, {1, 0}}]").expr
        assert isinstance(image, woxi.Image)
        assert (image.width, image.height) == (2, 2)

    def test_plot_is_an_opaque_graphic(self):
        graphic = woxi.evaluate("Plot[Sin[x], {x, 0, 10}]").expr
        assert isinstance(graphic, woxi.Graphics)
        assert graphic.is_3d is False
        assert "<svg" in graphic.svg


class TestNodeTypes:
    def test_symbol_equality_and_hashing(self):
        assert Symbol("x") == Symbol("x")
        assert Symbol("x") != Symbol("y")
        assert Symbol("x") != "x"
        assert len({Symbol("x"), Symbol("x"), Symbol("y")}) == 2

    def test_expr_equality_is_structural(self):
        assert Expr(Symbol("f"), [1]) == Expr(Symbol("f"), [1])
        assert Expr(Symbol("f"), [1]) != Expr(Symbol("f"), [2])
        assert Expr(Symbol("f"), [1]) != Symbol("f")

    def test_expr_is_hashable_when_its_args_are(self):
        assert len({Expr(Symbol("f"), [1]), Expr(Symbol("f"), [1])}) == 1
        with pytest.raises(TypeError):
            hash(Expr(Symbol("f"), [[1]]))  # a list arg is unhashable

    def test_calling_builds_an_application(self):
        assert Symbol("f")(1, 2) == Expr(Symbol("f"), [1, 2])
        assert Symbol("f")(1)(2) == Expr(Expr(Symbol("f"), [1]), [2])

    def test_expr_args_accepts_any_iterable(self):
        assert Expr(Symbol("f"), (i for i in [1, 2])) == Expr(Symbol("f"), [1, 2])

    def test_expr_rejects_non_iterable_args(self):
        with pytest.raises(TypeError):
            Expr(Symbol("f"), 1)

    def test_repr_round_trips(self):
        node = Expr(Symbol("f"), [1, Symbol("x")])
        assert repr(node) == "Expr(Symbol(\"f\"), [1, Symbol(\"x\")])"


class TestWlBuilder:
    def test_attribute_is_a_symbol(self):
        assert wl.x == Symbol("x")
        assert wl.Integrate == Symbol("Integrate")

    def test_call_builds_an_expression(self):
        assert wl.f(1, 2) == Expr(Symbol("f"), [1, 2])

    def test_dunder_lookup_is_an_attribute_error(self):
        with pytest.raises(AttributeError):
            wl.__nonexistent__


class TestEvaluationResult:
    def test_expr_accompanies_the_string_result(self):
        res = woxi.evaluate("1 + 1")
        assert res.result == "2"
        assert res.expr == 2

    def test_suppressed_output_has_no_expr(self):
        woxi.clear_state()
        res = woxi.evaluate("x = 1;")
        assert res.result == "Null"
        assert res.expr is None
        woxi.clear_state()

    def test_a_null_value_is_reported_as_null(self):
        # No value at all (None) is distinct from the value Null.
        assert woxi.evaluate('Print["hi"]').expr == Symbol("Null")

    def test_expr_is_not_stale_after_a_failure(self):
        woxi.clear_state()
        woxi.evaluate("2 + 3")
        with pytest.raises(woxi.WolframError):
            woxi.evaluate("1 +")
        assert woxi.evaluate("x = 1;").expr is None
        woxi.clear_state()

    def test_repr_includes_expr(self):
        assert "expr=2" in repr(woxi.evaluate("1 + 1"))
