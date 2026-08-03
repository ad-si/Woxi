//! `interpret_to_expr` is the Expr-returning program entry point behind the
//! playground and `Manipulate`. It has to evaluate a whole program, not just
//! the first statement it happens to recognize.

use woxi::interpret_to_expr;
use woxi::syntax::expr_to_string;

fn run(src: &str) -> String {
  expr_to_string(&interpret_to_expr(src).unwrap())
}

#[test]
fn the_last_statement_is_the_result() {
  // These used to return the *first* statement's value.
  assert_eq!(run("a = 1; {a}"), "{1}");
  assert_eq!(run("b = 2; b + 1"), "3");
  assert_eq!(run("c = 1; c = c + 1; c * 10"), "20");
}

#[test]
fn a_definition_with_a_pattern_is_not_skipped() {
  // The `FunctionDefinition` statement was dropped, so the call stayed
  // unevaluated.
  assert_eq!(run("f[x_] := x * 2; f[3]"), "6");
  assert_eq!(run("g[x__] := {x}; g[1, 2]"), "{1, 2}");
}

/// A newline ends a statement here just as it does in `interpret`. Without
/// that, a definition cell written one definition per line — the shape of a
/// Wolfram Demonstration's initialization code — parsed as a single
/// statement glued together by implicit multiplication, and every definition
/// but the first was lost.
#[test]
fn a_newline_separates_statements() {
  assert_eq!(run("aa[x_] := x + 1\nbb[y_] := aa[y] * 2\nbb[3]"), "8");
  assert_eq!(run("p = 1\nq = p + 1\n{p, q}"), "{1, 2}");
  // A statement that is merely *spread* over several lines still parses as
  // one: the break falls inside brackets, not at the top level.
  assert_eq!(run("Total[{1, 2,\n 3}]"), "6");
}

#[test]
fn a_single_expression_still_works() {
  assert_eq!(run("1 + 1"), "2");
  assert_eq!(run("{1, 2, 3}"), "{1, 2, 3}");
}

/// The entry points must not drift apart again. `interpret` (the String API),
/// `interpret_to_expr` (the Expr API behind the playground and Manipulate) and
/// `ToExpression` (the in-language one) all walk a parsed program, and each
/// used to carry its own copy of "which nodes are statements". Two bugs came
/// from those copies disagreeing.
#[test]
fn the_program_entry_points_agree() {
  for src in [
    "a = 1; {a}",
    "b = 2; b + 1",
    "f[x_] := x * 2; f[3]",
    "g[x__] := {x}; g[1, 2]",
    "h[x_, y_: 2] := {x, y}; h[1]",
    "1 + 1",
    "{1, 2, 3}",
    "c = 1; c = c + 1; c * 10",
  ] {
    let via_expr = run(src);
    let via_string = woxi::interpret(src).unwrap();
    assert_eq!(
      via_expr, via_string,
      "interpret vs interpret_to_expr: {src}"
    );

    // ToExpression takes the same program as a string literal.
    let quoted = format!("ToExpression[\"{}\"]", src.replace('"', "\\\""));
    let via_to_expression = woxi::interpret(&quoted).unwrap();
    assert_eq!(
      via_expr, via_to_expression,
      "interpret_to_expr vs ToExpression: {src}"
    );
  }
}
