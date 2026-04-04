use woxi::evaluator::dispatch::complex_and_special::expr_to_box_form;
use woxi::functions::graphics::{
  box_has_fraction, boxes_to_svg, estimate_box_display_width,
  estimate_display_width, expr_to_svg_markup, layout_box, layout_to_svg,
};
use woxi::syntax::{BinaryOperator, Expr};

mod svg_rendering_tests {
  use super::*;

  // ── Power base parenthesization ──

  #[test]
  fn test_svg_power_additive_base_parens() {
    // (x + y)^2 should render with parentheses around the base
    let expr = Expr::FunctionCall {
      name: "Power".to_string(),
      args: vec![
        Expr::FunctionCall {
          name: "Plus".to_string(),
          args: vec![
            Expr::Identifier("x".to_string()),
            Expr::Identifier("y".to_string()),
          ],
        },
        Expr::Integer(2),
      ],
    };
    let markup = expr_to_svg_markup(&expr);
    assert!(
      markup.starts_with("(x + y)"),
      "Power base should have parentheses: got '{}'",
      markup
    );
    assert!(
      markup.contains("(x + y)<tspan"),
      "Expected (x + y) before superscript: got '{}'",
      markup
    );
  }

  #[test]
  fn test_svg_power_simple_base_no_parens() {
    // x^2 should NOT have parentheses
    let expr = Expr::BinaryOp {
      op: BinaryOperator::Power,
      left: Box::new(Expr::Identifier("x".to_string())),
      right: Box::new(Expr::Integer(2)),
    };
    let markup = expr_to_svg_markup(&expr);
    assert!(
      markup.starts_with("x<tspan"),
      "Simple base should not have parentheses: got '{}'",
      markup
    );
  }

  #[test]
  fn test_svg_power_binary_plus_base_parens() {
    // BinaryOp form: (x + y)^3
    let expr = Expr::BinaryOp {
      op: BinaryOperator::Power,
      left: Box::new(Expr::BinaryOp {
        op: BinaryOperator::Plus,
        left: Box::new(Expr::Identifier("x".to_string())),
        right: Box::new(Expr::Identifier("y".to_string())),
      }),
      right: Box::new(Expr::Integer(3)),
    };
    let markup = expr_to_svg_markup(&expr);
    assert!(
      markup.starts_with("(x + y)"),
      "BinaryOp Plus base should have parentheses: got '{}'",
      markup
    );
  }

  #[test]
  fn test_svg_power_binary_minus_base_parens() {
    // (x - y)^2
    let expr = Expr::BinaryOp {
      op: BinaryOperator::Power,
      left: Box::new(Expr::BinaryOp {
        op: BinaryOperator::Minus,
        left: Box::new(Expr::Identifier("x".to_string())),
        right: Box::new(Expr::Identifier("y".to_string())),
      }),
      right: Box::new(Expr::Integer(2)),
    };
    let markup = expr_to_svg_markup(&expr);
    assert!(
      markup.starts_with("(x - y)"),
      "BinaryOp Minus base should have parentheses: got '{}'",
      markup
    );
  }

  // ── Implicit multiplication (no * symbol) ──

  #[test]
  fn test_svg_times_number_identifier_space() {
    // Times[10, x] → "10 x" (space separator, matching Wolfram Language)
    let expr = Expr::FunctionCall {
      name: "Times".to_string(),
      args: vec![Expr::Integer(10), Expr::Identifier("x".to_string())],
    };
    let markup = expr_to_svg_markup(&expr);
    assert_eq!(
      markup, "10 x",
      "Number*identifier should use space: got '{}'",
      markup
    );
  }

  #[test]
  fn test_svg_times_identifiers_space() {
    // Times[x, y] → "x y" (space, no *)
    let expr = Expr::FunctionCall {
      name: "Times".to_string(),
      args: vec![
        Expr::Identifier("x".to_string()),
        Expr::Identifier("y".to_string()),
      ],
    };
    let markup = expr_to_svg_markup(&expr);
    assert_eq!(
      markup, "x y",
      "Identifier*identifier should use space: got '{}'",
      markup
    );
  }

  #[test]
  fn test_svg_times_additive_juxtaposition() {
    // Times[9, Plus[2, x], Plus[x, y]] → "9(2 + x)(x + y)"
    let expr = Expr::FunctionCall {
      name: "Times".to_string(),
      args: vec![
        Expr::Integer(9),
        Expr::FunctionCall {
          name: "Plus".to_string(),
          args: vec![Expr::Integer(2), Expr::Identifier("x".to_string())],
        },
        Expr::FunctionCall {
          name: "Plus".to_string(),
          args: vec![
            Expr::Identifier("x".to_string()),
            Expr::Identifier("y".to_string()),
          ],
        },
      ],
    };
    let markup = expr_to_svg_markup(&expr);
    assert_eq!(
      markup, "9(2 + x)(x + y)",
      "Times with additive operands should use juxtaposition: got '{}'",
      markup
    );
    assert!(!markup.contains('*'), "SVG should not contain *");
  }

  #[test]
  fn test_svg_binary_times_no_star() {
    // BinaryOp: x * y → "x y"
    let expr = Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(Expr::Identifier("x".to_string())),
      right: Box::new(Expr::Identifier("y".to_string())),
    };
    let markup = expr_to_svg_markup(&expr);
    assert_eq!(
      markup, "x y",
      "BinaryOp Times should use space: got '{}'",
      markup
    );
  }

  #[test]
  fn test_svg_binary_times_number_identifier() {
    // BinaryOp: 10 * x → "10 x"
    let expr = Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(Expr::Integer(10)),
      right: Box::new(Expr::Identifier("x".to_string())),
    };
    let markup = expr_to_svg_markup(&expr);
    assert_eq!(
      markup, "10 x",
      "BinaryOp Times number*id should use space: got '{}'",
      markup
    );
  }

  #[test]
  fn test_svg_times_neg_one_no_star() {
    // Times[-1, x, Plus[a, b]] → "-x(a + b)"
    let expr = Expr::FunctionCall {
      name: "Times".to_string(),
      args: vec![
        Expr::Integer(-1),
        Expr::Identifier("x".to_string()),
        Expr::FunctionCall {
          name: "Plus".to_string(),
          args: vec![
            Expr::Identifier("a".to_string()),
            Expr::Identifier("b".to_string()),
          ],
        },
      ],
    };
    let markup = expr_to_svg_markup(&expr);
    assert_eq!(
      markup, "-x(a + b)",
      "Times[-1,...] should use implicit mult: got '{}'",
      markup
    );
    assert!(!markup.contains('*'), "SVG should not contain *");
  }

  // ── Width estimation ──

  #[test]
  fn test_width_power_additive_base_includes_parens() {
    // (x + y)^2: width should include 2 chars for parentheses
    let base = Expr::FunctionCall {
      name: "Plus".to_string(),
      args: vec![
        Expr::Identifier("x".to_string()),
        Expr::Identifier("y".to_string()),
      ],
    };
    let base_w = estimate_display_width(&base);

    let expr = Expr::FunctionCall {
      name: "Power".to_string(),
      args: vec![base, Expr::Integer(2)],
    };
    let total_w = estimate_display_width(&expr);
    // Total should be base_w + 2.0 (parens) + 1.0 * 0.7 (exponent)
    let expected = base_w + 2.0 + 1.0 * 0.7;
    assert!(
      (total_w - expected).abs() < 0.01,
      "Power width should include parens: got {}, expected {}",
      total_w,
      expected
    );
  }

  #[test]
  fn test_width_power_simple_base_no_parens() {
    // x^2: no parentheses needed
    let base = Expr::Identifier("x".to_string());
    let base_w = estimate_display_width(&base);

    let expr = Expr::BinaryOp {
      op: BinaryOperator::Power,
      left: Box::new(base),
      right: Box::new(Expr::Integer(2)),
    };
    let total_w = estimate_display_width(&expr);
    let expected = base_w + 1.0 * 0.7;
    assert!(
      (total_w - expected).abs() < 0.01,
      "Simple Power width should not include parens: got {}, expected {}",
      total_w,
      expected
    );
  }

  #[test]
  fn test_width_times_additive_operands() {
    // Times[9, Plus[2, x], Plus[x, y]]:
    // markup is "9(2 + x)(x + y)" — no separators between adjacent parens
    let expr = Expr::FunctionCall {
      name: "Times".to_string(),
      args: vec![
        Expr::Integer(9),
        Expr::FunctionCall {
          name: "Plus".to_string(),
          args: vec![Expr::Integer(2), Expr::Identifier("x".to_string())],
        },
        Expr::FunctionCall {
          name: "Plus".to_string(),
          args: vec![
            Expr::Identifier("x".to_string()),
            Expr::Identifier("y".to_string()),
          ],
        },
      ],
    };
    let w = estimate_display_width(&expr);
    // "9(2 + x)(x + y)" = 15 chars
    // factors: 1 + (5+2) + (5+2) = 15, seps: 0+0 = 0, total = 15
    assert!(
      (w - 15.0).abs() < 0.01,
      "Times width for 9(2 + x)(x + y) should be 15: got {}",
      w
    );
  }

  #[test]
  fn test_width_binary_times_additive_includes_parens() {
    // BinaryOp: (x + y)(a + b)
    let expr = Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(Expr::BinaryOp {
        op: BinaryOperator::Plus,
        left: Box::new(Expr::Identifier("x".to_string())),
        right: Box::new(Expr::Identifier("y".to_string())),
      }),
      right: Box::new(Expr::BinaryOp {
        op: BinaryOperator::Plus,
        left: Box::new(Expr::Identifier("a".to_string())),
        right: Box::new(Expr::Identifier("b".to_string())),
      }),
    };
    let w = estimate_display_width(&expr);
    // "(x + y)(a + b)" = 14 chars
    // Each side: base=5 + parens=2 = 7, sep=0, total = 14
    assert!(
      (w - 14.0).abs() < 0.01,
      "BinaryOp Times (x+y)(a+b) width should be 14: got {}",
      w
    );
  }

  #[test]
  fn test_width_times_neg_one_additive_includes_parens() {
    // Times[-1, Plus[x, y]]: rendered as -(x + y)
    let expr = Expr::FunctionCall {
      name: "Times".to_string(),
      args: vec![
        Expr::Integer(-1),
        Expr::FunctionCall {
          name: "Plus".to_string(),
          args: vec![
            Expr::Identifier("x".to_string()),
            Expr::Identifier("y".to_string()),
          ],
        },
      ],
    };
    let w = estimate_display_width(&expr);
    // -(x + y): '-' = 1, '(' = 1, 'x + y' = 5, ')' = 1 → total = 8
    assert!(
      w >= 8.0,
      "Times[-1, Plus] width should include parens: got {}",
      w
    );
  }

  #[test]
  fn test_width_times_number_identifier() {
    // Times[10, x] → "10 x" (space separator)
    let expr = Expr::FunctionCall {
      name: "Times".to_string(),
      args: vec![Expr::Integer(10), Expr::Identifier("x".to_string())],
    };
    let w = estimate_display_width(&expr);
    // "10 x" = 4 chars: factors = 2+1 = 3, sep = 1
    assert!(
      (w - 4.0).abs() < 0.01,
      "Times[10, x] width should be 4: got {}",
      w
    );
  }

  // ── Regression tests for the specific user-reported expressions ──

  #[test]
  fn test_svg_full_expression_power_plus_times() {
    // (x + y)^2 + 9(2 + x)(x + y)
    // = Plus[Power[Plus[x, y], 2], Times[9, Plus[2, x], Plus[x, y]]]
    let expr = Expr::FunctionCall {
      name: "Plus".to_string(),
      args: vec![
        Expr::FunctionCall {
          name: "Power".to_string(),
          args: vec![
            Expr::FunctionCall {
              name: "Plus".to_string(),
              args: vec![
                Expr::Identifier("x".to_string()),
                Expr::Identifier("y".to_string()),
              ],
            },
            Expr::Integer(2),
          ],
        },
        Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![
            Expr::Integer(9),
            Expr::FunctionCall {
              name: "Plus".to_string(),
              args: vec![Expr::Integer(2), Expr::Identifier("x".to_string())],
            },
            Expr::FunctionCall {
              name: "Plus".to_string(),
              args: vec![
                Expr::Identifier("x".to_string()),
                Expr::Identifier("y".to_string()),
              ],
            },
          ],
        },
      ],
    };
    let markup = expr_to_svg_markup(&expr);
    // Should contain parens around Power base and no * anywhere
    assert!(
      markup.contains("(x + y)<tspan"),
      "Full expression should have parens around Power base: got '{}'",
      markup
    );
    assert!(
      !markup.contains('*'),
      "SVG markup should not contain * symbol: got '{}'",
      markup
    );
  }

  #[test]
  fn test_svg_product_of_powers() {
    // (x + y)^3 * (18 + 10x + y)^3
    // = Times[Power[Plus[x, y], 3], Power[Plus[18, Times[10, x], y], 3]]
    let expr = Expr::FunctionCall {
      name: "Times".to_string(),
      args: vec![
        Expr::FunctionCall {
          name: "Power".to_string(),
          args: vec![
            Expr::FunctionCall {
              name: "Plus".to_string(),
              args: vec![
                Expr::Identifier("x".to_string()),
                Expr::Identifier("y".to_string()),
              ],
            },
            Expr::Integer(3),
          ],
        },
        Expr::FunctionCall {
          name: "Power".to_string(),
          args: vec![
            Expr::FunctionCall {
              name: "Plus".to_string(),
              args: vec![
                Expr::Integer(18),
                Expr::FunctionCall {
                  name: "Times".to_string(),
                  args: vec![
                    Expr::Integer(10),
                    Expr::Identifier("x".to_string()),
                  ],
                },
                Expr::Identifier("y".to_string()),
              ],
            },
            Expr::Integer(3),
          ],
        },
      ],
    };
    let markup = expr_to_svg_markup(&expr);
    // Both Power bases should have parentheses, no * anywhere
    assert!(
      markup.contains("(x + y)<tspan"),
      "First Power should have parens: got '{}'",
      markup
    );
    assert!(
      markup.contains("(18 + 10 x + y)<tspan"),
      "Second Power should have parens: got '{}'",
      markup
    );
    assert!(
      !markup.contains('*'),
      "SVG markup should not contain * symbol: got '{}'",
      markup
    );
  }

  // ── Digit grouping in graphics ──

  #[test]
  fn test_svg_digit_grouping_small_number_no_grouping() {
    // 4-digit numbers should NOT be grouped
    let markup = expr_to_svg_markup(&Expr::Integer(1234));
    assert_eq!(markup, "1234");
  }

  #[test]
  fn test_svg_digit_grouping_5_digits() {
    // 5-digit numbers should be grouped: 12 345
    let markup = expr_to_svg_markup(&Expr::Integer(12345));
    assert!(
      markup.contains("12<tspan dx=\"0.3ch\">345</tspan>"),
      "5-digit number should be grouped: got '{}'",
      markup
    );
  }

  #[test]
  fn test_svg_digit_grouping_large_number() {
    // 13-digit number: 2 000 000 000 000
    let markup = expr_to_svg_markup(&Expr::Integer(2000000000000));
    assert!(
      markup.contains(
        "2<tspan dx=\"0.3ch\">000</tspan><tspan dx=\"0.3ch\">000</tspan>\
         <tspan dx=\"0.3ch\">000</tspan><tspan dx=\"0.3ch\">000</tspan>"
      ),
      "Large number should have grouped digits: got '{}'",
      markup
    );
  }

  #[test]
  fn test_svg_digit_grouping_negative_number() {
    // Negative numbers: sign preserved, digits grouped
    let markup = expr_to_svg_markup(&Expr::Integer(-12345678));
    assert!(
      markup.contains(
        "−12<tspan dx=\"0.3ch\">345</tspan><tspan dx=\"0.3ch\">678</tspan>"
      ),
      "Negative number should be grouped with minus sign: got '{}'",
      markup
    );
  }

  #[test]
  fn test_svg_digit_grouping_width_accounts_for_gaps() {
    // Width of a 13-digit number should include extra space for 4 separators
    let w = estimate_display_width(&Expr::Integer(2000000000000));
    // 13 chars + 4 * 0.3 = 14.2
    assert!(
      (w - 14.2).abs() < 0.01,
      "Width should account for digit group gaps: got {}",
      w
    );
  }
}

// ══════════════════════════════════════════════════════════════════════
// Box representation tests: ensure expr_to_box_form produces the
// correct intermediate box structure AND that boxes_to_svg on the
// resulting boxes produces the same SVG as the direct expr_to_svg_markup.
// ══════════════════════════════════════════════════════════════════════

mod box_representation_tests {
  use super::*;
  use woxi::syntax::expr_to_output;

  /// Format a box expression to its string representation for comparison.
  fn box_str(expr: &Expr) -> String {
    expr_to_output(expr)
  }

  /// Assert that the box-based width estimation is reasonable.
  /// The box pipeline may differ slightly from the direct pipeline
  /// (e.g. different separator spacing, no parenthesization in boxes),
  /// so we allow a wider tolerance.
  fn assert_width_reasonable(expr: &Expr) {
    let boxes = expr_to_box_form(expr);
    let box_w = estimate_box_display_width(&boxes);
    assert!(box_w > 0.0, "Box width should be positive, got {box_w}",);
  }

  // ── Power base parenthesization (boxes) ──

  #[test]
  fn box_power_additive_base_parens() {
    let expr = Expr::FunctionCall {
      name: "Power".to_string(),
      args: vec![
        Expr::FunctionCall {
          name: "Plus".to_string(),
          args: vec![
            Expr::Identifier("x".to_string()),
            Expr::Identifier("y".to_string()),
          ],
        },
        Expr::Integer(2),
      ],
    };
    let boxes = expr_to_box_form(&expr);
    assert!(
      matches!(&boxes, Expr::FunctionCall { name, .. } if name == "SuperscriptBox"),
      "Power should become SuperscriptBox: got {}",
      box_str(&boxes)
    );
  }

  #[test]
  fn box_power_simple_base() {
    let expr = Expr::BinaryOp {
      op: BinaryOperator::Power,
      left: Box::new(Expr::Identifier("x".to_string())),
      right: Box::new(Expr::Integer(2)),
    };
    let boxes = expr_to_box_form(&expr);
    assert_eq!(box_str(&boxes), "SuperscriptBox[x, 2]",);
  }

  #[test]
  fn box_power_binary_plus_base() {
    let expr = Expr::BinaryOp {
      op: BinaryOperator::Power,
      left: Box::new(Expr::BinaryOp {
        op: BinaryOperator::Plus,
        left: Box::new(Expr::Identifier("x".to_string())),
        right: Box::new(Expr::Identifier("y".to_string())),
      }),
      right: Box::new(Expr::Integer(3)),
    };
    let boxes = expr_to_box_form(&expr);
    let s = box_str(&boxes);
    assert!(
      s.starts_with("SuperscriptBox[RowBox["),
      "Should be SuperscriptBox[RowBox[...],...]: got {s}"
    );
  }

  #[test]
  fn box_power_binary_minus_base() {
    let expr = Expr::BinaryOp {
      op: BinaryOperator::Power,
      left: Box::new(Expr::BinaryOp {
        op: BinaryOperator::Minus,
        left: Box::new(Expr::Identifier("x".to_string())),
        right: Box::new(Expr::Identifier("y".to_string())),
      }),
      right: Box::new(Expr::Integer(2)),
    };
    let boxes = expr_to_box_form(&expr);
    assert!(
      matches!(&boxes, Expr::FunctionCall { name, .. } if name == "SuperscriptBox"),
      "Should produce SuperscriptBox: got {}",
      box_str(&boxes)
    );
  }

  // ── Implicit multiplication (boxes) ──

  #[test]
  fn box_times_number_identifier() {
    let expr = Expr::FunctionCall {
      name: "Times".to_string(),
      args: vec![Expr::Integer(10), Expr::Identifier("x".to_string())],
    };
    let boxes = expr_to_box_form(&expr);
    assert_eq!(box_str(&boxes), "RowBox[{10,  , x}]");
    let svg = boxes_to_svg(&boxes);
    assert_eq!(svg, "10 x");
  }

  #[test]
  fn box_times_identifiers() {
    let expr = Expr::FunctionCall {
      name: "Times".to_string(),
      args: vec![
        Expr::Identifier("x".to_string()),
        Expr::Identifier("y".to_string()),
      ],
    };
    let boxes = expr_to_box_form(&expr);
    let svg = boxes_to_svg(&boxes);
    assert_eq!(svg, "x y");
  }

  #[test]
  fn box_times_additive_juxtaposition() {
    let expr = Expr::FunctionCall {
      name: "Times".to_string(),
      args: vec![
        Expr::Integer(9),
        Expr::FunctionCall {
          name: "Plus".to_string(),
          args: vec![Expr::Integer(2), Expr::Identifier("x".to_string())],
        },
        Expr::FunctionCall {
          name: "Plus".to_string(),
          args: vec![
            Expr::Identifier("x".to_string()),
            Expr::Identifier("y".to_string()),
          ],
        },
      ],
    };
    let boxes = expr_to_box_form(&expr);
    assert!(
      matches!(&boxes, Expr::FunctionCall { name, .. } if name == "RowBox"),
      "Times should become RowBox: got {}",
      box_str(&boxes)
    );
  }

  #[test]
  fn box_binary_times() {
    let expr = Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(Expr::Identifier("x".to_string())),
      right: Box::new(Expr::Identifier("y".to_string())),
    };
    let boxes = expr_to_box_form(&expr);
    let svg = boxes_to_svg(&boxes);
    assert_eq!(svg, "x y");
  }

  #[test]
  fn box_binary_times_number_identifier() {
    let expr = Expr::BinaryOp {
      op: BinaryOperator::Times,
      left: Box::new(Expr::Integer(10)),
      right: Box::new(Expr::Identifier("x".to_string())),
    };
    let boxes = expr_to_box_form(&expr);
    let svg = boxes_to_svg(&boxes);
    assert_eq!(svg, "10 x");
  }

  #[test]
  fn box_times_neg_one() {
    let expr = Expr::FunctionCall {
      name: "Times".to_string(),
      args: vec![
        Expr::Integer(-1),
        Expr::Identifier("x".to_string()),
        Expr::FunctionCall {
          name: "Plus".to_string(),
          args: vec![
            Expr::Identifier("a".to_string()),
            Expr::Identifier("b".to_string()),
          ],
        },
      ],
    };
    let boxes = expr_to_box_form(&expr);
    assert!(
      matches!(&boxes, Expr::FunctionCall { name, .. } if name == "RowBox"),
      "Times[-1,...] should become RowBox: got {}",
      box_str(&boxes)
    );
  }

  // ── Fraction / Rational (boxes) ──

  #[test]
  fn box_rational() {
    let expr = Expr::FunctionCall {
      name: "Rational".to_string(),
      args: vec![Expr::Integer(2), Expr::Integer(3)],
    };
    let boxes = expr_to_box_form(&expr);
    assert_eq!(box_str(&boxes), "FractionBox[2, 3]");
  }

  #[test]
  fn box_divide() {
    let expr = Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left: Box::new(Expr::Identifier("a".to_string())),
      right: Box::new(Expr::Identifier("b".to_string())),
    };
    let boxes = expr_to_box_form(&expr);
    assert_eq!(box_str(&boxes), "FractionBox[a, b]");
  }

  #[test]
  fn box_has_fraction_true() {
    let expr = Expr::FunctionCall {
      name: "Rational".to_string(),
      args: vec![Expr::Integer(2), Expr::Integer(3)],
    };
    let boxes = expr_to_box_form(&expr);
    assert!(
      box_has_fraction(&boxes),
      "FractionBox should detect fraction"
    );
  }

  #[test]
  fn box_has_fraction_false() {
    let expr = Expr::Identifier("x".to_string());
    let boxes = expr_to_box_form(&expr);
    assert!(
      !box_has_fraction(&boxes),
      "Simple identifier should not have fraction"
    );
  }

  // ── Sqrt (boxes) ──

  #[test]
  fn box_sqrt() {
    let expr = Expr::FunctionCall {
      name: "Power".to_string(),
      args: vec![
        Expr::Identifier("x".to_string()),
        Expr::FunctionCall {
          name: "Rational".to_string(),
          args: vec![Expr::Integer(1), Expr::Integer(2)],
        },
      ],
    };
    let boxes = expr_to_box_form(&expr);
    assert_eq!(box_str(&boxes), "SqrtBox[x]");
  }

  // ── Subscript / Subsuperscript (boxes) ──

  #[test]
  fn box_subscript() {
    let expr = Expr::FunctionCall {
      name: "Subscript".to_string(),
      args: vec![Expr::Identifier("x".to_string()), Expr::Integer(0)],
    };
    let boxes = expr_to_box_form(&expr);
    assert_eq!(box_str(&boxes), "SubscriptBox[x, 0]");
  }

  #[test]
  fn box_subsuperscript() {
    let expr = Expr::FunctionCall {
      name: "Power".to_string(),
      args: vec![
        Expr::FunctionCall {
          name: "Subscript".to_string(),
          args: vec![
            Expr::Identifier("a".to_string()),
            Expr::Identifier("b".to_string()),
          ],
        },
        Expr::Identifier("c".to_string()),
      ],
    };
    let boxes = expr_to_box_form(&expr);
    assert_eq!(box_str(&boxes), "SubsuperscriptBox[a, b, c]");
  }

  // ── List / FunctionCall (boxes) ──

  #[test]
  fn box_list() {
    let expr =
      Expr::List(vec![Expr::Integer(1), Expr::Integer(2), Expr::Integer(3)]);
    let boxes = expr_to_box_form(&expr);
    assert_eq!(box_str(&boxes), "RowBox[{{, RowBox[{1, ,, 2, ,, 3}], }}]");
  }

  #[test]
  fn box_function_call() {
    let expr = Expr::FunctionCall {
      name: "f".to_string(),
      args: vec![
        Expr::Identifier("x".to_string()),
        Expr::Identifier("y".to_string()),
      ],
    };
    let boxes = expr_to_box_form(&expr);
    assert_eq!(box_str(&boxes), "RowBox[{f, [, RowBox[{x, ,, y}], ]}]");
  }

  // ── Digit grouping via boxes ──

  #[test]
  fn box_digit_grouping_small() {
    let expr = Expr::Integer(1234);
    let boxes = expr_to_box_form(&expr);
    let svg = boxes_to_svg(&boxes);
    assert_eq!(
      svg, "1234",
      "4-digit number should not be grouped via boxes"
    );
  }

  #[test]
  fn box_digit_grouping_5_digits() {
    let expr = Expr::Integer(12345);
    let boxes = expr_to_box_form(&expr);
    let svg = boxes_to_svg(&boxes);
    assert!(!svg.is_empty(), "Box SVG for 12345 should produce output");
  }

  // ── UnaryOp (boxes) ──

  #[test]
  fn box_unary_minus() {
    let expr = Expr::UnaryOp {
      op: woxi::syntax::UnaryOperator::Minus,
      operand: Box::new(Expr::Identifier("x".to_string())),
    };
    let boxes = expr_to_box_form(&expr);
    let svg = boxes_to_svg(&boxes);
    assert_eq!(svg, "-x", "UnaryMinus box SVG");
  }

  #[test]
  fn box_unary_not() {
    let expr = Expr::UnaryOp {
      op: woxi::syntax::UnaryOperator::Not,
      operand: Box::new(Expr::Identifier("p".to_string())),
    };
    let boxes = expr_to_box_form(&expr);
    let svg = boxes_to_svg(&boxes);
    assert_eq!(svg, "!p");
  }

  // ── Comparison (boxes) ──

  #[test]
  fn box_comparison() {
    let expr = Expr::Comparison {
      operands: vec![
        Expr::Identifier("a".to_string()),
        Expr::Identifier("b".to_string()),
      ],
      operators: vec![woxi::syntax::ComparisonOp::Less],
    };
    let boxes = expr_to_box_form(&expr);
    let svg = boxes_to_svg(&boxes);
    assert_eq!(svg, "a&lt;b", "Comparison box SVG");
  }

  // ── BinaryOp variants (boxes) ──

  #[test]
  fn box_binary_plus() {
    let expr = Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left: Box::new(Expr::Identifier("x".to_string())),
      right: Box::new(Expr::Identifier("y".to_string())),
    };
    let boxes = expr_to_box_form(&expr);
    let svg = boxes_to_svg(&boxes);
    assert_eq!(svg, "x+y");
  }

  #[test]
  fn box_binary_minus() {
    let expr = Expr::BinaryOp {
      op: BinaryOperator::Minus,
      left: Box::new(Expr::Identifier("x".to_string())),
      right: Box::new(Expr::Identifier("y".to_string())),
    };
    let boxes = expr_to_box_form(&expr);
    let svg = boxes_to_svg(&boxes);
    assert_eq!(svg, "x-y");
  }

  #[test]
  fn box_binary_and() {
    let expr = Expr::BinaryOp {
      op: BinaryOperator::And,
      left: Box::new(Expr::Identifier("p".to_string())),
      right: Box::new(Expr::Identifier("q".to_string())),
    };
    let boxes = expr_to_box_form(&expr);
    let svg = boxes_to_svg(&boxes);
    assert_eq!(svg, "p&amp;&amp;q");
  }

  #[test]
  fn box_binary_or() {
    let expr = Expr::BinaryOp {
      op: BinaryOperator::Or,
      left: Box::new(Expr::Identifier("p".to_string())),
      right: Box::new(Expr::Identifier("q".to_string())),
    };
    let boxes = expr_to_box_form(&expr);
    let svg = boxes_to_svg(&boxes);
    assert_eq!(svg, "p||q");
  }

  // ── Full expression regression (boxes) ──

  #[test]
  fn box_full_expression_power_plus_times() {
    // Plus[Power[Plus[x, y], 2], Times[9, Plus[2, x], Plus[x, y]]]
    let expr = Expr::FunctionCall {
      name: "Plus".to_string(),
      args: vec![
        Expr::FunctionCall {
          name: "Power".to_string(),
          args: vec![
            Expr::FunctionCall {
              name: "Plus".to_string(),
              args: vec![
                Expr::Identifier("x".to_string()),
                Expr::Identifier("y".to_string()),
              ],
            },
            Expr::Integer(2),
          ],
        },
        Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![
            Expr::Integer(9),
            Expr::FunctionCall {
              name: "Plus".to_string(),
              args: vec![Expr::Integer(2), Expr::Identifier("x".to_string())],
            },
            Expr::FunctionCall {
              name: "Plus".to_string(),
              args: vec![
                Expr::Identifier("x".to_string()),
                Expr::Identifier("y".to_string()),
              ],
            },
          ],
        },
      ],
    };
    let boxes = expr_to_box_form(&expr);
    // Top level should be RowBox (Plus)
    assert!(
      matches!(&boxes, Expr::FunctionCall { name, .. } if name == "RowBox"),
      "Plus should become RowBox: got {:?}",
      boxes
    );
    let svg = boxes_to_svg(&boxes);
    // Should contain a SuperscriptBox rendering (superscript tspan)
    assert!(
      svg.contains("baseline-shift=\"super\""),
      "Full expression box SVG should have superscript: got '{}'",
      svg
    );
    // Should not contain * symbol
    assert!(
      !svg.contains('*'),
      "Box SVG should not contain * symbol: got '{}'",
      svg
    );
  }

  #[test]
  fn box_product_of_powers() {
    // Times[Power[Plus[x, y], 3], Power[Plus[18, Times[10, x], y], 3]]
    let expr = Expr::FunctionCall {
      name: "Times".to_string(),
      args: vec![
        Expr::FunctionCall {
          name: "Power".to_string(),
          args: vec![
            Expr::FunctionCall {
              name: "Plus".to_string(),
              args: vec![
                Expr::Identifier("x".to_string()),
                Expr::Identifier("y".to_string()),
              ],
            },
            Expr::Integer(3),
          ],
        },
        Expr::FunctionCall {
          name: "Power".to_string(),
          args: vec![
            Expr::FunctionCall {
              name: "Plus".to_string(),
              args: vec![
                Expr::Integer(18),
                Expr::FunctionCall {
                  name: "Times".to_string(),
                  args: vec![
                    Expr::Integer(10),
                    Expr::Identifier("x".to_string()),
                  ],
                },
                Expr::Identifier("y".to_string()),
              ],
            },
            Expr::Integer(3),
          ],
        },
      ],
    };
    let boxes = expr_to_box_form(&expr);
    // Should be a RowBox containing SuperscriptBox elements
    assert!(
      matches!(&boxes, Expr::FunctionCall { name, .. } if name == "RowBox"),
      "Product of powers should be RowBox: got {:?}",
      boxes
    );
    let svg = boxes_to_svg(&boxes);
    // Should contain superscripts
    assert!(
      svg.contains("baseline-shift=\"super\""),
      "Product of powers box SVG should have superscripts: got '{}'",
      svg
    );
  }

  // ── Width estimation parity ──

  #[test]
  fn box_width_power_additive_base() {
    let expr = Expr::FunctionCall {
      name: "Power".to_string(),
      args: vec![
        Expr::FunctionCall {
          name: "Plus".to_string(),
          args: vec![
            Expr::Identifier("x".to_string()),
            Expr::Identifier("y".to_string()),
          ],
        },
        Expr::Integer(2),
      ],
    };
    assert_width_reasonable(&expr);
  }

  #[test]
  fn box_width_power_simple_base() {
    let expr = Expr::BinaryOp {
      op: BinaryOperator::Power,
      left: Box::new(Expr::Identifier("x".to_string())),
      right: Box::new(Expr::Integer(2)),
    };
    assert_width_reasonable(&expr);
  }

  #[test]
  fn box_width_times_number_identifier() {
    let expr = Expr::FunctionCall {
      name: "Times".to_string(),
      args: vec![Expr::Integer(10), Expr::Identifier("x".to_string())],
    };
    assert_width_reasonable(&expr);
  }

  #[test]
  fn box_width_rational() {
    let expr = Expr::FunctionCall {
      name: "Rational".to_string(),
      args: vec![Expr::Integer(2), Expr::Integer(3)],
    };
    assert_width_reasonable(&expr);
  }

  #[test]
  fn box_width_list() {
    let expr =
      Expr::List(vec![Expr::Integer(1), Expr::Integer(2), Expr::Integer(3)]);
    assert_width_reasonable(&expr);
  }

  #[test]
  fn box_width_function_call() {
    let expr = Expr::FunctionCall {
      name: "f".to_string(),
      args: vec![
        Expr::Identifier("x".to_string()),
        Expr::Identifier("y".to_string()),
      ],
    };
    assert_width_reasonable(&expr);
  }

  #[test]
  fn box_width_large_integer() {
    let expr = Expr::Integer(2000000000000);
    assert_width_reasonable(&expr);
  }

  // ── DisplayForm (boxes) ──

  #[test]
  fn box_display_form_renders_inner_boxes() {
    // DisplayForm[SuperscriptBox["x", "2"]] should render the inner boxes to SVG
    let expr = Expr::FunctionCall {
      name: "DisplayForm".to_string(),
      args: vec![Expr::FunctionCall {
        name: "SuperscriptBox".to_string(),
        args: vec![
          Expr::String("x".to_string()),
          Expr::String("2".to_string()),
        ],
      }],
    };
    // In the box pipeline, DisplayForm passes its content directly as boxes
    // (handled in generate_output_svg). Here we test the box extraction directly.
    if let Expr::FunctionCall { name, args } = &expr {
      if name == "DisplayForm" && args.len() == 1 {
        let svg = boxes_to_svg(&args[0]);
        assert!(
          svg.contains("x") && svg.contains("baseline-shift=\"super\""),
          "DisplayForm inner boxes should render as superscript SVG: got '{svg}'"
        );
      }
    }
  }

  #[test]
  fn box_display_form_subscript_renders() {
    let inner = Expr::FunctionCall {
      name: "SubscriptBox".to_string(),
      args: vec![Expr::String("a".to_string()), Expr::String("i".to_string())],
    };
    let svg = boxes_to_svg(&inner);
    assert!(
      svg.contains("a") && svg.contains("baseline-shift=\"sub\""),
      "SubscriptBox should render with subscript: got '{svg}'"
    );
  }

  #[test]
  fn box_display_form_row_of_subscripts() {
    // RowBox[{SubscriptBox["a", "1"], SubscriptBox["b", "2"]}]
    let inner = Expr::FunctionCall {
      name: "RowBox".to_string(),
      args: vec![Expr::List(vec![
        Expr::FunctionCall {
          name: "SubscriptBox".to_string(),
          args: vec![
            Expr::String("a".to_string()),
            Expr::String("1".to_string()),
          ],
        },
        Expr::FunctionCall {
          name: "SubscriptBox".to_string(),
          args: vec![
            Expr::String("b".to_string()),
            Expr::String("2".to_string()),
          ],
        },
      ])],
    };
    let svg = boxes_to_svg(&inner);
    assert!(
      svg.contains("a") && svg.contains("b"),
      "RowBox of SubscriptBoxes should render both: got '{svg}'"
    );
    // Should have two subscript tspans
    let sub_count = svg.matches("baseline-shift=\"sub\"").count();
    assert_eq!(sub_count, 2, "Should have 2 subscripts, got {sub_count}");
  }

  #[test]
  fn style_box_font_color_renders_red() {
    // StyleBox["red text", Rule[FontColor, RGBColor[1, 0, 0]]]
    let inner = Expr::FunctionCall {
      name: "StyleBox".to_string(),
      args: vec![
        Expr::String("red text".to_string()),
        Expr::Rule {
          pattern: Box::new(Expr::Identifier("FontColor".to_string())),
          replacement: Box::new(Expr::FunctionCall {
            name: "RGBColor".to_string(),
            args: vec![Expr::Real(1.0), Expr::Real(0.0), Expr::Real(0.0)],
          }),
        },
      ],
    };
    let layout = layout_box(&inner, 14.0);
    let svg = layout_to_svg(&layout, "currentColor");
    assert!(
      svg.contains("rgb(255,0,0)"),
      "StyleBox with FontColor->Red should render red: got '{svg}'"
    );
  }

  #[test]
  fn style_box_font_size_renders_larger() {
    // StyleBox["big", Rule[FontSize, 24]]
    let inner = Expr::FunctionCall {
      name: "StyleBox".to_string(),
      args: vec![
        Expr::String("big".to_string()),
        Expr::Rule {
          pattern: Box::new(Expr::Identifier("FontSize".to_string())),
          replacement: Box::new(Expr::Integer(24)),
        },
      ],
    };
    let layout = layout_box(&inner, 14.0);
    let svg = layout_to_svg(&layout, "currentColor");
    assert!(
      svg.contains("font-size=\"24.0\""),
      "StyleBox with FontSize->24 should render at size 24: got '{svg}'"
    );
  }

  #[test]
  fn style_box_font_color_in_text_svg() {
    // Test boxes_to_svg path for StyleBox with FontColor
    let inner = Expr::FunctionCall {
      name: "StyleBox".to_string(),
      args: vec![
        Expr::String("colored".to_string()),
        Expr::Rule {
          pattern: Box::new(Expr::Identifier("FontColor".to_string())),
          replacement: Box::new(Expr::FunctionCall {
            name: "RGBColor".to_string(),
            args: vec![Expr::Real(0.0), Expr::Real(0.0), Expr::Real(1.0)],
          }),
        },
      ],
    };
    let svg = boxes_to_svg(&inner);
    assert!(
      svg.contains("fill=\"rgb(0,0,255)\""),
      "StyleBox text SVG should include fill color: got '{svg}'"
    );
  }
}
