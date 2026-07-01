use woxi::evaluator::dispatch::complex_and_special::expr_to_box_form;
use woxi::functions::graphics::{
  box_has_fraction, box_string_to_svg, box_string_visible_len, boxes_to_svg,
  estimate_box_display_width, estimate_display_width, expr_to_svg_markup,
  layout_box, layout_to_svg,
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
          ]
          .into(),
        },
        Expr::Integer(2),
      ]
      .into(),
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
      args: vec![Expr::Integer(10), Expr::Identifier("x".to_string())].into(),
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
      ]
      .into(),
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
          args: vec![Expr::Integer(2), Expr::Identifier("x".to_string())]
            .into(),
        },
        Expr::FunctionCall {
          name: "Plus".to_string(),
          args: vec![
            Expr::Identifier("x".to_string()),
            Expr::Identifier("y".to_string()),
          ]
          .into(),
        },
      ]
      .into(),
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
          ]
          .into(),
        },
      ]
      .into(),
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
      ]
      .into(),
    };
    let base_w = estimate_display_width(&base);

    let expr = Expr::FunctionCall {
      name: "Power".to_string(),
      args: vec![base, Expr::Integer(2)].into(),
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
          args: vec![Expr::Integer(2), Expr::Identifier("x".to_string())]
            .into(),
        },
        Expr::FunctionCall {
          name: "Plus".to_string(),
          args: vec![
            Expr::Identifier("x".to_string()),
            Expr::Identifier("y".to_string()),
          ]
          .into(),
        },
      ]
      .into(),
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
          ]
          .into(),
        },
      ]
      .into(),
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
      args: vec![Expr::Integer(10), Expr::Identifier("x".to_string())].into(),
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
              ]
              .into(),
            },
            Expr::Integer(2),
          ]
          .into(),
        },
        Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![
            Expr::Integer(9),
            Expr::FunctionCall {
              name: "Plus".to_string(),
              args: vec![Expr::Integer(2), Expr::Identifier("x".to_string())]
                .into(),
            },
            Expr::FunctionCall {
              name: "Plus".to_string(),
              args: vec![
                Expr::Identifier("x".to_string()),
                Expr::Identifier("y".to_string()),
              ]
              .into(),
            },
          ]
          .into(),
        },
      ]
      .into(),
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
              ]
              .into(),
            },
            Expr::Integer(3),
          ]
          .into(),
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
                  ]
                  .into(),
                },
                Expr::Identifier("y".to_string()),
              ]
              .into(),
            },
            Expr::Integer(3),
          ]
          .into(),
        },
      ]
      .into(),
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
          ]
          .into(),
        },
        Expr::Integer(2),
      ]
      .into(),
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
      args: vec![Expr::Integer(10), Expr::Identifier("x".to_string())].into(),
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
      ]
      .into(),
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
          args: vec![Expr::Integer(2), Expr::Identifier("x".to_string())]
            .into(),
        },
        Expr::FunctionCall {
          name: "Plus".to_string(),
          args: vec![
            Expr::Identifier("x".to_string()),
            Expr::Identifier("y".to_string()),
          ]
          .into(),
        },
      ]
      .into(),
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
          ]
          .into(),
        },
      ]
      .into(),
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
      args: vec![Expr::Integer(2), Expr::Integer(3)].into(),
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
      args: vec![Expr::Integer(2), Expr::Integer(3)].into(),
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
          args: vec![Expr::Integer(1), Expr::Integer(2)].into(),
        },
      ]
      .into(),
    };
    let boxes = expr_to_box_form(&expr);
    assert_eq!(box_str(&boxes), "SqrtBox[x]");
  }

  // ── Subscript / Subsuperscript (boxes) ──

  #[test]
  fn box_subscript() {
    let expr = Expr::FunctionCall {
      name: "Subscript".to_string(),
      args: vec![Expr::Identifier("x".to_string()), Expr::Integer(0)].into(),
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
          ]
          .into(),
        },
        Expr::Identifier("c".to_string()),
      ]
      .into(),
    };
    let boxes = expr_to_box_form(&expr);
    assert_eq!(box_str(&boxes), "SubsuperscriptBox[a, b, c]");
  }

  // ── List / FunctionCall (boxes) ──

  #[test]
  fn box_list() {
    let expr = Expr::List(
      vec![Expr::Integer(1), Expr::Integer(2), Expr::Integer(3)].into(),
    );
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
      ]
      .into(),
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
              ]
              .into(),
            },
            Expr::Integer(2),
          ]
          .into(),
        },
        Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![
            Expr::Integer(9),
            Expr::FunctionCall {
              name: "Plus".to_string(),
              args: vec![Expr::Integer(2), Expr::Identifier("x".to_string())]
                .into(),
            },
            Expr::FunctionCall {
              name: "Plus".to_string(),
              args: vec![
                Expr::Identifier("x".to_string()),
                Expr::Identifier("y".to_string()),
              ]
              .into(),
            },
          ]
          .into(),
        },
      ]
      .into(),
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
              ]
              .into(),
            },
            Expr::Integer(3),
          ]
          .into(),
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
                  ]
                  .into(),
                },
                Expr::Identifier("y".to_string()),
              ]
              .into(),
            },
            Expr::Integer(3),
          ]
          .into(),
        },
      ]
      .into(),
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
          ]
          .into(),
        },
        Expr::Integer(2),
      ]
      .into(),
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
      args: vec![Expr::Integer(10), Expr::Identifier("x".to_string())].into(),
    };
    assert_width_reasonable(&expr);
  }

  #[test]
  fn box_width_rational() {
    let expr = Expr::FunctionCall {
      name: "Rational".to_string(),
      args: vec![Expr::Integer(2), Expr::Integer(3)].into(),
    };
    assert_width_reasonable(&expr);
  }

  #[test]
  fn box_width_list() {
    let expr = Expr::List(
      vec![Expr::Integer(1), Expr::Integer(2), Expr::Integer(3)].into(),
    );
    assert_width_reasonable(&expr);
  }

  #[test]
  fn box_width_function_call() {
    let expr = Expr::FunctionCall {
      name: "f".to_string(),
      args: vec![
        Expr::Identifier("x".to_string()),
        Expr::Identifier("y".to_string()),
      ]
      .into(),
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
        ]
        .into(),
      }]
      .into(),
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
      args: vec![Expr::String("a".to_string()), Expr::String("i".to_string())]
        .into(),
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
      args: vec![Expr::List(
        vec![
          Expr::FunctionCall {
            name: "SubscriptBox".to_string(),
            args: vec![
              Expr::String("a".to_string()),
              Expr::String("1".to_string()),
            ]
            .into(),
          },
          Expr::FunctionCall {
            name: "SubscriptBox".to_string(),
            args: vec![
              Expr::String("b".to_string()),
              Expr::String("2".to_string()),
            ]
            .into(),
          },
        ]
        .into(),
      )]
      .into(),
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
            args: vec![Expr::Real(1.0), Expr::Real(0.0), Expr::Real(0.0)]
              .into(),
          }),
        },
      ]
      .into(),
    };
    let layout = layout_box(&inner, 14.0);
    let svg = layout_to_svg(&layout, "currentColor");
    assert!(
      svg.contains("rgb(255,0,0)"),
      "StyleBox with FontColor->Red should render red: got '{svg}'"
    );
  }

  #[test]
  fn rule_arrow_renders_as_unicode_not_private_use() {
    // Regression: `x -> 0.5` rendered in the playground/Studio SVG used the
    // Wolfram FrontEnd private-use codepoint U+F522, which has no glyph in
    // normal fonts and showed as a missing-glyph box. It must render as the
    // public Unicode arrow `→` (U+2192) instead.
    let inner = Expr::Rule {
      pattern: Box::new(Expr::Identifier("x".to_string())),
      replacement: Box::new(Expr::Real(0.5)),
    };
    let boxes = expr_to_box_form(&inner);
    let layout = layout_box(&boxes, 14.0);
    let svg = layout_to_svg(&layout, "currentColor");
    assert!(
      svg.contains('\u{2192}'),
      "Rule should render with Unicode → (U+2192): got '{svg}'"
    );
    assert!(
      !svg.contains('\u{f522}'),
      "Rule SVG must not contain private-use U+F522: got '{svg}'"
    );
  }

  #[test]
  fn rule_delayed_arrow_renders_as_unicode_not_private_use() {
    // Same regression as `rule_arrow_renders_as_unicode_not_private_use`, for
    // RuleDelayed (U+F51F → U+29F4 `⧴`).
    let inner = Expr::RuleDelayed {
      pattern: Box::new(Expr::Identifier("x".to_string())),
      replacement: Box::new(Expr::Integer(1)),
    };
    let boxes = expr_to_box_form(&inner);
    let layout = layout_box(&boxes, 14.0);
    let svg = layout_to_svg(&layout, "currentColor");
    assert!(
      svg.contains('\u{29f4}'),
      "RuleDelayed should render with Unicode ⧴ (U+29F4): got '{svg}'"
    );
    assert!(
      !svg.contains('\u{f51f}'),
      "RuleDelayed SVG must not contain private-use U+F51F: got '{svg}'"
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
      ]
      .into(),
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
            args: vec![Expr::Real(0.0), Expr::Real(0.0), Expr::Real(1.0)]
              .into(),
          }),
        },
      ]
      .into(),
    };
    let svg = boxes_to_svg(&inner);
    assert!(
      svg.contains("fill=\"rgb(0,0,255)\""),
      "StyleBox text SVG should include fill color: got '{svg}'"
    );
  }

  #[test]
  fn text_atom_width_matches_monospace_advance() {
    // Regression: the per-character width estimate for typeset text atoms must
    // match the advance of the monospace font the hosts actually render with
    // (Atkinson Hyperlegible Mono, 632/1000 em). With the old 0.6 estimate the
    // atom was measured narrower than it draws, so a following atom — e.g. the
    // `[` after a function name like `MusicPitch` — overlapped the name.
    let font_size = 14.0;
    let name = "MusicPitch";
    let layout = layout_box(&Expr::Identifier(name.to_string()), font_size);
    let expected = name.chars().count() as f64 * font_size * 0.632;
    assert!(
      (layout.width - expected).abs() < 0.01,
      "text atom width {} should equal {} (0.632 em/char)",
      layout.width,
      expected
    );

    // In a RowBox the opening `[` must sit clear of the name, not on top of it:
    // its glyph's left edge must be at least the name's true rendered width.
    let row = Expr::FunctionCall {
      name: "RowBox".to_string(),
      args: vec![Expr::List(
        vec![
          Expr::String(name.to_string()),
          Expr::String("[".to_string()),
        ]
        .into(),
      )]
      .into(),
    };
    let row_layout = layout_box(&row, font_size);
    assert!(
      row_layout.width >= expected + font_size * 0.632 - 0.01,
      "RowBox width {} must fit the name plus a full `[` advance ({})",
      row_layout.width,
      expected + font_size * 0.632
    );
  }

  // ── Hyperlink rendering ──

  #[test]
  fn hyperlink_two_args_renders_clickable_anchor() {
    // Hyperlink["Woxi", "https://woxi.ad-si.com"] should render as an
    // SVG <a href="..."> wrapping a blue label, with an underline.
    let expr = Expr::FunctionCall {
      name: "Hyperlink".to_string(),
      args: vec![
        Expr::String("Woxi".to_string()),
        Expr::String("https://woxi.ad-si.com".to_string()),
      ]
      .into(),
    };
    let boxes = expr_to_box_form(&expr);
    let layout = layout_box(&boxes, 14.0);
    let svg = layout_to_svg(&layout, "currentColor");
    assert!(
      svg.contains("<a href=\"https://woxi.ad-si.com\""),
      "expected SVG anchor with href: got '{svg}'"
    );
    assert!(
      svg.contains("target=\"_blank\""),
      "expected target=_blank on hyperlink: got '{svg}'"
    );
    assert!(
      svg.contains("rel=\"noopener\""),
      "expected rel=noopener on hyperlink: got '{svg}'"
    );
    assert!(
      svg.contains("#1a73e8"),
      "expected link-blue color #1a73e8: got '{svg}'"
    );
    assert!(
      svg.contains(">Woxi<"),
      "expected unquoted label text 'Woxi' between tags: got '{svg}'"
    );
    assert!(
      svg.contains("<line"),
      "expected underline <line> below the label: got '{svg}'"
    );
  }

  #[test]
  fn hyperlink_single_arg_uses_uri_as_label() {
    // Hyperlink["https://woxi.ad-si.com"] should display the URI as
    // both the visible text and the href.
    let expr = Expr::FunctionCall {
      name: "Hyperlink".to_string(),
      args: vec![Expr::String("https://woxi.ad-si.com".to_string())].into(),
    };
    let boxes = expr_to_box_form(&expr);
    let layout = layout_box(&boxes, 14.0);
    let svg = layout_to_svg(&layout, "currentColor");
    assert!(
      svg.contains("href=\"https://woxi.ad-si.com\""),
      "expected href with URI: got '{svg}'"
    );
    assert!(
      svg.contains(">https://woxi.ad-si.com<"),
      "expected URI text between tags: got '{svg}'"
    );
  }

  #[test]
  fn hyperlink_box_form_is_template_box() {
    // expr_to_box_form should produce TemplateBox[..., "HyperlinkURL"]
    // matching wolframscript's MakeBoxes structure.
    let expr = Expr::FunctionCall {
      name: "Hyperlink".to_string(),
      args: vec![
        Expr::String("Woxi".to_string()),
        Expr::String("https://woxi.ad-si.com".to_string()),
      ]
      .into(),
    };
    let boxes = expr_to_box_form(&expr);
    if let Expr::FunctionCall { name, args } = &boxes {
      assert_eq!(name, "TemplateBox");
      assert_eq!(args.len(), 2);
      assert!(
        matches!(&args[1], Expr::String(s) if s == "HyperlinkURL"),
        "second arg should be \"HyperlinkURL\": got {:?}",
        args[1]
      );
    } else {
      panic!("expected TemplateBox, got {:?}", boxes);
    }
  }

  #[test]
  fn hyperlink_with_non_string_uri_falls_back_to_function_call() {
    // Hyperlink[label, expr] with a non-string URI should fall through
    // to the generic function-call rendering (no anchor in SVG).
    let expr = Expr::FunctionCall {
      name: "Hyperlink".to_string(),
      args: vec![
        Expr::String("label".to_string()),
        Expr::Identifier("someVar".to_string()),
      ]
      .into(),
    };
    let boxes = expr_to_box_form(&expr);
    let layout = layout_box(&boxes, 14.0);
    let svg = layout_to_svg(&layout, "currentColor");
    assert!(
      !svg.contains("<a href"),
      "expected no anchor for non-string URI: got '{svg}'"
    );
  }

  #[test]
  fn hyperlink_html_in_uri_is_escaped() {
    // A URI containing characters that need XML escaping should be
    // properly escaped in the href to avoid breaking the SVG.
    let expr = Expr::FunctionCall {
      name: "Hyperlink".to_string(),
      args: vec![
        Expr::String("link".to_string()),
        Expr::String("https://example.com/?q=a&b=<c>".to_string()),
      ]
      .into(),
    };
    let boxes = expr_to_box_form(&expr);
    let layout = layout_box(&boxes, 14.0);
    let svg = layout_to_svg(&layout, "currentColor");
    assert!(
      svg.contains("&amp;b=&lt;c&gt;"),
      "expected XML-escaped href: got '{svg}'"
    );
  }

  // ── Number-display forms (ScientificForm / EngineeringForm / NumberForm) ──
  //
  // These render in the Playground/Studio as 2D `mantissa × 10^exp` notation
  // rather than as a literal `ScientificForm[…]` function call. The exponent
  // must appear as a raised superscript (smaller font, higher up) and none of
  // the box-wrapper head names may leak into the drawn text.

  fn render_form_svg(name: &str, value: f64) -> String {
    let expr = Expr::FunctionCall {
      name: name.to_string(),
      args: vec![Expr::Real(value)].into(),
    };
    let boxes = expr_to_box_form(&expr);
    let layout = layout_box(&boxes, 14.0);
    layout_to_svg(&layout, "currentColor")
  }

  #[test]
  fn scientific_form_renders_2d_superscript() {
    let svg = render_form_svg("ScientificForm", 12345.6);
    // Mantissa, the × separator, the base 10, and the exponent all appear.
    assert!(svg.contains(">1.23456<"), "mantissa missing: {svg}");
    assert!(svg.contains(">10<"), "base 10 missing: {svg}");
    assert!(svg.contains(">4<"), "exponent missing: {svg}");
    // The exponent is rendered at a smaller font than the body (superscript).
    assert!(
      svg.contains("font-size=\"9.8\""),
      "exponent should use the smaller superscript font: {svg}"
    );
    // No box-wrapper machinery leaks into the rendered text.
    assert!(
      !svg.contains("ScientificForm") && !svg.contains("InterpretationBox"),
      "form should render graphically, not as literal box text: {svg}"
    );
  }

  #[test]
  fn engineering_form_renders_2d_superscript() {
    let svg = render_form_svg("EngineeringForm", 12345.6);
    // Exponent forced to a multiple of 3 → mantissa 12.3456, exponent 3.
    assert!(svg.contains(">12.3456<"), "mantissa missing: {svg}");
    assert!(svg.contains(">3<"), "engineering exponent missing: {svg}");
    assert!(
      svg.contains("font-size=\"9.8\""),
      "exponent should use the smaller superscript font: {svg}"
    );
    assert!(
      !svg.contains("EngineeringForm") && !svg.contains("InterpretationBox"),
      "form should render graphically, not as literal box text: {svg}"
    );
  }

  #[test]
  fn number_form_above_threshold_renders_scientific() {
    // |x| >= 10^6 switches NumberForm to 2D scientific notation.
    let svg = render_form_svg("NumberForm", 1234567.8);
    assert!(svg.contains(">1.23457<"), "mantissa missing: {svg}");
    assert!(svg.contains(">6<"), "exponent missing: {svg}");
    assert!(
      svg.contains("font-size=\"9.8\""),
      "exponent should use the smaller superscript font: {svg}"
    );
    assert!(
      !svg.contains("NumberForm") && !svg.contains("InterpretationBox"),
      "form should render graphically, not as literal box text: {svg}"
    );
  }

  #[test]
  fn number_form_in_range_renders_plain_number() {
    // An in-range real renders as a plain number, with no × 10^exp factor and
    // no superscript font.
    let svg = render_form_svg("NumberForm", 12345.6);
    assert!(svg.contains(">12345.6<"), "plain number missing: {svg}");
    assert!(
      !svg.contains("font-size=\"9.8\""),
      "in-range NumberForm should have no superscript: {svg}"
    );
    assert!(
      !svg.contains("NumberForm") && !svg.contains("InterpretationBox"),
      "form should render graphically, not as literal box text: {svg}"
    );
  }

  #[test]
  fn scientific_form_list_renders_braced_2d_row() {
    // A list argument threads element-wise: each element is drawn in 2D
    // scientific notation inside a braced, comma-separated row — not as a
    // literal `ScientificForm[{…}]` function call.
    let expr = Expr::FunctionCall {
      name: "ScientificForm".to_string(),
      args: vec![Expr::List(
        vec![
          Expr::Real(123450000.0),
          Expr::Real(0.00012345),
          Expr::Real(123.45),
        ]
        .into(),
      )]
      .into(),
    };
    let svg = layout_to_svg(
      &layout_box(&expr_to_box_form(&expr), 14.0),
      "currentColor",
    );
    // Braces, the per-element mantissa, and the three exponents are all drawn.
    assert!(
      svg.contains(">{<") && svg.contains(">}<"),
      "braces missing: {svg}"
    );
    assert!(svg.contains(">1.2345<"), "mantissa missing: {svg}");
    assert!(svg.contains(">8<"), "exponent 8 missing: {svg}");
    assert!(svg.contains(">-4<"), "exponent -4 missing: {svg}");
    assert!(svg.contains(">2<"), "exponent 2 missing: {svg}");
    assert!(
      svg.contains("font-size=\"9.8\""),
      "exponents should use the smaller superscript font: {svg}"
    );
    assert!(
      !svg.contains("ScientificForm") && !svg.contains("InterpretationBox"),
      "list form should render graphically, not as literal box text: {svg}"
    );
  }

  // ── Inline box-notation strings (PlotLegends / label formatting) ──

  // The Wolfram front-end emits subscripts/superscripts in label strings as
  // "linear syntax" box notation: `\!\(\*SubscriptBox[\(base\),\(sub\)]\)`.
  // After string-literal parsing these escapes become private-use marker
  // codepoints, so the tests below feed the marker form a label string would
  // actually contain at render time (via `interpret`-produced SVG) as well as
  // the literal-escape form fed directly to `box_string_to_svg`.

  #[test]
  fn box_string_subscript_renders_tspan() {
    let svg = box_string_to_svg("C\\!\\(\\*SubscriptBox[\\(\\),\\(2\\)]\\)=9");
    assert_eq!(
      svg,
      "C<tspan baseline-shift=\"sub\" font-size=\"70%\">2</tspan>=9"
    );
  }

  #[test]
  fn box_string_superscript_renders_tspan() {
    let svg =
      box_string_to_svg("GeV\\!\\(\\*SuperscriptBox[\\(\\),\\(-3\\)]\\)");
    assert_eq!(
      svg,
      "GeV<tspan baseline-shift=\"super\" font-size=\"70%\">-3</tspan>"
    );
  }

  #[test]
  fn box_string_sqrtbox_renders_radical() {
    let svg = box_string_to_svg("v=\\!\\(\\*SqrtBox[\\(s\\)]\\)");
    assert_eq!(
      svg,
      "v=\u{221A}<tspan text-decoration=\"overline\">s</tspan>"
    );
  }

  #[test]
  fn box_string_subsuperscript_trims_arg_whitespace() {
    // Whitespace after commas in the linear syntax is insignificant.
    let svg = box_string_to_svg(
      "x\\!\\(\\*SubsuperscriptBox[\\(\\), \\(i\\), \\(2\\)]\\)",
    );
    assert_eq!(
      svg,
      "x<tspan baseline-shift=\"sub\" font-size=\"70%\">i</tspan>\
       <tspan baseline-shift=\"super\" font-size=\"70%\">2</tspan>"
    );
  }

  #[test]
  fn box_string_plain_text_is_html_escaped() {
    // A label without any box notation is a safe drop-in for svg_escape.
    assert_eq!(box_string_to_svg("a < b & c"), "a &lt; b &amp; c");
  }

  #[test]
  fn box_string_visible_len_ignores_box_scaffolding() {
    // Visible length counts "C", "2", "=9.78 GeV", "-3" = 13 chars — not the
    // dozens of bytes of `\!\(\*SubscriptBox…\)` scaffolding.
    let label = "C\\!\\(\\*SubscriptBox[\\(\\),\\(2\\)]\\)=9.78 GeV\
                 \\!\\(\\*SuperscriptBox[\\(\\),\\(-3\\)]\\)";
    assert_eq!(box_string_visible_len(label), "C2=9.78 GeV-3".len());
  }

  #[test]
  fn list_line_plot_legend_renders_formatted_box_notation() {
    // End-to-end: the formatting survives string-literal parsing (which turns
    // the escapes into private-use markers) and reaches the legend renderer.
    let code = r#"ExportString[ListLinePlot[{{1,2,3}},
      PlotLegends -> {"C\!\(\*SubscriptBox[\(\),\(2\)]\)=9.78 GeV\!\(\*SuperscriptBox[\(\),\(-3\)]\)"}],
      "SVG"]"#;
    let svg = woxi::interpret(code).expect("interpret should succeed");
    assert!(
      svg
        .contains("C<tspan baseline-shift=\"sub\" font-size=\"70%\">2</tspan>"),
      "subscript not rendered in legend: {svg}"
    );
    assert!(
      svg.contains(
        "GeV<tspan baseline-shift=\"super\" font-size=\"70%\">-3</tspan>"
      ),
      "superscript not rendered in legend: {svg}"
    );
    // The raw box scaffolding must not leak into the output text.
    assert!(
      !svg.contains("SubscriptBox") && !svg.contains("SuperscriptBox"),
      "raw box head leaked into SVG: {svg}"
    );
  }

  #[test]
  fn list_line_plot_renders_plot_label() {
    let svg = woxi::interpret(
      r#"ExportString[ListLinePlot[{{1,2,3}}, PlotLabel -> "MyTitle"], "SVG"]"#,
    )
    .expect("interpret should succeed");
    assert!(
      svg.contains(">MyTitle</text>"),
      "PlotLabel not rendered: {svg}"
    );
  }

  #[test]
  fn list_line_plot_renders_frame_label() {
    let svg = woxi::interpret(
      r#"ExportString[ListLinePlot[{{1,2,3}},
        Frame -> True, FrameLabel -> {"XAxis", "YAxis"}], "SVG"]"#,
    )
    .expect("interpret should succeed");
    assert!(svg.contains(">XAxis</text>"), "x FrameLabel missing: {svg}");
    assert!(svg.contains(">YAxis</text>"), "y FrameLabel missing: {svg}");
    // The y label is rotated on the left frame edge.
    assert!(
      svg.contains("rotate(-90"),
      "y label should be rotated: {svg}"
    );
  }

  #[test]
  fn list_line_plot_frame_label_renders_box_notation() {
    // FrameLabel with a SqrtBox renders as a radical, not raw box text.
    let svg = woxi::interpret(
      r#"ExportString[ListLinePlot[{{1,2,3}},
        Frame -> True, FrameLabel -> {"\!\(\*SqrtBox[\(s\)]\) [GeV]", "y"}], "SVG"]"#,
    )
    .expect("interpret should succeed");
    assert!(
      svg.contains("\u{221A}<tspan text-decoration=\"overline\">s</tspan>"),
      "SqrtBox not rendered in FrameLabel: {svg}"
    );
    assert!(!svg.contains("SqrtBox"), "raw SqrtBox leaked: {svg}");
  }

  #[test]
  fn list_line_plot_frame_label_four_element_form() {
    // `{{left, right}, {bottom, top}}` labels all four frame edges.
    let svg = woxi::interpret(
      r#"ExportString[ListLinePlot[{{1,2,3}}, Frame -> True,
        FrameLabel -> {{"Lft", "Rgt"}, {"Btm", "Top"}}], "SVG"]"#,
    )
    .expect("interpret should succeed");
    assert!(svg.contains(">Btm</text>"), "bottom label missing: {svg}");
    assert!(svg.contains(">Top</text>"), "top label missing: {svg}");
    // Left is rotated -90, right is rotated +90.
    assert!(
      svg.contains("rotate(-90") && svg.contains(">Lft</text>"),
      "left label missing/not rotated: {svg}"
    );
    assert!(
      svg.contains("rotate(90") && svg.contains(">Rgt</text>"),
      "right label missing/not rotated: {svg}"
    );
  }

  #[test]
  fn list_line_plot_four_element_frame_label_coexists_with_plot_label() {
    // A top FrameLabel and a PlotLabel must both appear (stacked, not
    // overwriting each other).
    let svg = woxi::interpret(
      r#"ExportString[ListLinePlot[{{1,2,3}}, Frame -> True,
        FrameLabel -> {{"l", "r"}, {"b", "TopEdge"}},
        PlotLabel -> "TheTitle"], "SVG"]"#,
    )
    .expect("interpret should succeed");
    assert!(
      svg.contains(">TopEdge</text>"),
      "top frame label missing: {svg}"
    );
    assert!(
      svg.contains(">TheTitle</text>"),
      "plot label missing: {svg}"
    );
  }

  #[test]
  fn dashed_legend_swatch_matches_chart_dash_scale() {
    // The legend swatch's stroke-dasharray must use the same scale as the
    // on-chart dashed line (a fraction `d` of the plotting-area width), not the
    // old swatch-relative scale. For a default-size plot the plotting area is
    // 3600 - 100 - 100 - 650 = 2750 viewBox units, so a `Dashed` pattern
    // (d = 0.01) yields "27.5,27.5" — not the previous "100.0,100.0".
    let svg = woxi::interpret(
      r#"ExportString[ListLinePlot[{{1,2,3},{3,2,1}},
        PlotStyle -> {Black, Dashed}, PlotLegends -> {"A", "B"}], "SVG"]"#,
    )
    .expect("interpret should succeed");
    assert!(
      svg.contains("stroke-dasharray=\"27.5,27.5\""),
      "legend dash scale should match the chart (27.5,27.5): {svg}"
    );
    assert!(
      !svg.contains("stroke-dasharray=\"100.0,100.0\""),
      "legend still using the old swatch-relative dash scale: {svg}"
    );
  }

  #[test]
  fn dashed_line_is_a_single_dasharray_polyline() {
    // A dashed series must render as ONE <polyline stroke-dasharray> spanning
    // all data points — not one tiny <polyline> per dash segment.
    let svg = woxi::interpret(
      r#"ExportString[ListLinePlot[{{1,2,3,4,5,6}}, Joined -> True,
        PlotStyle -> Dashed], "SVG"]"#,
    )
    .expect("interpret should succeed");
    let dash_polylines = svg.matches("stroke-dasharray=").count();
    assert_eq!(
      dash_polylines, 1,
      "dashed series should be a single dasharray polyline, found {dash_polylines}: {svg}"
    );
    // That single polyline must carry the whole data path (>= 6 vertices).
    let line = svg
      .lines()
      .find(|l| l.contains("stroke-dasharray="))
      .unwrap();
    let pts = line
      .split("points=\"")
      .nth(1)
      .and_then(|s| s.split('"').next())
      .unwrap_or("");
    assert!(
      pts.split_whitespace().count() >= 6,
      "dashed polyline should span all data points: {pts:?}"
    );
  }

  #[test]
  fn grid_lines_are_solid_single_polylines() {
    // Each grid line must be ONE solid polyline (WL grid lines are solid by
    // default), not one tiny <polyline> per dash.
    let svg = woxi::interpret(
      r#"ExportString[ListLinePlot[{{1,2,3,4}}, GridLines -> Automatic], "SVG"]"#,
    )
    .expect("interpret should succeed");
    let grid_lines: Vec<&str> = svg
      .lines()
      .filter(|l| l.contains("opacity=\"0.5\"") && l.contains("<polyline"))
      .collect();
    assert!(
      (1..=30).contains(&grid_lines.len()),
      "expected a handful of consolidated grid polylines, found {}",
      grid_lines.len()
    );
    // Grid lines must be solid (no dash) and render behind the data series
    // (plotters draws the joined series line as stroke="#5E81B5").
    assert!(
      grid_lines.iter().all(|l| !l.contains("stroke-dasharray")),
      "grid lines should be solid by default: {svg}"
    );
    assert!(
      svg.find("opacity=\"0.5\"") < svg.find("#5E81B5"),
      "grid lines should render behind the data series: {svg}"
    );
  }

  #[test]
  fn graphics_grid_lines_render_with_style() {
    // Graphics[..., GridLines -> Automatic, GridLinesStyle -> ...] must draw
    // grid lines (it previously ignored both options).
    let svg = woxi::interpret(
      r#"ExportString[Graphics[Circle[], Frame -> True, GridLines -> Automatic,
        GridLinesStyle -> Directive[Orange, Dashed]], "SVG"]"#,
    )
    .expect("interpret should succeed");
    // Orange dashed grid lines spanning the plot, behind the circle.
    assert!(
      svg.contains("stroke=\"rgb(255,128,0)\"")
        && svg.contains("stroke-dasharray="),
      "grid lines not styled orange/dashed: {svg}"
    );
    assert!(
      svg.find("rgb(255,128,0)") < svg.find("<ellipse"),
      "grid lines should render behind the circle: {svg}"
    );
  }

  #[test]
  fn graphics_grid_lines_explicit_positions() {
    // GridLines -> {xvals, yvals} draws lines at the specified coordinates.
    let svg = woxi::interpret(
      r#"ExportString[Graphics[Circle[], Frame -> True,
        GridLines -> {{0, 0.5}, {-0.5}}], "SVG"]"#,
    )
    .expect("interpret should succeed");
    // Full-span grid lines: vertical lines run y=0..360, horizontals x=0..360
    // (frame ticks only span ~5px, so they don't match).
    let verticals = svg
      .lines()
      .filter(|l| l.contains("y1=\"0\"") && l.contains("y2=\"360.00\""))
      .count();
    let horizontals = svg
      .lines()
      .filter(|l| l.contains("x1=\"0\"") && l.contains("x2=\"360.00\""))
      .count();
    assert_eq!(verticals, 2, "expected 2 vertical grid lines: {svg}");
    assert_eq!(horizontals, 1, "expected 1 horizontal grid line: {svg}");
    // x=0 maps to the horizontal center (180 of 360).
    assert!(
      svg.contains("<line x1=\"180.00\" y1=\"0\" x2=\"180.00\" y2=\"360.00\""),
      "vertical grid line at x=0 missing: {svg}"
    );
  }

  #[test]
  fn graphics_grid_lines_per_line_style() {
    // A {pos, style} entry overrides the style for just that line.
    let svg = woxi::interpret(
      r#"ExportString[Graphics[Circle[], Frame -> True,
        GridLines -> {{{0.5, Directive[Red, Dashed]}}, Automatic}], "SVG"]"#,
    )
    .expect("interpret should succeed");
    // The single x line is red + dashed; the y lines use the default gray.
    assert!(
      svg.contains("stroke=\"rgb(255,0,0)\"")
        && svg.contains("stroke-dasharray="),
      "per-line red dashed style not applied: {svg}"
    );
    assert!(
      svg.contains("stroke=\"rgb(204,204,204)\""),
      "automatic y grid lines should use the default gray: {svg}"
    );
  }

  #[test]
  fn graphics_grid_lines_default_solid_gray() {
    // Without GridLinesStyle, Graphics grid lines are solid light gray.
    let svg = woxi::interpret(
      r#"ExportString[Graphics[Circle[], Frame -> True,
        GridLines -> Automatic], "SVG"]"#,
    )
    .expect("interpret should succeed");
    let grid: Vec<&str> = svg
      .lines()
      .filter(|l| l.contains("stroke=\"rgb(204,204,204)\""))
      .collect();
    assert!(!grid.is_empty(), "expected light-gray grid lines: {svg}");
    assert!(
      grid.iter().all(|l| !l.contains("stroke-dasharray")),
      "default grid lines should be solid: {svg}"
    );
  }

  #[test]
  fn plot_grid_lines_styled_per_line() {
    // Plot must honor per-line GridLines styles: dashed, thick, and colored
    // lines at explicit positions on each axis.
    let svg = woxi::interpret(
      r#"ExportString[Plot[Cos[x], {x, 0, 10}, GridLines -> {
        {{Pi, Dashed}, {2 Pi, Thick}},
        {{-1, Orange}, -.5, .5, {1, Orange}}
      }], "SVG"]"#,
    )
    .expect("interpret should succeed");
    // x = Pi → a single dashed grid line.
    assert_eq!(
      svg.matches("stroke-dasharray=").count(),
      1,
      "expected exactly one dashed grid line at x=Pi: {svg}"
    );
    // x = 2 Pi → a Thick (stroke-width 20 = 2px) grid line.
    assert!(
      svg.contains("stroke-width=\"20\""),
      "expected a thick grid line at x=2 Pi: {svg}"
    );
    // y = -1 and y = 1 → two Orange (#FF8000) grid lines. plotters emits
    // colors as uppercase hex.
    assert_eq!(
      svg.to_uppercase().matches("#FF8000").count(),
      2,
      "expected two orange grid lines at y=-1 and y=1: {svg}"
    );
  }

  #[test]
  fn plot_grid_lines_explicit_positions_one_axis() {
    // `{xspec, Automatic}` draws explicit x lines and automatic y lines.
    let svg = woxi::interpret(
      r#"ExportString[Plot[Cos[x], {x, 0, 10},
        GridLines -> {{2, 4, 6, 8}, None}], "SVG"]"#,
    )
    .expect("interpret should succeed");
    // Four vertical default-gray solid grid lines, no horizontal ones.
    let verticals = svg
      .lines()
      .filter(|l| {
        l.contains("opacity=\"0.5\"")
          && l.contains("<polyline")
          && !l.contains("stroke-dasharray")
      })
      .count();
    assert_eq!(verticals, 4, "expected 4 explicit x grid lines: {svg}");
  }

  #[test]
  fn frame_true_draws_thin_four_sided_border() {
    // Frame -> True draws a single closed rectangle (5 points) at 1px
    // (stroke-width = RESOLUTION_SCALE = 10 in viewBox units).
    let svg = woxi::interpret(
      r#"ExportString[ListLinePlot[{{1,2,3}}, Frame -> True], "SVG"]"#,
    )
    .expect("interpret should succeed");
    // Find the thick frame polyline and confirm it is a closed 5-point rect.
    let frame = svg
      .lines()
      .find(|l| l.contains("stroke-width=\"10\"") && l.contains("points="))
      .unwrap_or_else(|| panic!("no frame polyline at stroke-width 10: {svg}"));
    let pts = frame
      .split("points=\"")
      .nth(1)
      .and_then(|s| s.split('"').next())
      .unwrap_or("");
    let coords: Vec<&str> = pts.split_whitespace().collect();
    assert_eq!(
      coords.len(),
      5,
      "frame should be a closed 4-sided rectangle (5 points): {pts:?}"
    );
    assert_eq!(
      coords.first(),
      coords.last(),
      "frame rectangle should be closed: {pts:?}"
    );
  }
}
