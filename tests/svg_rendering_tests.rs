use woxi::functions::graphics::{estimate_display_width, expr_to_svg_markup};
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
