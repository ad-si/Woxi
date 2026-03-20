use super::*;
use num_bigint::BigInt;
use proptest::prelude::*;

/// Strategy that generates large integers (up to ~100 digits).
fn big_int() -> impl Strategy<Value = BigInt> {
  prop::collection::vec(0u8..10, 1..100).prop_flat_map(|digits| {
    let magnitude: BigInt =
      digits.iter().fold(BigInt::from(0), |acc, &d| acc * 10 + d);
    prop::bool::ANY.prop_map(move |neg| {
      if neg && magnitude != BigInt::from(0) {
        -magnitude.clone()
      } else {
        magnitude.clone()
      }
    })
  })
}

/// Strategy that generates symbolic variable names.
fn symbol() -> impl Strategy<Value = String> {
  prop::sample::select(vec!["x", "y", "z", "a", "b", "c", "w", "t", "u", "v"])
    .prop_map(|s| s.to_string())
}

/// Strategy that generates symbolic expressions of bounded depth.
fn symbolic_expr(depth: u32) -> impl Strategy<Value = String> {
  let leaf =
    prop_oneof![(-1000i64..1000).prop_map(|n| n.to_string()), symbol(),];
  if depth == 0 {
    return leaf.boxed();
  }
  prop_oneof![
    leaf,
    // n * var
    (1i64..20, symbol()).prop_map(|(n, v)| format!("{n}*{v}")),
    // var^n
    (symbol(), 2i64..5).prop_map(|(v, n)| format!("{v}^{n}")),
    // f[var]
    (
      prop::sample::select(vec!["Sin", "Cos", "Log", "Exp"]),
      symbol(),
    )
      .prop_map(|(f, v)| format!("{f}[{v}]")),
    // (a + b)
    (symbolic_expr(depth - 1), symbolic_expr(depth - 1))
      .prop_map(|(a, b)| format!("({a} + {b})")),
  ]
  .boxed()
}

mod plus {
  use super::*;

  fn eval(code: &str) -> String {
    interpret(code).unwrap()
  }

  proptest! {
    // --- Integer properties ---

    #[test]
    fn identity(a in big_int()) {
      prop_assert_eq!(eval(&format!("{a} + 0")), a.to_string());
      prop_assert_eq!(eval(&format!("0 + {a}")), a.to_string());
    }

    #[test]
    fn commutativity(a in big_int(), b in big_int()) {
      prop_assert_eq!(eval(&format!("{a} + {b}")), eval(&format!("{b} + {a}")));
    }

    #[test]
    fn associativity(a in big_int(), b in big_int(), c in big_int()) {
      prop_assert_eq!(
        eval(&format!("({a} + {b}) + {c}")),
        eval(&format!("{a} + ({b} + {c})")),
      );
    }

    #[test]
    fn inverse(a in big_int()) {
      let neg = -&a;
      prop_assert_eq!(eval(&format!("{a} + ({neg})")), "0".to_string());
    }

    #[test]
    fn integer_result(a in big_int(), b in big_int()) {
      let result = eval(&format!("{a} + {b}"));
      let expected = (&a + &b).to_string();
      prop_assert_eq!(result, expected);
    }

    #[test]
    fn n_ary(a in big_int(), b in big_int(), c in big_int()) {
      let result = eval(&format!("Plus[{a}, {b}, {c}]"));
      let expected = (&a + &b + &c).to_string();
      prop_assert_eq!(result, expected);
    }

    #[test]
    fn nullary_returns_zero(dummy in 0..1i32) {
      let _ = dummy;
      prop_assert_eq!(eval("Plus[]"), "0");
    }

    #[test]
    fn unary(a in big_int()) {
      prop_assert_eq!(eval(&format!("Plus[{a}]")), a.to_string());
    }

    // --- Float properties ---

    #[test]
    fn float_commutativity(a in -1e15f64..1e15, b in -1e15f64..1e15) {
      prop_assert_eq!(
        eval(&format!("{a} + {b}")),
        eval(&format!("{b} + {a}")),
      );
    }

    // --- Rational properties ---

    #[test]
    fn rational_stays_exact(
      a_num in -10_000i64..10_000,
      a_den in 1i64..10_000,
      b_num in -10_000i64..10_000,
      b_den in 1i64..10_000,
    ) {
      let result = eval(&format!("{a_num}/{a_den} + {b_num}/{b_den}"));
      prop_assert!(
        !result.contains('.'),
        "Rational + Rational produced a float: {result}",
      );
    }

    // --- Symbolic properties ---

    #[test]
    fn like_term_collection(a in big_int(), b in big_int()) {
      let expected_coeff = &a + &b;
      let expected = if expected_coeff == BigInt::from(0) {
        "0".to_string()
      } else if expected_coeff == BigInt::from(1) {
        "x".to_string()
      } else if expected_coeff == BigInt::from(-1) {
        "-x".to_string()
      } else {
        format!("{expected_coeff}*x")
      };
      prop_assert_eq!(eval(&format!("{a}*x + {b}*x")), expected);
    }

    #[test]
    fn symbolic_commutativity(
      a in symbolic_expr(1),
      b in symbolic_expr(1),
    ) {
      prop_assert_eq!(eval(&format!("{a} + {b}")), eval(&format!("{b} + {a}")));
    }

    #[test]
    fn symbolic_associativity(
      a in symbolic_expr(1),
      b in symbolic_expr(1),
      c in symbolic_expr(1),
    ) {
      prop_assert_eq!(
        eval(&format!("({a} + {b}) + {c}")),
        eval(&format!("{a} + ({b} + {c})")),
      );
    }

    #[test]
    fn symbolic_identity(a in symbolic_expr(1)) {
      let result_with_zero = eval(&format!("{a} + 0"));
      let result_alone = eval(&format!("{a}"));
      prop_assert_eq!(result_with_zero, result_alone);
    }

    #[test]
    fn symbolic_self_cancel(v in symbol()) {
      // v - v == 0
      prop_assert_eq!(eval(&format!("{v} - {v}")), "0");
    }

    #[test]
    fn symbolic_double(v in symbol()) {
      // v + v == 2*v
      prop_assert_eq!(eval(&format!("{v} + {v}")), format!("2*{v}"));
    }

    #[test]
    fn symbolic_n_copies(v in symbol(), n in 2u32..20) {
      // v + v + ... + v (n times) == n*v
      let terms: Vec<_> = std::iter::repeat_n(&v, n as usize).collect();
      let expr = terms.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(" + ");
      prop_assert_eq!(eval(&expr), format!("{n}*{v}"));
    }

    #[test]
    fn flattening(a in big_int(), b in big_int(), c in big_int()) {
      // Plus[Plus[a, b], c] == Plus[a, b, c]
      prop_assert_eq!(
        eval(&format!("Plus[Plus[{a}, {b}], {c}]")),
        eval(&format!("Plus[{a}, {b}, {c}]")),
      );
    }

    #[test]
    fn mixed_symbolic_numeric_commutativity(
      n in -1000i64..1000,
      v in symbol(),
    ) {
      prop_assert_eq!(
        eval(&format!("{n} + {v}")),
        eval(&format!("{v} + {n}")),
      );
    }
  }
}
