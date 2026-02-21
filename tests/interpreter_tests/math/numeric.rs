use super::*;

mod n_arbitrary_precision {
  use super::*;

  #[test]
  fn n_default_precision_unchanged() {
    // N[expr] with 1 arg should still return f64 precision
    assert_eq!(interpret("N[Pi]").unwrap(), "3.141592653589793");
    assert_eq!(interpret("N[E]").unwrap(), "2.718281828459045");
    assert_eq!(interpret("N[Degree]").unwrap(), "0.017453292519943295");
  }

  #[test]
  fn n_pi_arbitrary_first_digits() {
    // N[Pi, 50] — check that first 50 significant digits are correct
    let result = interpret("N[Pi, 50]").unwrap();
    assert!(
      result
        .starts_with("3.14159265358979323846264338327950288419716939937510")
    );
    assert!(result.ends_with("`50."));
  }

  #[test]
  fn n_e_arbitrary_first_digits() {
    // N[E, 30] — check first 30 significant digits
    let result = interpret("N[E, 30]").unwrap();
    assert!(result.starts_with("2.7182818284590452353602874713"));
    assert!(result.ends_with("`30."));
  }

  #[test]
  fn n_integer_arbitrary() {
    // N[100, 20] should return 100.`20.
    assert_eq!(interpret("N[100, 20]").unwrap(), "100.`20.");
    // N[7, 20] should return 7.`20.
    assert_eq!(interpret("N[7, 20]").unwrap(), "7.`20.");
  }

  #[test]
  fn n_rational_arbitrary() {
    // N[1/3, 20] — should start with 0.3333...
    let result = interpret("N[1/3, 20]").unwrap();
    assert!(result.starts_with("0.3333333333333333333"));
    assert!(result.ends_with("`20."));
  }

  #[test]
  fn n_sqrt_arbitrary() {
    // N[Sqrt[2], 20] — check first 20 digits
    let result = interpret("N[Sqrt[2], 20]").unwrap();
    assert!(result.starts_with("1.414213562373095048801688724"));
    assert!(result.ends_with("`20."));
  }

  #[test]
  fn n_pi_10000_digits() {
    // N[Pi, 10000] — the main todo item
    let result = interpret("N[Pi, 10000]").unwrap();
    // Check the suffix
    assert!(result.ends_with("`10000."));
    // Check first 50 digits of Pi
    assert!(
      result
        .starts_with("3.14159265358979323846264338327950288419716939937510")
    );
    // Check digit count: should have > 10000 sig digits
    let digits_part = result.split('`').next().unwrap();
    let sig_digits: usize =
      digits_part.chars().filter(|c| c.is_ascii_digit()).count();
    assert!(sig_digits >= 10000);
  }

  #[test]
  fn n_pi_100_digits() {
    // N[Pi, 100] — check first 100 digits
    let result = interpret("N[Pi, 100]").unwrap();
    assert!(result.starts_with("3.14159265358979323846264338327950288419716939937510582097494459230781640628620899862803482534211706798214808651"));
    assert!(result.ends_with("`100."));
  }

  #[test]
  fn n_head_is_real() {
    // Head[N[Pi, 50]] should be Real
    assert_eq!(interpret("Head[N[Pi, 50]]").unwrap(), "Real");
  }

  #[test]
  fn n_number_q() {
    // NumberQ[N[Pi, 50]] should be True
    assert_eq!(interpret("NumberQ[N[Pi, 50]]").unwrap(), "True");
  }
}

mod real_precision {
  use super::*;

  #[test]
  fn full_precision_when_needed() {
    // Power[1.5, 2.5] needs full precision
    assert_eq!(interpret("Power[1.5, 2.5]").unwrap(), "2.7556759606310752");
  }

  #[test]
  fn short_precision_when_clean() {
    // Simple addition should round cleanly
    assert_eq!(interpret("1.5 + 2.7").unwrap(), "4.2");
  }
}

mod precision {
  use super::*;

  #[test]
  fn integer() {
    assert_eq!(interpret("Precision[1]").unwrap(), "Infinity");
  }

  #[test]
  fn rational() {
    assert_eq!(interpret("Precision[1/2]").unwrap(), "Infinity");
  }

  #[test]
  fn machine_real() {
    assert_eq!(interpret("Precision[0.5]").unwrap(), "MachinePrecision");
  }
}

mod accuracy {
  use super::*;

  #[test]
  fn integer() {
    assert_eq!(interpret("Accuracy[1]").unwrap(), "Infinity");
  }

  #[test]
  fn symbol() {
    assert_eq!(interpret("Accuracy[A]").unwrap(), "Infinity");
  }

  #[test]
  fn machine_real() {
    // Accuracy[0.5] ≈ 16.2556...
    let result = interpret("Accuracy[0.5]").unwrap();
    let val: f64 = result.parse().unwrap();
    assert!((val - 16.2556).abs() < 0.01);
  }
}

mod exponent {
  use super::*;

  #[test]
  fn basic() {
    assert_eq!(interpret("Exponent[(x^3 + 1)^2 + 1, x]").unwrap(), "6");
  }

  #[test]
  fn zero_expr() {
    assert_eq!(interpret("Exponent[0, x]").unwrap(), "-Infinity");
  }

  #[test]
  fn min_form() {
    assert_eq!(interpret("Exponent[(x^2 + 1)^3 - 1, x, Min]").unwrap(), "2");
  }
}

mod overflow_safety {
  use super::*;

  #[test]
  fn large_product_no_panic() {
    let result = interpret("IntegerLength[Times@@Range[5000]]");
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), "16326");
  }

  #[test]
  fn rationalize_no_panic() {
    let result = interpret("Rationalize[N[Pi], 0]");
    assert!(result.is_ok());
  }

  #[test]
  fn rationalize_zero_tolerance() {
    assert_eq!(
      interpret("Rationalize[N[Pi], 0]").unwrap(),
      "245850922/78256779"
    );
  }
}
