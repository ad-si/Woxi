use woxi::interpret;

mod high_level_functions_tests {
  use super::*;

  mod evenq_tests {
    use super::*;
    #[test]
    fn test_for_negative() {
      assert_eq!(interpret("EvenQ[-2]").unwrap(), "True",);
      assert_eq!(interpret("EvenQ[-1]").unwrap(), "False",);
    }
    #[test]
    fn test_for_zero() {
      assert_eq!(interpret("EvenQ[0]").unwrap(), "True",);
    }
    #[test]
    fn test_for_positive() {
      assert_eq!(interpret("EvenQ[1]").unwrap(), "False",);
      assert_eq!(interpret("EvenQ[2]").unwrap(), "True",);
      assert_eq!(interpret("EvenQ[3]").unwrap(), "False",);
    }
    #[test]
    fn test_for_float() {
      assert_eq!(interpret("EvenQ[1.2]").unwrap(), "False",);
      assert_eq!(interpret("EvenQ[1.3]").unwrap(), "False",);
    }
  }

  mod oddq_tests {
    use super::*;
    #[test]
    fn test_for_negative() {
      assert_eq!(interpret("OddQ[-1]").unwrap(), "True");
    }
    #[test]
    fn test_for_zero() {
      assert_eq!(interpret("OddQ[0]").unwrap(), "False");
    }
    #[test]
    fn test_for_positive() {
      assert_eq!(interpret("OddQ[1]").unwrap(), "True");
      assert_eq!(interpret("OddQ[2]").unwrap(), "False");
    }
    #[test]
    fn test_for_float() {
      assert_eq!(interpret("OddQ[1.2]").unwrap(), "False");
      assert_eq!(interpret("OddQ[1.3]").unwrap(), "False");
    }
  }

  // #[test]
  // fn test_first_function() {
  //   assert_eq!(interpret("First[{1, 2, 3}]").unwrap(), "1");
  //   assert_eq!(interpret("First[{a, b, c}]").unwrap(), "a");
  //   assert_eq!(interpret("First[{True, False, False}]").unwrap(), "True");
  // }

  mod prime_function {
    use super::*;

    #[test]
    fn test_prime_function() {
      assert_eq!(interpret("Prime[1]").unwrap(), "2");
      assert_eq!(interpret("Prime[2]").unwrap(), "3");
      assert_eq!(interpret("Prime[3]").unwrap(), "5");
      assert_eq!(interpret("Prime[4]").unwrap(), "7");
      assert_eq!(interpret("Prime[5]").unwrap(), "11");
      assert_eq!(interpret("Prime[100]").unwrap(), "541");
    }

    #[test]
    fn test_prime_function_invalid_input() {
      assert!(interpret("Prime[0]").is_err());
      assert!(interpret("Prime[-1]").is_err());
      assert!(interpret("Prime[1.5]").is_err());
    }
  }

  // ─── Hyperbolic Trig Functions ────────────────────────────────────
  mod sinh_tests {
    use super::*;
    #[test]
    fn test_sinh_zero() {
      assert_eq!(interpret("Sinh[0]").unwrap(), "0");
    }
    #[test]
    fn test_sinh_real() {
      assert_eq!(interpret("Sinh[1.0]").unwrap(), "1.1752011936438014");
    }
    #[test]
    fn test_sinh_symbolic() {
      assert_eq!(interpret("Sinh[x]").unwrap(), "Sinh[x]");
    }
  }

  mod cosh_tests {
    use super::*;
    #[test]
    fn test_cosh_zero() {
      assert_eq!(interpret("Cosh[0]").unwrap(), "1");
    }
    #[test]
    fn test_cosh_real() {
      assert_eq!(interpret("Cosh[1.0]").unwrap(), "1.5430806348152437");
    }
    #[test]
    fn test_cosh_symbolic() {
      assert_eq!(interpret("Cosh[x]").unwrap(), "Cosh[x]");
    }
  }

  mod tanh_tests {
    use super::*;
    #[test]
    fn test_tanh_zero() {
      assert_eq!(interpret("Tanh[0]").unwrap(), "0");
    }
    #[test]
    fn test_tanh_real() {
      assert_eq!(interpret("Tanh[1.0]").unwrap(), "0.7615941559557649");
    }
  }

  mod sech_tests {
    use super::*;
    #[test]
    fn test_sech_zero() {
      assert_eq!(interpret("Sech[0]").unwrap(), "1");
    }
    #[test]
    fn test_sech_real() {
      // Sech[1.0] = 1/Cosh[1.0]
      assert_eq!(interpret("Sech[1.0]").unwrap(), "0.6480542736638855");
    }
  }

  mod arcsinh_tests {
    use super::*;
    #[test]
    fn test_arcsinh_zero() {
      assert_eq!(interpret("ArcSinh[0]").unwrap(), "0");
    }
    #[test]
    fn test_arcsinh_real() {
      assert_eq!(interpret("ArcSinh[1.0]").unwrap(), "0.881373587019543");
    }
  }

  mod arccosh_tests {
    use super::*;
    #[test]
    fn test_arccosh_one() {
      assert_eq!(interpret("ArcCosh[1]").unwrap(), "0");
    }
    #[test]
    fn test_arccosh_real() {
      assert_eq!(interpret("ArcCosh[2.0]").unwrap(), "1.3169578969248166");
    }
  }

  mod arctanh_tests {
    use super::*;
    #[test]
    fn test_arctanh_zero() {
      assert_eq!(interpret("ArcTanh[0]").unwrap(), "0");
    }
    #[test]
    fn test_arctanh_real() {
      assert_eq!(interpret("ArcTanh[0.5]").unwrap(), "0.5493061443340549");
    }
  }

  // ─── String Functions ──────────────────────────────────────────────
  mod capitalize_tests {
    use super::*;
    #[test]
    fn test_capitalize() {
      assert_eq!(
        interpret(r#"Capitalize["hello world"]"#).unwrap(),
        "Hello world"
      );
    }
    #[test]
    fn test_capitalize_empty() {
      assert_eq!(interpret(r#"Capitalize[""]"#).unwrap(), "");
    }
    #[test]
    fn test_capitalize_already() {
      assert_eq!(interpret(r#"Capitalize["Hello"]"#).unwrap(), "Hello");
    }
  }

  mod decapitalize_tests {
    use super::*;
    #[test]
    fn test_decapitalize() {
      assert_eq!(
        interpret(r#"Decapitalize["Hello World"]"#).unwrap(),
        "hello World"
      );
    }
    #[test]
    fn test_decapitalize_empty() {
      assert_eq!(interpret(r#"Decapitalize[""]"#).unwrap(), "");
    }
  }

  mod string_insert_tests {
    use super::*;
    #[test]
    fn test_string_insert() {
      assert_eq!(
        interpret(r#"StringInsert["abcdef", "X", 3]"#).unwrap(),
        "abXcdef"
      );
    }
    #[test]
    fn test_string_insert_start() {
      assert_eq!(interpret(r#"StringInsert["abc", "X", 1]"#).unwrap(), "Xabc");
    }
    #[test]
    fn test_string_insert_negative() {
      assert_eq!(
        interpret(r#"StringInsert["abc", "X", -1]"#).unwrap(),
        "abcX"
      );
    }
  }

  mod string_delete_tests {
    use super::*;
    #[test]
    fn test_string_delete() {
      assert_eq!(interpret(r#"StringDelete["abcabc", "b"]"#).unwrap(), "acac");
    }
    #[test]
    fn test_string_delete_none() {
      assert_eq!(interpret(r#"StringDelete["abc", "x"]"#).unwrap(), "abc");
    }
  }

  // ─── Catch/Throw ───────────────────────────────────────────────────
  mod catch_throw_tests {
    use super::*;
    #[test]
    fn test_catch_with_throw() {
      assert_eq!(interpret("Catch[1 + Throw[2]]").unwrap(), "2");
    }
    #[test]
    fn test_catch_no_throw() {
      assert_eq!(interpret("Catch[1 + 2]").unwrap(), "3");
    }
    #[test]
    fn test_catch_with_tag() {
      assert_eq!(
        interpret(r#"Catch[Throw["hello", "tag"], "tag"]"#).unwrap(),
        "hello"
      );
    }
    #[test]
    fn test_throw_value() {
      assert_eq!(interpret("Catch[Throw[42]]").unwrap(), "42");
    }
    #[test]
    fn test_nested_catch() {
      assert_eq!(interpret("Catch[Catch[Throw[1, a], b], a]").unwrap(), "1");
    }
  }
}
