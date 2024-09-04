use wolfram_parser::interpret;

#[test]
fn test_addition() {
  assert_eq!(interpret("1 + 2").unwrap(), 3.0);
}

#[test]
fn test_subtraction() {
  assert_eq!(interpret("3 - 1").unwrap(), 2.0);
}

#[test]
fn test_multiple_operations() {
  assert_eq!(interpret("1 + 2 - 3 + 4").unwrap(), 4.0);
}

#[test]
fn test_decimal_numbers() {
  assert_eq!(interpret("1.5 + 2.7").unwrap(), 4.2);
}

#[test]
fn test_negative_numbers() {
  assert_eq!(interpret("-1 + 3").unwrap(), 2.0);
}

#[test]
fn test_error_invalid_input() {
  assert!(interpret("1 + ").is_err());
}

#[test]
fn test_error_unknown_operator() {
  assert!(interpret("1 * 2").is_err());
}

#[test]
fn test_calculation() {
  assert_eq!(interpret("1 + 2").unwrap(), 3.0);
}

#[test]
fn test_complex_calculation() {
  assert_eq!(interpret("1 + 2 - 3 + 4").unwrap(), 4.0);
}
