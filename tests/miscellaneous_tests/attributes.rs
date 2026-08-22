#[cfg(test)]
mod tests {
  use woxi::evaluator::Attributes;

  fn test_attributes(test: fn(&str, Attributes) -> bool) {
    let names = woxi::evaluator::all_builtin_symbol_names();
    let mut failures: Vec<&str> = vec![];
    for name in names {
      let mask = woxi::evaluator::get_builtin_attributes_mask(name);
      if !test(name, mask) {
        failures.push(name);
      }
    }

    assert!(
      failures.is_empty(),
      "mask/is_builtin mismatch: {}",
      failures.join(", ")
    );
  }

  #[test]
  fn is_builtin_flat() {
    test_attributes(|name: &str, mask: Attributes| -> bool {
      mask.contains(Attributes::Flat) == woxi::evaluator::is_builtin_flat(name)
    });
  }

  #[test]
  fn is_builtin_orderless() {
    test_attributes(|name: &str, mask: Attributes| -> bool {
      mask.contains(Attributes::Orderless)
        == woxi::evaluator::is_builtin_orderless(name)
    });
  }

  #[test]
  fn is_builtin_listable() {
    test_attributes(|name: &str, mask: Attributes| -> bool {
      mask.contains(Attributes::Listable)
        == woxi::evaluator::is_builtin_listable(name)
    });
  }
}
