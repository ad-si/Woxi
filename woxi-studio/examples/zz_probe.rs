fn main() {
  let code = std::fs::read_to_string(std::env::args().nth(1).unwrap()).unwrap();
  eprintln!("@@ with_stdout");
  let r = woxi::interpret_with_stdout(&code);
  eprintln!("@@ warnings: {:?}", r.as_ref().map(|r| r.warnings.len()));
  eprintln!("@@ to_expr");
  let _ = woxi::interpret_to_expr(&code);
  eprintln!("@@ done");
}
