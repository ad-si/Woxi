use pest::iterators::Pair;

use crate::{InterpreterError, Rule};

/// Represents a symbolic expression for differentiation
#[derive(Debug, Clone)]
enum SymExpr {
  Num(i64),
  Var(String),
  Add(Box<SymExpr>, Box<SymExpr>),
  Sub(Box<SymExpr>, Box<SymExpr>),
  Mul(Box<SymExpr>, Box<SymExpr>),
  Div(Box<SymExpr>, Box<SymExpr>),
  Pow(Box<SymExpr>, Box<SymExpr>),
}

impl SymExpr {
  /// Check if this expression equals a constant value
  fn is_const(&self, val: i64) -> bool {
    matches!(self, SymExpr::Num(n) if *n == val)
  }

  /// Check if this expression is a constant (doesn't contain the variable)
  fn is_constant_wrt(&self, var: &str) -> bool {
    match self {
      SymExpr::Num(_) => true,
      SymExpr::Var(v) => v != var,
      SymExpr::Add(a, b)
      | SymExpr::Sub(a, b)
      | SymExpr::Mul(a, b)
      | SymExpr::Div(a, b)
      | SymExpr::Pow(a, b) => a.is_constant_wrt(var) && b.is_constant_wrt(var),
    }
  }
}

/// Parse a Pair into a symbolic expression
fn parse_to_sym(pair: Pair<Rule>) -> Result<SymExpr, InterpreterError> {
  match pair.as_rule() {
    Rule::Integer => {
      let n = pair.as_str().parse::<i64>().map_err(|_| {
        InterpreterError::EvaluationError("Invalid integer".into())
      })?;
      Ok(SymExpr::Num(n))
    }
    Rule::Real => {
      // For symbolic computation, we'll treat reals as symbols for now
      Ok(SymExpr::Var(pair.as_str().to_string()))
    }
    Rule::Identifier => Ok(SymExpr::Var(pair.as_str().to_string())),
    Rule::NumericValue => {
      let inner = pair.into_inner().next().unwrap();
      parse_to_sym(inner)
    }
    Rule::Term => {
      let inner = pair.into_inner().next().unwrap();
      parse_to_sym(inner)
    }
    Rule::Expression => {
      let items: Vec<_> = pair.into_inner().collect();

      if items.len() == 1 {
        return parse_to_sym(items.into_iter().next().unwrap());
      }

      // Parse expression with operators
      // Collect terms and operators
      let mut terms: Vec<SymExpr> = vec![];
      let mut ops: Vec<String> = vec![];

      let mut iter = items.into_iter();
      if let Some(first) = iter.next() {
        terms.push(parse_to_sym(first)?);
      }

      while let Some(op_pair) = iter.next() {
        if op_pair.as_rule() == Rule::Operator {
          ops.push(op_pair.as_str().to_string());
          if let Some(term) = iter.next() {
            terms.push(parse_to_sym(term)?);
          }
        }
      }

      // Build expression tree respecting operator precedence
      // First pass: handle ^ (highest precedence, right-associative)
      let mut i = ops.len();
      while i > 0 {
        i -= 1;
        if ops[i] == "^" {
          let right = terms.remove(i + 1);
          let left = terms.remove(i);
          terms.insert(i, SymExpr::Pow(Box::new(left), Box::new(right)));
          ops.remove(i);
        }
      }

      // Second pass: handle * and /
      let mut i = 0;
      while i < ops.len() {
        if ops[i] == "*" {
          let right = terms.remove(i + 1);
          let left = terms.remove(i);
          terms.insert(i, SymExpr::Mul(Box::new(left), Box::new(right)));
          ops.remove(i);
        } else if ops[i] == "/" {
          let right = terms.remove(i + 1);
          let left = terms.remove(i);
          terms.insert(i, SymExpr::Div(Box::new(left), Box::new(right)));
          ops.remove(i);
        } else {
          i += 1;
        }
      }

      // Third pass: handle + and -
      let mut result = terms.remove(0);
      for (op, term) in ops.into_iter().zip(terms.into_iter()) {
        result = match op.as_str() {
          "+" => SymExpr::Add(Box::new(result), Box::new(term)),
          "-" => SymExpr::Sub(Box::new(result), Box::new(term)),
          _ => {
            return Err(InterpreterError::EvaluationError(format!(
              "Unexpected operator: {}",
              op
            )))
          }
        };
      }

      Ok(result)
    }
    _ => Err(InterpreterError::EvaluationError(format!(
      "Cannot convert {:?} to symbolic expression",
      pair.as_rule()
    ))),
  }
}

/// Differentiate a symbolic expression with respect to a variable
fn differentiate(expr: &SymExpr, var: &str) -> SymExpr {
  match expr {
    SymExpr::Num(_) => SymExpr::Num(0),
    SymExpr::Var(v) => {
      if v == var {
        SymExpr::Num(1)
      } else {
        SymExpr::Num(0)
      }
    }
    SymExpr::Add(a, b) => {
      let da = differentiate(a, var);
      let db = differentiate(b, var);
      SymExpr::Add(Box::new(da), Box::new(db))
    }
    SymExpr::Sub(a, b) => {
      let da = differentiate(a, var);
      let db = differentiate(b, var);
      SymExpr::Sub(Box::new(da), Box::new(db))
    }
    SymExpr::Mul(a, b) => {
      // Product rule: (a*b)' = a'*b + a*b'
      let da = differentiate(a, var);
      let db = differentiate(b, var);
      SymExpr::Add(
        Box::new(SymExpr::Mul(Box::new(da), b.clone())),
        Box::new(SymExpr::Mul(a.clone(), Box::new(db))),
      )
    }
    SymExpr::Div(a, b) => {
      // Quotient rule: (a/b)' = (a'*b - a*b') / b^2
      let da = differentiate(a, var);
      let db = differentiate(b, var);
      SymExpr::Div(
        Box::new(SymExpr::Sub(
          Box::new(SymExpr::Mul(Box::new(da), b.clone())),
          Box::new(SymExpr::Mul(a.clone(), Box::new(db))),
        )),
        Box::new(SymExpr::Pow(b.clone(), Box::new(SymExpr::Num(2)))),
      )
    }
    SymExpr::Pow(base, exp) => {
      // Power rule for x^n where n is constant: n*x^(n-1)
      // General case with chain rule: x^n * (n' * ln(x) + n * x'/x)
      // For simplicity, we'll handle x^n where n is constant wrt var
      if exp.is_constant_wrt(var) {
        // d/dx[f(x)^n] = n * f(x)^(n-1) * f'(x)
        let df = differentiate(base, var);
        SymExpr::Mul(
          Box::new(SymExpr::Mul(
            exp.clone(),
            Box::new(SymExpr::Pow(
              base.clone(),
              Box::new(SymExpr::Sub(exp.clone(), Box::new(SymExpr::Num(1)))),
            )),
          )),
          Box::new(df),
        )
      } else if base.is_constant_wrt(var) {
        // d/dx[a^g(x)] = a^g(x) * ln(a) * g'(x)
        // For now, return unevaluated
        SymExpr::Mul(
          Box::new(expr.clone()),
          Box::new(SymExpr::Mul(
            Box::new(SymExpr::Var(format!("Log[{}]", format_sym(base)))),
            Box::new(differentiate(exp, var)),
          )),
        )
      } else {
        // General case: f(x)^g(x)
        // For now, handle the common case of var^var
        // d/dx[x^x] = x^x * (1 + ln(x))
        // This is complex, return unevaluated for now
        SymExpr::Var(format!("D[{}, {}]", format_sym(expr), var))
      }
    }
  }
}

/// Simplify a symbolic expression
fn simplify(expr: SymExpr) -> SymExpr {
  match expr {
    SymExpr::Num(_) | SymExpr::Var(_) => expr,
    SymExpr::Add(a, b) => {
      let a = simplify(*a);
      let b = simplify(*b);
      // 0 + x = x
      if a.is_const(0) {
        return b;
      }
      // x + 0 = x
      if b.is_const(0) {
        return a;
      }
      // n + m = n+m
      if let (SymExpr::Num(n), SymExpr::Num(m)) = (&a, &b) {
        return SymExpr::Num(n + m);
      }
      SymExpr::Add(Box::new(a), Box::new(b))
    }
    SymExpr::Sub(a, b) => {
      let a = simplify(*a);
      let b = simplify(*b);
      // x - 0 = x
      if b.is_const(0) {
        return a;
      }
      // 0 - x = -x (represented as -1 * x)
      if a.is_const(0) {
        return simplify(SymExpr::Mul(
          Box::new(SymExpr::Num(-1)),
          Box::new(b),
        ));
      }
      // n - m = n-m
      if let (SymExpr::Num(n), SymExpr::Num(m)) = (&a, &b) {
        return SymExpr::Num(n - m);
      }
      SymExpr::Sub(Box::new(a), Box::new(b))
    }
    SymExpr::Mul(a, b) => {
      let a = simplify(*a);
      let b = simplify(*b);
      // 0 * x = 0
      if a.is_const(0) || b.is_const(0) {
        return SymExpr::Num(0);
      }
      // 1 * x = x
      if a.is_const(1) {
        return b;
      }
      // x * 1 = x
      if b.is_const(1) {
        return a;
      }
      // n * m = n*m
      if let (SymExpr::Num(n), SymExpr::Num(m)) = (&a, &b) {
        return SymExpr::Num(n * m);
      }
      SymExpr::Mul(Box::new(a), Box::new(b))
    }
    SymExpr::Div(a, b) => {
      let a = simplify(*a);
      let b = simplify(*b);
      // 0 / x = 0
      if a.is_const(0) {
        return SymExpr::Num(0);
      }
      // x / 1 = x
      if b.is_const(1) {
        return a;
      }
      SymExpr::Div(Box::new(a), Box::new(b))
    }
    SymExpr::Pow(base, exp) => {
      let base = simplify(*base);
      let exp = simplify(*exp);
      // x^0 = 1
      if exp.is_const(0) {
        return SymExpr::Num(1);
      }
      // x^1 = x
      if exp.is_const(1) {
        return base;
      }
      // 0^n = 0 (for n > 0)
      if base.is_const(0) {
        return SymExpr::Num(0);
      }
      // 1^n = 1
      if base.is_const(1) {
        return SymExpr::Num(1);
      }
      SymExpr::Pow(Box::new(base), Box::new(exp))
    }
  }
}

/// Collect terms from an Add/Sub expression, with their signs
fn collect_additive_terms(expr: &SymExpr) -> Vec<(i8, SymExpr)> {
  match expr {
    SymExpr::Add(a, b) => {
      let mut terms = collect_additive_terms(a);
      terms.extend(collect_additive_terms(b));
      terms
    }
    SymExpr::Sub(a, b) => {
      let mut terms = collect_additive_terms(a);
      for (sign, term) in collect_additive_terms(b) {
        terms.push((-sign, term));
      }
      terms
    }
    _ => vec![(1, expr.clone())],
  }
}

/// Format an additive expression (sum) with constants first (Wolfram style)
fn format_additive(expr: &SymExpr) -> String {
  let terms = collect_additive_terms(expr);

  // Separate constant and non-constant terms
  let mut const_terms: Vec<(i8, i64)> = vec![];
  let mut var_terms: Vec<(i8, SymExpr)> = vec![];

  for (sign, term) in terms {
    if let SymExpr::Num(n) = term {
      const_terms.push((sign, n));
    } else {
      var_terms.push((sign, term));
    }
  }

  // Sum up constants
  let const_sum: i64 = const_terms.iter().map(|(s, n)| (*s as i64) * n).sum();

  // Build output: constants first, then variables
  let mut parts: Vec<String> = vec![];

  if const_sum != 0 || var_terms.is_empty() {
    parts.push(const_sum.to_string());
  }

  for (sign, term) in var_terms {
    let ts = format_sym(&term);
    if parts.is_empty() {
      if sign == -1 {
        parts.push(format!("-{}", ts));
      } else {
        parts.push(ts);
      }
    } else if sign == -1 {
      parts.push(format!("- {}", ts));
    } else {
      parts.push(format!("+ {}", ts));
    }
  }

  parts.join(" ")
}

/// Format a symbolic expression as a string
fn format_sym(expr: &SymExpr) -> String {
  match expr {
    SymExpr::Num(n) => n.to_string(),
    SymExpr::Var(v) => v.clone(),
    SymExpr::Add(_, _) | SymExpr::Sub(_, _) => format_additive(expr),
    SymExpr::Mul(a, b) => {
      let as_ = format_sym_factor(a);
      let bs = format_sym_factor(b);
      format!("{}*{}", as_, bs)
    }
    SymExpr::Div(a, b) => {
      format!("{}/{}", format_sym_factor(a), format_sym_factor(b))
    }
    SymExpr::Pow(base, exp) => {
      let base_s = format_sym_factor(base);
      let exp_s = format_sym(exp);
      // Wrap complex exponents in parentheses
      if needs_parens_exp(exp) {
        format!("{}^({})", base_s, exp_s)
      } else {
        format!("{}^{}", base_s, exp_s)
      }
    }
  }
}

/// Format a factor (adds parentheses for complex expressions)
fn format_sym_factor(expr: &SymExpr) -> String {
  match expr {
    SymExpr::Num(_) | SymExpr::Var(_) | SymExpr::Pow(_, _) => format_sym(expr),
    _ => format!("({})", format_sym(expr)),
  }
}

/// Check if expression needs parentheses in exponent position
fn needs_parens_exp(expr: &SymExpr) -> bool {
  matches!(
    expr,
    SymExpr::Add(_, _) | SymExpr::Sub(_, _) | SymExpr::Mul(_, _) | SymExpr::Div(_, _)
  )
}

/// Handle D[expr, var] - symbolic differentiation
pub fn derivative(args_pairs: &[Pair<Rule>]) -> Result<String, InterpreterError> {
  if args_pairs.len() != 2 {
    return Err(InterpreterError::EvaluationError(
      "D expects exactly 2 arguments".into(),
    ));
  }

  // Get the variable name
  let var_pair = &args_pairs[1];
  let var_name = match var_pair.as_rule() {
    Rule::Identifier => var_pair.as_str().to_string(),
    Rule::Expression => {
      let mut inner = var_pair.clone().into_inner();
      if let Some(first) = inner.next() {
        if first.as_rule() == Rule::Identifier && inner.next().is_none() {
          first.as_str().to_string()
        } else {
          return Err(InterpreterError::EvaluationError(
            "Second argument of D must be a symbol".into(),
          ));
        }
      } else {
        return Err(InterpreterError::EvaluationError(
          "Second argument of D must be a symbol".into(),
        ));
      }
    }
    _ => {
      return Err(InterpreterError::EvaluationError(
        "Second argument of D must be a symbol".into(),
      ))
    }
  };

  // Parse the expression to differentiate
  let expr_pair = args_pairs[0].clone();
  let sym_expr = parse_to_sym(expr_pair)?;

  // Differentiate
  let derivative = differentiate(&sym_expr, &var_name);

  // Simplify
  let simplified = simplify(derivative);

  // Format
  Ok(format_sym(&simplified))
}
