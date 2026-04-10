#[allow(unused_imports)]
use super::utilities::*;
#[allow(unused_imports)]
use super::*;

/// Wolfram canonical ordering for expressions.
/// For strings: case-insensitive first, then lowercase before uppercase for ties.
/// For numbers: numeric comparison.
/// Mixed: numbers before strings.
/// Extract (real, imaginary) parts from a numeric expression for sorting.
/// Returns None for non-numeric expressions.
fn expr_to_complex_parts(e: &Expr) -> Option<(f64, f64)> {
  use crate::functions::math_ast::try_eval_to_f64;
  use crate::functions::math_ast::try_eval_to_f64_with_infinity;
  // Pure real number (including Infinity/-Infinity)
  if let Some(v) = try_eval_to_f64_with_infinity(e) {
    return Some((v, 0.0));
  }
  // Check if expression contains I (complex unit)
  let s = crate::syntax::expr_to_string(e);
  if !s.contains('I') {
    return None;
  }
  match e {
    // Pure imaginary: n*I (BinaryOp form)
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Times,
      left,
      right,
    } => {
      if matches!(right.as_ref(), Expr::Identifier(name) if name == "I")
        && let Some(im) = try_eval_to_f64(left)
      {
        return Some((0.0, im));
      }
      if matches!(left.as_ref(), Expr::Identifier(name) if name == "I")
        && let Some(im) = try_eval_to_f64(right)
      {
        return Some((0.0, im));
      }
      None
    }
    // a + b*I (BinaryOp form)
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Plus,
      left,
      right,
    } => {
      if let Some(re) = try_eval_to_f64(left)
        && let Some((_, im)) = expr_to_complex_parts(right)
      {
        return Some((re, im));
      }
      None
    }
    // a - b*I (BinaryOp form)
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Minus,
      left,
      right,
    } => {
      if let Some(re) = try_eval_to_f64(left)
        && let Some((_, im)) = expr_to_complex_parts(right)
      {
        return Some((re, -im));
      }
      None
    }
    // FunctionCall Plus[re, Times[im, I]]
    Expr::FunctionCall { name, args } if name == "Plus" && args.len() == 2 => {
      if let Some(re) = try_eval_to_f64(&args[0])
        && let Some((_, im)) = expr_to_complex_parts(&args[1])
      {
        return Some((re, im));
      }
      if let Some(re) = try_eval_to_f64(&args[1])
        && let Some((_, im)) = expr_to_complex_parts(&args[0])
      {
        return Some((0.0 + im, re)); // im is imaginary coefficient
      }
      None
    }
    // FunctionCall Times[n, I]
    Expr::FunctionCall { name, args } if name == "Times" && args.len() == 2 => {
      if matches!(&args[1], Expr::Identifier(n) if n == "I")
        && let Some(im) = try_eval_to_f64(&args[0])
      {
        return Some((0.0, im));
      }
      if matches!(&args[0], Expr::Identifier(n) if n == "I")
        && let Some(im) = try_eval_to_f64(&args[1])
      {
        return Some((0.0, im));
      }
      None
    }
    Expr::FunctionCall { name, args }
      if name == "Complex" && args.len() == 2 =>
    {
      if let (Some(re), Some(im)) =
        (try_eval_to_f64(&args[0]), try_eval_to_f64(&args[1]))
      {
        return Some((re, im));
      }
      None
    }
    // Just I
    Expr::Identifier(name) if name == "I" => Some((0.0, 1.0)),
    // Negated: -I, -(a+bI)
    Expr::UnaryOp {
      op: crate::syntax::UnaryOperator::Minus,
      operand,
    } => {
      if let Some((re, im)) = expr_to_complex_parts(operand) {
        return Some((-re, -im));
      }
      None
    }
    _ => None,
  }
}

/// Public wrapper for expr_to_complex_parts.
pub fn expr_to_complex_parts_pub(e: &Expr) -> Option<(f64, f64)> {
  expr_to_complex_parts(e)
}

/// Check if an expression is Infinity or -Infinity (DirectedInfinity).
/// In Wolfram's canonical ordering, these sort after all finite numbers.
fn is_infinity_expr(e: &Expr) -> Option<i8> {
  let s = crate::syntax::expr_to_string(e);
  if s == "Infinity" {
    Some(1)
  } else if s == "-Infinity" {
    Some(-1)
  } else if s == "ComplexInfinity" {
    Some(0)
  } else {
    None
  }
}

pub fn canonical_cmp(a: &Expr, b: &Expr) -> std::cmp::Ordering {
  // Handle Infinity/-Infinity separately: they sort after all finite numbers
  let a_inf = is_infinity_expr(a);
  let b_inf = is_infinity_expr(b);
  match (a_inf, b_inf) {
    (Some(ai), Some(bi)) => {
      // Both infinity: -Infinity < ComplexInfinity < Infinity
      return ai.cmp(&bi);
    }
    (Some(_), None) => return std::cmp::Ordering::Greater, // Infinity after everything
    (None, Some(_)) => return std::cmp::Ordering::Less,
    (None, None) => {}
  }

  // Try numeric comparison (including complex numbers)
  let a_num = expr_to_complex_parts(a);
  let b_num = expr_to_complex_parts(b);

  match (a_num, b_num) {
    (Some((a_re, a_im)), Some((b_re, b_im))) => {
      // Both numeric: compare by real part first, then imaginary part
      match a_re.partial_cmp(&b_re).unwrap_or(std::cmp::Ordering::Equal) {
        std::cmp::Ordering::Equal => {
          // Same real part: pure reals (im=0) come first
          if a_im == 0.0 && b_im != 0.0 {
            return std::cmp::Ordering::Less;
          }
          if a_im != 0.0 && b_im == 0.0 {
            return std::cmp::Ordering::Greater;
          }
          a_im.partial_cmp(&b_im).unwrap_or(std::cmp::Ordering::Equal)
        }
        other => other,
      }
    }
    (Some(_), None) => std::cmp::Ordering::Less, // numbers before non-numbers
    (None, Some(_)) => std::cmp::Ordering::Greater,
    (None, None) => {
      // Non-numeric: string comparison
      let sa = crate::syntax::expr_to_string(a);
      let sb = crate::syntax::expr_to_string(b);
      let la = sa.to_lowercase();
      let lb = sb.to_lowercase();
      match la.cmp(&lb) {
        std::cmp::Ordering::Equal => {
          for (ca, cb) in sa.chars().zip(sb.chars()) {
            if ca != cb {
              if ca.to_lowercase().eq(cb.to_lowercase()) {
                if ca.is_lowercase() {
                  return std::cmp::Ordering::Less;
                } else {
                  return std::cmp::Ordering::Greater;
                }
              }
              return ca.cmp(&cb);
            }
          }
          sa.len().cmp(&sb.len())
        }
        other => other,
      }
    }
  }
}

/// AST-based SortBy: sort elements by the value of a function.
/// SortBy[{a, b, c}, f] -> elements sorted by f[x]
pub fn sort_by_ast(list: &Expr, func: &Expr) -> Result<Expr, InterpreterError> {
  match list {
    Expr::List(items) => {
      let mut keyed: Vec<(Expr, Expr)> = items
        .iter()
        .map(|item| {
          let key = apply_func_ast(func, item)?;
          Ok((item.clone(), key))
        })
        .collect::<Result<_, InterpreterError>>()?;

      keyed.sort_by(|a, b| {
        let key_ord = canonical_cmp(&a.1, &b.1);
        if key_ord == std::cmp::Ordering::Equal {
          canonical_cmp(&a.0, &b.0)
        } else {
          key_ord
        }
      });

      Ok(Expr::List(
        keyed.into_iter().map(|(item, _)| item).collect(),
      ))
    }
    Expr::Association(pairs) => {
      let mut keyed: Vec<((Expr, Expr), Expr)> = pairs
        .iter()
        .map(|(k, v)| {
          let key = apply_func_ast(func, v)?;
          Ok(((k.clone(), v.clone()), key))
        })
        .collect::<Result<_, InterpreterError>>()?;

      keyed.sort_by(|a, b| {
        let key_ord = canonical_cmp(&a.1, &b.1);
        if key_ord == std::cmp::Ordering::Equal {
          canonical_cmp(&(a.0).1, &(b.0).1)
        } else {
          key_ord
        }
      });

      Ok(Expr::Association(
        keyed.into_iter().map(|(pair, _)| pair).collect(),
      ))
    }
    Expr::FunctionCall { name, args } => {
      let mut keyed: Vec<(Expr, Expr)> = args
        .iter()
        .map(|item| {
          let key = apply_func_ast(func, item)?;
          Ok((item.clone(), key))
        })
        .collect::<Result<_, InterpreterError>>()?;

      keyed.sort_by(|a, b| {
        let key_ord = canonical_cmp(&a.1, &b.1);
        if key_ord == std::cmp::Ordering::Equal {
          canonical_cmp(&a.0, &b.0)
        } else {
          key_ord
        }
      });

      Ok(Expr::FunctionCall {
        name: name.clone(),
        args: keyed.into_iter().map(|(item, _)| item).collect(),
      })
    }
    _ => Ok(Expr::FunctionCall {
      name: "SortBy".to_string(),
      args: vec![list.clone(), func.clone()],
    }),
  }
}

///// Ordering[list] / Ordering[list, n] / Ordering[list, n, p]
pub fn ordering_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 3 {
    return Err(InterpreterError::EvaluationError(
      "Ordering expects 1, 2, or 3 arguments".into(),
    ));
  }

  let items = match &args[0] {
    Expr::List(items) => items,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "Ordering".to_string(),
        args: args.to_vec(),
      });
    }
  };

  let mut indexed: Vec<(usize, &Expr)> = items.iter().enumerate().collect();

  // Decide which comparator to use. With 3 args, the third is either a
  // bare symbol (Less/Greater) or a predicate p such that p[a, b] is True
  // when a should come before b.
  let comparator: Option<&Expr> = args.get(2);
  let use_greater = matches!(
    comparator,
    Some(Expr::Identifier(n)) if n == "Greater"
  );
  let use_less = matches!(
    comparator,
    Some(Expr::Identifier(n)) if n == "Less"
  );
  let use_predicate = comparator.is_some() && !use_greater && !use_less;

  if use_predicate {
    let p = comparator.unwrap().clone();
    let mut err: Option<InterpreterError> = None;
    indexed.sort_by(|a, b| {
      if err.is_some() {
        return std::cmp::Ordering::Equal;
      }
      match crate::functions::list_helpers_ast::apply_func_to_two_args(
        &p, a.1, b.1,
      ) {
        Ok(Expr::Identifier(ref s)) if s == "True" => std::cmp::Ordering::Less,
        Ok(_) => std::cmp::Ordering::Greater,
        Err(e) => {
          err = Some(e);
          std::cmp::Ordering::Equal
        }
      }
    });
    if let Some(e) = err {
      return Err(e);
    }
  } else {
    indexed.sort_by(|a, b| {
      let ord = crate::functions::list_helpers_ast::canonical_cmp(a.1, b.1);
      if use_greater { ord.reverse() } else { ord }
    });
  }

  let mut result: Vec<Expr> = indexed
    .iter()
    .map(|(idx, _)| Expr::Integer((*idx + 1) as i128))
    .collect();

  // The second argument, if present and not the symbol `All`, limits the
  // number of positions returned.
  if args.len() >= 2 {
    let is_all = matches!(&args[1], Expr::Identifier(n) if n == "All");
    if !is_all && let Some(n) = expr_to_i128(&args[1]) {
      if n >= 0 {
        result.truncate(n as usize);
      } else {
        // Negative n: take last |n| elements (largest positions)
        let abs_n = n.unsigned_abs() as usize;
        if abs_n <= result.len() {
          result = result.split_off(result.len() - abs_n);
        }
      }
    }
  }

  Ok(Expr::List(result))
}

/// MinimalBy[list, f] - Returns all elements that minimize f
pub fn minimal_by_ast(
  list: &Expr,
  func: &Expr,
) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) if !items.is_empty() => items,
    Expr::List(_) => return Ok(Expr::List(vec![])),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "MinimalBy".to_string(),
        args: vec![list.clone(), func.clone()],
      });
    }
  };

  let keyed: Vec<(Expr, Expr)> = items
    .iter()
    .map(|item| {
      let key = apply_func_ast(func, item)?;
      Ok((item.clone(), key))
    })
    .collect::<Result<_, InterpreterError>>()?;

  let min_key = keyed
    .iter()
    .map(|(_, k)| k)
    .min_by(|a, b| {
      let ka = crate::syntax::expr_to_string(a);
      let kb = crate::syntax::expr_to_string(b);
      if let (Ok(na), Ok(nb)) = (ka.parse::<f64>(), kb.parse::<f64>()) {
        na.partial_cmp(&nb).unwrap_or(std::cmp::Ordering::Equal)
      } else {
        ka.cmp(&kb)
      }
    })
    .cloned();

  if let Some(min_k) = min_key {
    let min_str = crate::syntax::expr_to_string(&min_k);
    let result: Vec<Expr> = keyed
      .into_iter()
      .filter(|(_, k)| crate::syntax::expr_to_string(k) == min_str)
      .map(|(item, _)| item)
      .collect();
    Ok(Expr::List(result))
  } else {
    Ok(Expr::List(vec![]))
  }
}

/// MaximalBy[list, f] - Returns all elements that maximize f
pub fn maximal_by_ast(
  list: &Expr,
  func: &Expr,
) -> Result<Expr, InterpreterError> {
  let items = match list {
    Expr::List(items) if !items.is_empty() => items,
    Expr::List(_) => return Ok(Expr::List(vec![])),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "MaximalBy".to_string(),
        args: vec![list.clone(), func.clone()],
      });
    }
  };

  let keyed: Vec<(Expr, Expr)> = items
    .iter()
    .map(|item| {
      let key = apply_func_ast(func, item)?;
      Ok((item.clone(), key))
    })
    .collect::<Result<_, InterpreterError>>()?;

  let max_key = keyed
    .iter()
    .map(|(_, k)| k)
    .max_by(|a, b| {
      let ka = crate::syntax::expr_to_string(a);
      let kb = crate::syntax::expr_to_string(b);
      if let (Ok(na), Ok(nb)) = (ka.parse::<f64>(), kb.parse::<f64>()) {
        na.partial_cmp(&nb).unwrap_or(std::cmp::Ordering::Equal)
      } else {
        ka.cmp(&kb)
      }
    })
    .cloned();

  if let Some(max_k) = max_key {
    let max_str = crate::syntax::expr_to_string(&max_k);
    let result: Vec<Expr> = keyed
      .into_iter()
      .filter(|(_, k)| crate::syntax::expr_to_string(k) == max_str)
      .map(|(item, _)| item)
      .collect();
    Ok(Expr::List(result))
  } else {
    Ok(Expr::List(vec![]))
  }
}

/// AST-based Sort: sort a list.
pub fn sort_ast(list: &Expr) -> Result<Expr, InterpreterError> {
  match list {
    Expr::List(items) => {
      let mut sorted = items.clone();
      sorted.sort_by(canonical_cmp);
      Ok(Expr::List(sorted))
    }
    Expr::Association(pairs) => {
      let mut sorted = pairs.clone();
      sorted.sort_by(|a, b| canonical_cmp(&a.1, &b.1));
      Ok(Expr::Association(sorted))
    }
    Expr::FunctionCall { name, args } => {
      let mut sorted = args.clone();
      sorted.sort_by(canonical_cmp);
      Ok(Expr::FunctionCall {
        name: name.clone(),
        args: sorted,
      })
    }
    _ => Ok(Expr::FunctionCall {
      name: "Sort".to_string(),
      args: vec![list.clone()],
    }),
  }
}

/// OrderedQ[list] - Tests if a list is in sorted (non-decreasing) order
pub fn ordered_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 1 {
    return Err(InterpreterError::EvaluationError(
      "OrderedQ expects exactly 1 argument".into(),
    ));
  }

  if let Expr::List(items) = &args[0] {
    if items.len() <= 1 {
      return Ok(Expr::Identifier("True".to_string()));
    }
    for i in 0..items.len() - 1 {
      if !expr_le(&items[i], &items[i + 1]) {
        return Ok(Expr::Identifier("False".to_string()));
      }
    }
    Ok(Expr::Identifier("True".to_string()))
  } else {
    Ok(Expr::FunctionCall {
      name: "OrderedQ".to_string(),
      args: args.to_vec(),
    })
  }
}

/// Compare two Expr values for canonical ordering.
/// Returns 1 if a < b, -1 if a > b, 0 if equal (Wolfram Order convention).
pub fn compare_exprs(a: &Expr, b: &Expr) -> i64 {
  use crate::functions::math_ast::try_eval_to_f64_with_infinity;
  // Try numeric comparison first (including Infinity/-Infinity)
  let a_num = try_eval_to_f64_with_infinity(a);
  let b_num = try_eval_to_f64_with_infinity(b);
  if let (Some(an), Some(bn)) = (a_num, b_num) {
    return if an < bn {
      1
    } else if an > bn {
      -1
    } else {
      0
    };
  }
  // Numbers come before non-numbers
  if a_num.is_some() {
    return 1;
  }
  if b_num.is_some() {
    return -1;
  }

  // Wolfram canonical ordering: symbols and compounds are compared structurally
  // Classification: atom-like (atoms, constants, powers) sort before function calls
  let a_is_atom = is_atom_expr(a);
  let b_is_atom = is_atom_expr(b);
  let a_is_power = is_power_expr(a);
  let b_is_power = is_power_expr(b);
  let a_is_func_call = !a_is_atom && !a_is_power && is_plain_func_call(a);
  let b_is_func_call = !b_is_atom && !b_is_power && is_plain_func_call(b);

  // Atoms and powers always sort before plain function calls
  let a_is_atom_like = a_is_atom || a_is_power;
  let b_is_atom_like = b_is_atom || b_is_power;

  if a_is_atom_like && b_is_func_call {
    1 // atom/power always before function call
  } else if a_is_func_call && b_is_atom_like {
    -1 // function call always after atom/power
  } else {
    // Same category: use standard ordering
    match (a_is_atom, b_is_atom) {
      (true, true) => {
        // Both atoms: alphabetical comparison
        let a_str = crate::syntax::expr_to_string(a);
        let b_str = crate::syntax::expr_to_string(b);
        wolfram_string_order(&a_str, &b_str)
      }
      (true, false) => {
        // Atom vs compound: compare atom with compound's sort key
        // Special case: Plus[neg, atom] with same atom sorts before the atom
        // (Wolfram: (-3+x)*x not x*(-3+x))
        if crate::functions::additive_is_neg_const_plus_ident(b, a) {
          return -1; // compound (Plus) comes before atom
        }
        let b_key = expr_sort_key(b);
        let a_str = crate::syntax::expr_to_string(a);
        let cmp = wolfram_string_order(&a_str, &b_key);
        if cmp == 0 {
          1 // atom comes before compound with same key
        } else {
          cmp
        }
      }
      (false, true) => {
        // Compound vs atom: reverse of above
        if crate::functions::additive_is_neg_const_plus_ident(a, b) {
          return 1; // compound (Plus) comes before atom
        }
        let a_key = expr_sort_key(a);
        let b_str = crate::syntax::expr_to_string(b);
        let cmp = wolfram_string_order(&a_key, &b_str);
        if cmp == 0 {
          -1 // compound comes after atom with same key
        } else {
          cmp
        }
      }
      (false, false) => {
        // Both compounds: compare sort keys, then by full string
        let a_key = expr_sort_key(a);
        let b_key = expr_sort_key(b);
        let cmp = wolfram_string_order(&a_key, &b_key);
        if cmp != 0 {
          return cmp;
        }
        let a_str = crate::syntax::expr_to_string(a);
        let b_str = crate::syntax::expr_to_string(b);
        wolfram_string_order(&a_str, &b_str)
      }
    }
  }
}

/// Check if an expression is a Power (BinaryOp or FunctionCall)
fn is_power_expr(e: &Expr) -> bool {
  matches!(
    e,
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Power,
      ..
    }
  ) || matches!(e, Expr::FunctionCall { name, args } if name == "Power" && args.len() == 2)
}

/// Check if an expression is a plain function call (not Plus/Times/Power/Rational)
fn is_plain_func_call(e: &Expr) -> bool {
  matches!(e, Expr::FunctionCall { name, .. }
    if name != "Plus" && name != "Times" && name != "Power" && name != "Rational")
}

/// Extract the sort key for a compound expression.
/// For Plus/Times: the last (largest) symbolic argument
/// For Power: the base
/// For other functions: the last argument, or the function name
pub fn expr_sort_key(e: &Expr) -> String {
  match e {
    Expr::FunctionCall { name, args } if !args.is_empty() => {
      // For Plus/Times (Orderless), use the last symbolic argument
      if (name == "Plus" || name == "Times")
        && let Some(last) = args.last()
      {
        if is_atom_expr(last) {
          return crate::syntax::expr_to_string(last);
        }
        return expr_sort_key(last);
      }
      // For Power/Sqrt: sort key is the base (same as BinaryOp::Power)
      if name == "Power" && args.len() == 2 {
        if is_atom_expr(&args[0]) {
          return crate::syntax::expr_to_string(&args[0]);
        }
        return expr_sort_key(&args[0]);
      }
      if let Some(sqrt_arg) = crate::functions::math_ast::is_sqrt(e) {
        if is_atom_expr(sqrt_arg) {
          return crate::syntax::expr_to_string(sqrt_arg);
        }
        return expr_sort_key(sqrt_arg);
      }
      // For other function calls (like C[1], Sin[x]), use the function name
      name.clone()
    }
    Expr::BinaryOp { op, left, right } => {
      use crate::syntax::BinaryOperator;
      match op {
        BinaryOperator::Power => {
          // Power: sort key is the base (recurse for compound bases)
          if is_atom_expr(left) {
            crate::syntax::expr_to_string(left)
          } else {
            expr_sort_key(left)
          }
        }
        BinaryOperator::Plus | BinaryOperator::Times => {
          // For binary plus/times: use the "larger" operand
          let l = crate::syntax::expr_to_string(left);
          let r = crate::syntax::expr_to_string(right);
          if wolfram_string_order(&l, &r) >= 0 {
            r
          } else {
            l
          }
        }
        _ => crate::syntax::expr_to_string(e),
      }
    }
    _ => crate::syntax::expr_to_string(e),
  }
}

/// Wolfram canonical string ordering: case-insensitive alphabetical, then lowercase < uppercase
pub fn wolfram_string_order(a: &str, b: &str) -> i64 {
  let a_chars: Vec<char> = a.chars().collect();
  let b_chars: Vec<char> = b.chars().collect();

  for (ac, bc) in a_chars.iter().zip(b_chars.iter()) {
    let al = ac.to_lowercase().next().unwrap_or(*ac);
    let bl = bc.to_lowercase().next().unwrap_or(*bc);
    if al != bl {
      // Case-insensitive comparison first
      return if al < bl { 1 } else { -1 };
    }
    // Same letter, different case: lowercase comes first
    if ac != bc {
      // lowercase < uppercase in Wolfram ordering
      if ac.is_lowercase() && bc.is_uppercase() {
        return 1;
      } else if ac.is_uppercase() && bc.is_lowercase() {
        return -1;
      }
    }
  }
  // If all compared chars are equal, shorter string comes first
  match a_chars.len().cmp(&b_chars.len()) {
    std::cmp::Ordering::Less => 1,
    std::cmp::Ordering::Greater => -1,
    std::cmp::Ordering::Equal => 0,
  }
}

/// Helper: compare two Expr values for ordering (less-or-equal)
fn expr_le(a: &Expr, b: &Expr) -> bool {
  // Use canonical_cmp for consistency with Sort
  !matches!(canonical_cmp(a, b), std::cmp::Ordering::Greater)
}
