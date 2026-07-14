#[allow(unused_imports)]
use super::utilities::*;
#[allow(unused_imports)]
use super::*;
use crate::syntax::{BinaryOperator, UnaryOperator};

/// Wolfram canonical ordering for expressions.
/// For strings: case-insensitive first, then lowercase before uppercase for ties.
/// For numbers: numeric comparison.
/// Mixed: numbers before strings.
/// Extract (real, imaginary) parts from a numeric expression for sorting.
/// Returns None for non-numeric expressions.
pub fn expr_to_complex_parts(e: &Expr) -> Option<(f64, f64)> {
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
      op: BinaryOperator::Times,
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
      op: BinaryOperator::Plus,
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
      op: BinaryOperator::Minus,
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
      op: UnaryOperator::Minus,
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

/// Exact ordering for real exact numbers (`Integer`, `BigInteger`, and
/// `Rational[…]` with integer parts), via BigInt cross-multiplication. Used
/// ahead of the f64 comparison in `canonical_cmp`, which collapses magnitudes
/// beyond ~1.8e308 to ±inf and reports distinct values as equal — breaking
/// Sort/MaximalBy on very large BigIntegers (e.g. Egyptian-fraction
/// denominators). Returns None unless both arguments are exact real numbers,
/// so Reals, complex values, and symbolic terms fall through unchanged.
fn exact_real_cmp(a: &Expr, b: &Expr) -> Option<std::cmp::Ordering> {
  use num_bigint::{BigInt, Sign};
  fn as_ratio(e: &Expr) -> Option<(BigInt, BigInt)> {
    match e {
      Expr::Integer(n) => Some((BigInt::from(*n), BigInt::from(1))),
      Expr::BigInteger(n) => Some((n.clone(), BigInt::from(1))),
      Expr::FunctionCall { name, args }
        if name == "Rational" && args.len() == 2 =>
      {
        let n = crate::functions::math_ast::expr_to_bigint(&args[0])?;
        let d = crate::functions::math_ast::expr_to_bigint(&args[1])?;
        match d.sign() {
          Sign::NoSign => None,
          Sign::Minus => Some((-n, -d)),
          Sign::Plus => Some((n, d)),
        }
      }
      _ => None,
    }
  }
  let (an, ad) = as_ratio(a)?;
  let (bn, bd) = as_ratio(b)?;
  // a/ad vs b/bd  ⇔  an*bd vs bn*ad  (both denominators positive).
  Some((an * &bd).cmp(&(bn * &ad)))
}

pub fn canonical_cmp(a: &Expr, b: &Expr) -> std::cmp::Ordering {
  // Two compatible Quantities sort by their physical value (converted to a
  // common unit): Sort[{3 m, 100 cm, 2 m}] -> {100 cm, 2 m, 3 m}. A tie in
  // value falls through to the structural comparison below.
  if let Some(ord) = crate::functions::quantity_ast::try_quantity_compare(a, b)
    && ord != std::cmp::Ordering::Equal
  {
    return ord;
  }

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

  // Exact comparison for large integers/rationals before falling back to the
  // f64 path (which loses precision and collapses values beyond f64 range).
  if let Some(ord) = exact_real_cmp(a, b) {
    return ord;
  }

  // Only NUMBER LITERALS compare numerically; symbolic constants (Pi, E,
  // Degree, GoldenRatio) and numeric composites (2*Pi, Sqrt[2]) sort
  // structurally after all numbers, per Wolfram canonical order
  // (Sort[{E, 3}] = {3, E}, Sort[{Sqrt[2], 7}] = {7, Sqrt[2]}).
  let is_number_literal = |e: &Expr| -> bool {
    matches!(
      e,
      Expr::Integer(_)
        | Expr::BigInteger(_)
        | Expr::Real(_)
        | Expr::BigFloat(..)
    ) || matches!(e, Expr::FunctionCall { name, args } if name == "Rational" && args.len() == 2)
      || crate::functions::predicate_ast::is_complex_number(e)
  };
  let a_num = if is_number_literal(a) {
    expr_to_complex_parts(a)
  } else {
    None
  };
  let b_num = if is_number_literal(b) {
    expr_to_complex_parts(b)
  } else {
    None
  };

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
          match a_im.partial_cmp(&b_im).unwrap_or(std::cmp::Ordering::Equal) {
            std::cmp::Ordering::Equal => {
              // Numerically equal number atoms tie-break by type:
              // Integer before Real before Rational. wolframscript:
              // Sort[{1., 1}] = {1, 1.}, Sort[{3/2, 1.5}] = {1.5, 3/2}.
              let type_rank = |e: &Expr| -> Option<u8> {
                match e {
                  Expr::Integer(_) | Expr::BigInteger(_) => Some(0),
                  Expr::Real(_) | Expr::BigFloat(..) => Some(1),
                  Expr::FunctionCall { name, args }
                    if name == "Rational" && args.len() == 2 =>
                  {
                    Some(2)
                  }
                  _ => None,
                }
              };
              match (type_rank(a), type_rank(b)) {
                (Some(ra), Some(rb)) => ra.cmp(&rb),
                _ => std::cmp::Ordering::Equal,
              }
            }
            other => other,
          }
        }
        other => other,
      }
    }
    (Some(_), None) => std::cmp::Ordering::Less, // numbers before non-numbers
    (None, Some(_)) => std::cmp::Ordering::Greater,
    (None, None) => {
      // Number-literal coefficients strip off before comparing — Wolfram's
      // canonical order gives Sort[{Sqrt[11], 2*Sqrt[2]}] =
      // {2*Sqrt[2], Sqrt[11]} — and only break a tie on the symbolic
      // part, ascending (Sort[{2*x, x}] = {x, 2*x}).
      {
        let (ca, ra) = numeric_coeff_and_rest_expr(a);
        let (cb, rb) = numeric_coeff_and_rest_expr(b);
        if ra.is_some() || rb.is_some() {
          let ord =
            canonical_cmp(ra.as_ref().unwrap_or(a), rb.as_ref().unwrap_or(b));
          if ord != std::cmp::Ordering::Equal {
            return ord;
          }
          if ca != cb {
            return if ca < cb {
              std::cmp::Ordering::Less
            } else {
              std::cmp::Ordering::Greater
            };
          }
        }
      }
      // Powers of integer bases order by base then exponent, ascending
      // (Sort[{Sqrt[11], Sqrt[2]}] = {Sqrt[2], Sqrt[11]}); the
      // head-name/argument comparison below would order 11 before 2.
      if let (Some((ba, ea)), Some((bb, eb))) =
        (int_base_power(a), int_base_power(b))
      {
        match ba.cmp(&bb) {
          std::cmp::Ordering::Equal => {}
          other => return other,
        }
        if ea != eb {
          return if ea < eb {
            std::cmp::Ordering::Less
          } else {
            std::cmp::Ordering::Greater
          };
        }
        return std::cmp::Ordering::Equal;
      }
      // Handle compound expressions (lists, function calls) element-wise
      match (a, b) {
        // Both lists: Wolfram's canonical order compares by length first, then
        // element by element — so `{2}` and `{3}` precede `{1, 2}`.
        (Expr::List(a_items), Expr::List(b_items)) => {
          match a_items.len().cmp(&b_items.len()) {
            std::cmp::Ordering::Equal => {}
            other => return other,
          }
          for (ai, bi) in a_items.iter().zip(b_items.iter()) {
            let ord = canonical_cmp(ai, bi);
            if ord != std::cmp::Ordering::Equal {
              return ord;
            }
          }
          return std::cmp::Ordering::Equal;
        }
        // Both function calls: compare by name first, then args element-wise
        (
          Expr::FunctionCall {
            name: a_name,
            args: a_args,
          },
          Expr::FunctionCall {
            name: b_name,
            args: b_args,
          },
        ) => {
          let name_ord = wolfram_string_cmp(a_name, b_name);
          if name_ord != std::cmp::Ordering::Equal {
            return name_ord;
          }
          // Same head: Wolfram orders by argument count first, then argument
          // by argument (so f[3] precedes f[1, 2]).
          match a_args.len().cmp(&b_args.len()) {
            std::cmp::Ordering::Equal => {}
            other => return other,
          }
          for (ai, bi) in a_args.iter().zip(b_args.iter()) {
            let ord = canonical_cmp(ai, bi);
            if ord != std::cmp::Ordering::Equal {
              return ord;
            }
          }
          return std::cmp::Ordering::Equal;
        }
        // Lists sort after non-lists
        (Expr::List(_), _) => return std::cmp::Ordering::Greater,
        (_, Expr::List(_)) => return std::cmp::Ordering::Less,
        // Function calls sort after atoms but before lists
        (Expr::FunctionCall { .. }, _) => return std::cmp::Ordering::Greater,
        (_, Expr::FunctionCall { .. }) => return std::cmp::Ordering::Less,
        // Pattern-vs-Pattern: order by the head name (e.g. `_Integer` <
        // `_Symbol` because `Integer` < `Symbol` alphabetically), then by
        // the optional pattern variable name. Matches wolframscript's
        // `Sort[{_Symbol, _Integer}]` = `{_Integer, _Symbol}`.
        (
          Expr::Pattern {
            name: a_name,
            head: a_head,
            blank_type: a_bt,
          },
          Expr::Pattern {
            name: b_name,
            head: b_head,
            blank_type: b_bt,
          },
        ) => {
          let a_h = a_head.as_deref().unwrap_or("");
          let b_h = b_head.as_deref().unwrap_or("");
          let head_ord = wolfram_string_cmp(a_h, b_h);
          if head_ord != std::cmp::Ordering::Equal {
            return head_ord;
          }
          let bt_ord = a_bt.cmp(b_bt);
          if bt_ord != std::cmp::Ordering::Equal {
            return bt_ord;
          }
          return wolfram_string_cmp(a_name, b_name);
        }
        // Patterns (`_Symbol`, `x_Integer`, etc.) sort like function calls:
        // an atomic Identifier comes before a Pattern. Matches
        // wolframscript's `Sort[{a, _Symbol, _Integer}]` =
        // `{a, _Integer, _Symbol}`.
        (Expr::Pattern { .. }, _) => return std::cmp::Ordering::Greater,
        (_, Expr::Pattern { .. }) => return std::cmp::Ordering::Less,
        _ => {}
      }

      // Atomic non-numeric: string/symbol comparison
      let sa = crate::syntax::expr_to_string(a);
      let sb = crate::syntax::expr_to_string(b);
      wolfram_string_cmp(&sa, &sb)
    }
  }
}

/// AST-based SortBy: sort elements by the value of a function.
/// SortBy[{a, b, c}, f] -> elements sorted by f[x]
/// Compute the sort key for `item` under `func`. A list of functions
/// `{f1, …, fn}` yields the tuple `{f1[item], …, fn[item]}`, giving a
/// lexicographic multi-criteria sort; any other `func` is applied directly.
fn sort_key(func: &Expr, item: &Expr) -> Result<Expr, InterpreterError> {
  if let Expr::List(funcs) = func {
    let keys: Vec<Expr> = funcs
      .iter()
      .map(|f| apply_func_ast(f, item))
      .collect::<Result<_, InterpreterError>>()?;
    Ok(Expr::List(keys.into()))
  } else {
    apply_func_ast(func, item)
  }
}

pub fn sort_by_ast(list: &Expr, func: &Expr) -> Result<Expr, InterpreterError> {
  match list {
    Expr::List(items) => {
      let mut keyed: Vec<(Expr, Expr)> = items
        .iter()
        .map(|item| {
          let key = sort_key(func, item)?;
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
          let key = sort_key(func, v)?;
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
          let key = sort_key(func, item)?;
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
    other => {
      if is_atomic_arg(other) {
        emit_nonatomic_normal_message("SortBy", &[list.clone(), func.clone()]);
      }
      Ok(Expr::FunctionCall {
        name: "SortBy".to_string(),
        args: vec![list.clone(), func.clone()].into(),
      })
    }
  }
}

///// Ordering[list] / Ordering[list, n] / Ordering[list, n, p]
/// PositionLargest[list] / PositionSmallest[list]: the 1-based positions of all
/// occurrences of the maximum (resp. minimum) element of a numeric list, in
/// ascending order. Non-numeric lists are left unevaluated. (The 2-argument
/// n-extrema form is not handled here.)
pub fn position_extreme_ast(
  args: &[Expr],
  largest: bool,
) -> Result<Expr, InterpreterError> {
  let name = if largest {
    "PositionLargest"
  } else {
    "PositionSmallest"
  };
  let unevaluated = || {
    Ok(Expr::FunctionCall {
      name: name.to_string(),
      args: args.to_vec().into(),
    })
  };
  let Some(Expr::List(items)) = args.first() else {
    return unevaluated();
  };
  if items.is_empty() {
    return unevaluated();
  }
  let mut vals = Vec::with_capacity(items.len());
  for it in items.iter() {
    match crate::functions::math_ast::try_eval_to_f64(it) {
      Some(v) => vals.push(v),
      None => return unevaluated(),
    }
  }
  let target = if largest {
    vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
  } else {
    vals.iter().cloned().fold(f64::INFINITY, f64::min)
  };
  let positions: Vec<Expr> = vals
    .iter()
    .enumerate()
    .filter(|(_, v)| **v == target)
    .map(|(i, _)| Expr::Integer((i + 1) as i128))
    .collect();
  Ok(Expr::List(positions.into()))
}

pub fn ordering_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() || args.len() > 3 {
    return Err(InterpreterError::EvaluationError(
      "Ordering expects 1, 2, or 3 arguments".into(),
    ));
  }

  // Ordering works on any nonatomic expression; atoms emit ::normal.
  // An association is ordered by its values (the result is positional indices,
  // not keys): Ordering[<|a->3, b->1, c->2|>] -> {2, 3, 1}.
  let assoc_values: Vec<Expr>;
  let items: &[Expr] = match &args[0] {
    Expr::List(items) => items.as_slice(),
    Expr::Association(pairs) => {
      assoc_values = pairs.iter().map(|(_, v)| v.clone()).collect();
      &assoc_values
    }
    // Rational/Complex are atoms — their internal args have no ordering.
    Expr::FunctionCall { args: fc_args, .. } if !is_atomic_arg(&args[0]) => {
      fc_args.as_slice()
    }
    _ => {
      emit_nonatomic_normal_message("Ordering", args);
      return Ok(Expr::FunctionCall {
        name: "Ordering".to_string(),
        args: args.to_vec().into(),
      });
    }
  };

  let mut indexed: Vec<(usize, &Expr)> = items.iter().enumerate().collect();

  // With 3 args the third is an ordering function p (a bare symbol like
  // Less/Greater, or a predicate). p[a, b] is applied to each pair: a comes
  // before b when it is True, after when False, and when p[a, b] yields
  // neither (e.g. Less on non-numeric symbols, where `c < a` stays symbolic)
  // the pair is treated as incomparable so the original order is kept — this
  // is why `Ordering[{c, a, b}, All, Less]` is {1, 2, 3}, not the canonical
  // {2, 3, 1}. Without a comparator, the default canonical order is used.
  if let Some(comparator) = args.get(2) {
    let p = comparator.clone();
    let mut err: Option<InterpreterError> = None;
    // Apply the comparator both ways so the ordering is consistent: a < b when
    // p[a, b] is True, a > b when p[b, a] is True, and otherwise the pair is
    // incomparable (equal sort keys, or a symbolic non-Boolean result) and the
    // stable sort keeps the original order.
    let is_true = |e: &Expr| matches!(e, Expr::Identifier(s) if s == "True");
    indexed.sort_by(|a, b| {
      if err.is_some() {
        return std::cmp::Ordering::Equal;
      }
      let ab = crate::functions::list_helpers_ast::apply_func_to_two_args(
        &p, a.1, b.1,
      );
      match ab {
        Ok(ref r) if is_true(r) => return std::cmp::Ordering::Less,
        Ok(_) => {}
        Err(e) => {
          err = Some(e);
          return std::cmp::Ordering::Equal;
        }
      }
      match crate::functions::list_helpers_ast::apply_func_to_two_args(
        &p, b.1, a.1,
      ) {
        Ok(ref r) if is_true(r) => std::cmp::Ordering::Greater,
        Ok(_) => std::cmp::Ordering::Equal,
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
      crate::functions::list_helpers_ast::canonical_cmp(a.1, b.1)
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

  Ok(Expr::List(result.into()))
}

/// OrderingBy[list, f] / OrderingBy[list, f, n] — the positions that order
/// `list` by `f` applied to each element (ascending, stable). `n` limits the
/// number of positions: positive keeps the first `n`, negative the last `|n|`,
/// `All` keeps them all.
pub fn ordering_by_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 2 || args.len() > 3 {
    return Err(InterpreterError::EvaluationError(
      "OrderingBy expects 2 or 3 arguments".into(),
    ));
  }
  let items: &[Expr] = match &args[0] {
    Expr::List(items) => items.as_slice(),
    Expr::FunctionCall { args: fc_args, .. } => fc_args.as_slice(),
    _ => {
      return Ok(Expr::FunctionCall {
        name: "OrderingBy".to_string(),
        args: args.to_vec().into(),
      });
    }
  };

  let func = &args[1];
  let mut keyed: Vec<(usize, Expr)> = Vec::with_capacity(items.len());
  for (i, item) in items.iter().enumerate() {
    keyed.push((i, apply_func_ast(func, item)?));
  }
  keyed.sort_by(|a, b| by_key_cmp(&a.1, &b.1));

  let mut result: Vec<Expr> = keyed
    .iter()
    .map(|(idx, _)| Expr::Integer((*idx + 1) as i128))
    .collect();

  if args.len() == 3 {
    let is_all = matches!(&args[2], Expr::Identifier(n) if n == "All");
    if !is_all && let Some(n) = expr_to_i128(&args[2]) {
      if n >= 0 {
        result.truncate(n as usize);
      } else {
        let abs_n = n.unsigned_abs() as usize;
        if abs_n <= result.len() {
          result = result.split_off(result.len() - abs_n);
        }
      }
    }
  }

  Ok(Expr::List(result.into()))
}

/// Comparator for *By key expressions: numeric when possible, lexicographic fallback.
fn by_key_cmp(a: &Expr, b: &Expr) -> std::cmp::Ordering {
  // Exact ordering for integer/rational keys first: parsing them as f64
  // (below) collapses any magnitude beyond ~1.8e308 to inf, so e.g. a
  // 1348-digit and a 2847-digit denominator would compare equal — making
  // MaximalBy/MinimalBy pick the wrong element on huge BigIntegers.
  if let Some(ord) = exact_real_cmp(a, b) {
    return ord;
  }
  let ka = crate::syntax::expr_to_string(a);
  let kb = crate::syntax::expr_to_string(b);
  if let (Ok(na), Ok(nb)) = (ka.parse::<f64>(), kb.parse::<f64>()) {
    na.partial_cmp(&nb).unwrap_or(std::cmp::Ordering::Equal)
  } else {
    ka.cmp(&kb)
  }
}

/// MinimalBy[list, f] - Returns all elements that minimize f
/// MinimalBy[list, f, n] - Returns the n elements with smallest f values
/// MinimalBy/MaximalBy over an association: rank each (key, value) pair by
/// `func` applied to the *value*, returning an association of the selected
/// pairs. Without `n`, keep every pair tying the extreme value (in original
/// order). With `n`, keep the `n` pairs sorted by the criterion (stable).
fn minimal_maximal_by_assoc(
  pairs: &[(Expr, Expr)],
  func: &Expr,
  n: Option<i128>,
  maximal: bool,
) -> Result<Expr, InterpreterError> {
  if pairs.is_empty() {
    return Ok(Expr::Association(vec![]));
  }
  // (key, value, criterion = func[value])
  let keyed: Vec<((Expr, Expr), Expr)> = pairs
    .iter()
    .map(|(k, v)| Ok(((k.clone(), v.clone()), apply_func_ast(func, v)?)))
    .collect::<Result<_, InterpreterError>>()?;

  match n {
    Some(n_val) => {
      let mut indexed: Vec<(usize, &Expr)> =
        keyed.iter().enumerate().map(|(i, (_, c))| (i, c)).collect();
      indexed.sort_by(|(_, a), (_, b)| {
        if maximal {
          by_key_cmp(b, a)
        } else {
          by_key_cmp(a, b)
        }
      });
      let take = (n_val.max(0) as usize).min(keyed.len());
      let result: Vec<(Expr, Expr)> = indexed
        .into_iter()
        .take(take)
        .map(|(i, _)| keyed[i].0.clone())
        .collect();
      Ok(Expr::Association(result))
    }
    None => {
      let extreme = keyed
        .iter()
        .map(|(_, c)| c)
        .min_by(|a, b| {
          if maximal {
            by_key_cmp(b, a)
          } else {
            by_key_cmp(a, b)
          }
        })
        .cloned();
      let result: Vec<(Expr, Expr)> = match extreme {
        Some(ex) => {
          let ex_str = crate::syntax::expr_to_string(&ex);
          keyed
            .into_iter()
            .filter(|(_, c)| crate::syntax::expr_to_string(c) == ex_str)
            .map(|(kv, _)| kv)
            .collect()
        }
        None => vec![],
      };
      Ok(Expr::Association(result))
    }
  }
}

pub fn minimal_by_ast(
  list: &Expr,
  func: &Expr,
  n: Option<i128>,
) -> Result<Expr, InterpreterError> {
  // Association form: rank by f applied to each value, return an association.
  if let Expr::Association(pairs) = list {
    return minimal_maximal_by_assoc(pairs, func, n, false);
  }
  let items = match list {
    Expr::List(items) if !items.is_empty() => items,
    Expr::List(_) => return Ok(Expr::List(vec![].into())),
    _ => {
      let mut args = vec![list.clone(), func.clone()];
      if let Some(nv) = n {
        args.push(Expr::Integer(nv));
      }
      return Ok(Expr::FunctionCall {
        name: "MinimalBy".to_string(),
        args: args.into(),
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

  match n {
    Some(n_val) => {
      // Sort by key ascending, take n elements
      let mut indexed: Vec<(usize, &Expr)> =
        keyed.iter().enumerate().map(|(i, (_, k))| (i, k)).collect();
      indexed.sort_by(|(_, a), (_, b)| by_key_cmp(a, b));
      let take = (n_val as usize).min(keyed.len());
      let result: Vec<Expr> = indexed
        .into_iter()
        .take(take)
        .map(|(i, _)| keyed[i].0.clone())
        .collect();
      Ok(Expr::List(result.into()))
    }
    None => {
      let min_key = keyed
        .iter()
        .map(|(_, k)| k)
        .min_by(|a, b| by_key_cmp(a, b))
        .cloned();

      if let Some(min_k) = min_key {
        let min_str = crate::syntax::expr_to_string(&min_k);
        let result: Vec<Expr> = keyed
          .into_iter()
          .filter(|(_, k)| crate::syntax::expr_to_string(k) == min_str)
          .map(|(item, _)| item)
          .collect();
        Ok(Expr::List(result.into()))
      } else {
        Ok(Expr::List(vec![].into()))
      }
    }
  }
}

/// MaximalBy[list, f] - Returns all elements that maximize f
/// MaximalBy[list, f, n] - Returns the n elements with largest f values
pub fn maximal_by_ast(
  list: &Expr,
  func: &Expr,
  n: Option<i128>,
) -> Result<Expr, InterpreterError> {
  // Association form: rank by f applied to each value, return an association.
  if let Expr::Association(pairs) = list {
    return minimal_maximal_by_assoc(pairs, func, n, true);
  }
  let items = match list {
    Expr::List(items) if !items.is_empty() => items,
    Expr::List(_) => return Ok(Expr::List(vec![].into())),
    _ => {
      let mut args = vec![list.clone(), func.clone()];
      if let Some(nv) = n {
        args.push(Expr::Integer(nv));
      }
      return Ok(Expr::FunctionCall {
        name: "MaximalBy".to_string(),
        args: args.into(),
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

  match n {
    Some(n_val) => {
      // Sort by key descending, take n elements
      let mut indexed: Vec<(usize, &Expr)> =
        keyed.iter().enumerate().map(|(i, (_, k))| (i, k)).collect();
      indexed.sort_by(|(_, a), (_, b)| by_key_cmp(b, a));
      let take = (n_val as usize).min(keyed.len());
      let result: Vec<Expr> = indexed
        .into_iter()
        .take(take)
        .map(|(i, _)| keyed[i].0.clone())
        .collect();
      Ok(Expr::List(result.into()))
    }
    None => {
      let max_key = keyed
        .iter()
        .map(|(_, k)| k)
        .max_by(|a, b| by_key_cmp(a, b))
        .cloned();

      if let Some(max_k) = max_key {
        let max_str = crate::syntax::expr_to_string(&max_k);
        let result: Vec<Expr> = keyed
          .into_iter()
          .filter(|(_, k)| crate::syntax::expr_to_string(k) == max_str)
          .map(|(item, _)| item)
          .collect();
        Ok(Expr::List(result.into()))
      } else {
        Ok(Expr::List(vec![].into()))
      }
    }
  }
}

/// AST-based Sort: sort a list.
/// Whether `e` is an atomic argument for which list functions emit
/// `::normal` (numbers, strings, symbols, constants). Lists, function calls,
/// and associations are nonatomic and operable. Rational and Complex are
/// atoms despite their FunctionCall storage (AtomQ[5/3] = True), so list
/// functions must not operate on their internal arguments.
pub fn is_atomic_arg(e: &Expr) -> bool {
  matches!(
    e,
    Expr::Integer(_)
      | Expr::BigInteger(_)
      | Expr::Real(_)
      | Expr::BigFloat(_, _)
      | Expr::String(_)
      | Expr::Identifier(_)
      | Expr::Constant(_)
  ) || matches!(e, Expr::FunctionCall { name, args }
      if (name == "Rational" || name == "Complex") && args.len() == 2)
    || crate::functions::predicate_ast::is_complex_number(e)
}

/// Emit `<F>::normal: Nonatomic expression expected at position 1 in <call>.`,
/// matching wolframscript for list functions applied to an atom. The call
/// renders in 2D OutputForm — a rational argument spans three lines with the
/// message text on the baseline, exactly as wolframscript prints it.
pub fn emit_nonatomic_normal_message(name: &str, args: &[Expr]) {
  crate::emit_message(&crate::syntax::format_message_with_expr(
    &format!(
      "{}::normal: Nonatomic expression expected at position 1 in ",
      name
    ),
    &Expr::FunctionCall {
      name: name.to_string(),
      args: args.to_vec().into(),
    },
    ".",
  ));
}

/// Purely lexicographic comparison: unlike canonical `Sort`, lists are
/// compared element by element (shorter lists are a tie-break, not pulled to
/// the front), so `{2}` sorts between `{1, 9}` and `{3}`. Non-list expressions
/// fall back to the canonical order.
fn lexicographic_cmp(a: &Expr, b: &Expr) -> std::cmp::Ordering {
  if let (Expr::List(la), Expr::List(lb)) = (a, b) {
    for (ai, bi) in la.iter().zip(lb.iter()) {
      let ord = lexicographic_cmp(ai, bi);
      if ord != std::cmp::Ordering::Equal {
        return ord;
      }
    }
    return la.len().cmp(&lb.len());
  }
  canonical_cmp(a, b)
}

/// `LexicographicSort[list]` — sort by the purely lexicographic order.
pub fn lexicographic_sort_ast(list: &Expr) -> Result<Expr, InterpreterError> {
  match list {
    Expr::List(items) => {
      let mut sorted = items.clone();
      sorted.sort_by(lexicographic_cmp);
      Ok(Expr::List(sorted))
    }
    _ => sort_ast(list),
  }
}

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
    // Rational/Complex are atoms — their internal args must not sort
    // (Sort[5/3] emits Sort::normal, it does not become 3/5).
    Expr::FunctionCall { name, args } if !is_atomic_arg(list) => {
      let mut sorted = args.clone();
      sorted.sort_by(canonical_cmp);
      Ok(Expr::FunctionCall {
        name: name.clone(),
        args: sorted,
      })
    }
    other => {
      if is_atomic_arg(other) {
        emit_nonatomic_normal_message("Sort", &[other.clone()]);
      }
      Ok(Expr::FunctionCall {
        name: "Sort".to_string(),
        args: vec![list.clone()].into(),
      })
    }
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
      args: args.to_vec().into(),
    })
  }
}

/// Compare two Expr values for canonical ordering.
/// Returns 1 if a < b, -1 if a > b, 0 if equal (Wolfram Order convention).
pub fn compare_exprs(a: &Expr, b: &Expr) -> i64 {
  use crate::functions::math_ast::try_eval_to_f64_with_infinity;
  // ByteArray vs ByteArray: compare by decoded byte payload, not by the
  // wrapping `ByteArray["<base64>"]` string. wolframscript:
  //   Order[ByteArray[{1, 99}], ByteArray[{2, 0}]] = 1
  // because the first byte 1 < 2.
  if let (Some(a_bytes), Some(b_bytes)) =
    (decode_byte_array(a), decode_byte_array(b))
  {
    return match a_bytes.as_slice().cmp(b_bytes.as_slice()) {
      std::cmp::Ordering::Less => 1,
      std::cmp::Ordering::Greater => -1,
      std::cmp::Ordering::Equal => 0,
    };
  }
  // Only NUMBER LITERALS compare by value — symbolic constants (Pi, E)
  // and numeric composites (2*Pi, Sqrt[2]) sort structurally after all
  // numbers in Wolfram's canonical order (Sort[{Pi, 7}] = {7, Pi}).
  // Complex literals order by real part, then imaginary part
  // (Sort[{I, -I, 1, 1 + I}] = {-I, I, 1, 1 + I}).
  let literal_parts = |e: &Expr| -> Option<(f64, f64)> {
    match e {
      Expr::Integer(_)
      | Expr::BigInteger(_)
      | Expr::Real(_)
      | Expr::BigFloat(..) => {
        try_eval_to_f64_with_infinity(e).map(|v| (v, 0.0))
      }
      Expr::FunctionCall { name, args }
        if name == "Rational" && args.len() == 2 =>
      {
        try_eval_to_f64_with_infinity(e).map(|v| (v, 0.0))
      }
      _ if crate::functions::predicate_ast::is_complex_number(e) => {
        crate::functions::math_ast::try_extract_complex_float(e)
      }
      _ => None,
    }
  };
  let a_lit = literal_parts(a);
  let b_lit = literal_parts(b);
  if let (Some((ar, ai)), Some((br, bi))) = (a_lit, b_lit) {
    if ar != br {
      return if ar < br { 1 } else { -1 };
    }
    if ai != bi {
      return if ai < bi { 1 } else { -1 };
    }
    // Numerically equal number atoms tie-break by type: Integer
    // before Real before Rational. wolframscript: Sort[{1., 1}] =
    // {1, 1.} and Sort[{3/2, 1.5}] = {1.5, 3/2}.
    let type_rank = |e: &Expr| -> Option<i64> {
      match e {
        Expr::Integer(_) | Expr::BigInteger(_) => Some(0),
        Expr::Real(_) | Expr::BigFloat(..) => Some(1),
        Expr::FunctionCall { name, args }
          if name == "Rational" && args.len() == 2 =>
        {
          Some(2)
        }
        _ => None,
      }
    };
    return match (type_rank(a), type_rank(b)) {
      (Some(ra), Some(rb)) if ra < rb => 1,
      (Some(ra), Some(rb)) if ra > rb => -1,
      _ => 0,
    };
  }
  // Numbers come before non-numbers
  if a_lit.is_some() {
    return 1;
  }
  if b_lit.is_some() {
    return -1;
  }
  // Directed infinities (Infinity, -Infinity, ComplexInfinity) order as
  // their DirectedInfinity[...] head: after symbols, among composites,
  // with -Infinity before Infinity (Sort[{-Infinity, 5}] = {5, -Infinity}).
  let normalize_infinity = |e: &Expr| -> Option<Expr> {
    if matches!(e, Expr::FunctionCall { name, .. } if name == "DirectedInfinity")
    {
      return None; // already in canonical form
    }
    if !crate::functions::predicate_ast::is_directed_infinity(e) {
      return None;
    }
    let dir = if matches!(e, Expr::Identifier(s) | Expr::Constant(s) if s == "ComplexInfinity")
    {
      vec![]
    } else if crate::functions::math_ast::is_neg_infinity(e) {
      vec![Expr::Integer(-1)]
    } else {
      vec![Expr::Integer(1)]
    };
    Some(Expr::FunctionCall {
      name: "DirectedInfinity".to_string(),
      args: dir.into(),
    })
  };
  let a_inf = normalize_infinity(a);
  let b_inf = normalize_infinity(b);
  if a_inf.is_some() || b_inf.is_some() {
    return compare_exprs(
      a_inf.as_ref().unwrap_or(a),
      b_inf.as_ref().unwrap_or(b),
    );
  }

  // Two Lists: Wolfram's canonical order compares expressions with the same
  // head by length first, then element by element — so `{2}` and `{3}` (each
  // length 1) precede `{1, 2}` (length 2). Without this, Lists fell through to
  // a lexicographic string comparison that ordered `{1, 2}` first.
  if let (Expr::List(la), Expr::List(lb)) = (a, b) {
    match la.len().cmp(&lb.len()) {
      std::cmp::Ordering::Less => return 1,
      std::cmp::Ordering::Greater => return -1,
      std::cmp::Ordering::Equal => {}
    }
    for (ai, bi) in la.iter().zip(lb.iter()) {
      let ord = compare_exprs(ai, bi);
      if ord != 0 {
        return ord;
      }
    }
    return 0;
  }

  // Wolfram's canonical order compares terms with their number-literal
  // coefficients stripped first — Order[2*Sqrt[2], Sqrt[11]] = 1 because
  // Sqrt[2] < Sqrt[11], Order[-Sqrt[11], 2*Sqrt[6]] = -1 — and only breaks
  // a tie on the symbolic part by the coefficients, ascending:
  // Order[-6*x^2, -5*x^2] = 1, Order[-x, x] = 1, Order[2*x, x] = -1
  // (all wolframscript-verified).
  {
    let (ca, ra) = numeric_coeff_and_rest_expr(a);
    let (cb, rb) = numeric_coeff_and_rest_expr(b);
    if ra.is_some() || rb.is_some() {
      let ord = compare_exprs(ra.as_ref().unwrap_or(a), rb.as_ref().unwrap_or(b));
      if ord != 0 {
        return ord;
      }
      if ca != cb {
        return if ca < cb { 1 } else { -1 };
      }
      // Coefficient and symbolic part both tie (e.g. 2*x vs 2.*x): fall
      // through so the structural comparison below can still distinguish
      // numerically-equal-but-distinct coefficient types.
    }
  }

  // Two powers of integer bases with numeric exponents compare by base
  // ascending, then exponent ascending: Order[Sqrt[2], Sqrt[11]] = 1,
  // Order[2^(1/3), Sqrt[2]] = 1, Order[Sqrt[2], 2^(2/3)] = 1 (all
  // wolframscript-verified). The string comparisons below would order
  // "11" before "2" lexicographically.
  if let (Some((ba, ea)), Some((bb, eb))) =
    (int_base_power(a), int_base_power(b))
  {
    if ba != bb {
      return if ba < bb { 1 } else { -1 };
    }
    if ea != eb {
      return if ea < eb { 1 } else { -1 };
    }
    return 0;
  }

  // Wolfram canonical ordering: symbols and compounds are compared structurally
  // Classification: atom-like (atoms, constants, powers) sort before function calls
  let a_is_atom = is_atom_expr(a);
  let b_is_atom = is_atom_expr(b);
  let a_is_power = is_power_expr(a);
  let b_is_power = is_power_expr(b);
  let a_is_func_call = !a_is_atom && !a_is_power && is_plain_func_call(a);
  let b_is_func_call = !b_is_atom && !b_is_power && is_plain_func_call(b);
  // Patterns (`_Symbol`, `x_`, `x_Integer`, etc.) sort like function calls
  // for canonical-order purposes: an atom always comes before a Pattern,
  // matching wolframscript's `Sort[{a, _Symbol, _Integer}]` =
  // `{a, _Integer, _Symbol}`.
  let a_is_pattern = matches!(a, Expr::Pattern { .. });
  let b_is_pattern = matches!(b, Expr::Pattern { .. });

  // Atoms and powers always sort before plain function calls
  let a_is_atom_like = a_is_atom || a_is_power;
  let b_is_atom_like = b_is_atom || b_is_power;
  let a_is_compound = a_is_func_call || a_is_pattern;
  let b_is_compound = b_is_func_call || b_is_pattern;

  if a_is_atom_like && b_is_compound {
    1 // atom/power always before function call / pattern
  } else if a_is_compound && b_is_atom_like {
    -1 // function call / pattern always after atom/power
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
        // A sum whose highest canonical term is the NEGATED atom sorts
        // before the atom (coefficient -1 < 1 on the tied term), matching
        // wolframscript: Gamma[a - s]*Gamma[s], but Gamma[s]*Gamma[2 + s].
        if sum_highest_term_negates(b, a) {
          return -1;
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
        if sum_highest_term_negates(a, b) {
          return 1;
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
        // Same-head plain function calls: compare arguments structurally.
        // This matches Wolfram's canonical ordering, e.g. Cos[x] before
        // Cos[Cos[x]] (because x < Cos[x] — atoms precede function calls).
        if let (
          Expr::FunctionCall { name: na, args: aa },
          Expr::FunctionCall { name: nb, args: ab },
        ) = (a, b)
          && na == nb
          && a_is_func_call
          && b_is_func_call
        {
          // Same head: order by argument count first, then argument by
          // argument (Wolfram: f[3] precedes f[1, 2]).
          match aa.len().cmp(&ab.len()) {
            std::cmp::Ordering::Less => return 1,
            std::cmp::Ordering::Greater => return -1,
            std::cmp::Ordering::Equal => {}
          }
          for (ai, bi) in aa.iter().zip(ab.iter()) {
            let ord = compare_exprs(ai, bi);
            if ord != 0 {
              return ord;
            }
          }
          return 0;
        }
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

/// A power of an integer base with a numeric exponent — Sqrt[2],
/// 2^(1/3), Power[5, -1] — as `(base, exponent)`. These compare by base
/// ascending then exponent ascending in Wolfram's canonical order.
fn int_base_power(e: &Expr) -> Option<(i128, f64)> {
  use crate::functions::math_ast::try_eval_to_f64_with_infinity;
  let (base, exp) = match e {
    Expr::FunctionCall { name, args } if name == "Sqrt" && args.len() == 1 => {
      (args[0].clone(), Expr::FunctionCall {
        name: "Rational".to_string(),
        args: vec![Expr::Integer(1), Expr::Integer(2)].into(),
      })
    }
    Expr::FunctionCall { name, args } if name == "Power" && args.len() == 2 => {
      (args[0].clone(), args[1].clone())
    }
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Power,
      left,
      right,
    } => ((**left).clone(), (**right).clone()),
    _ => return None,
  };
  let Expr::Integer(b) = base else { return None };
  let e = try_eval_to_f64_with_infinity(&exp)?;
  Some((b, e))
}

/// Split a term into its leading number-literal coefficient and the
/// remaining symbolic factors. Returns `(coefficient, Some(rest))` when a
/// literal coefficient (or unary minus) was actually stripped, and
/// `(1.0, None)` when the term carries no literal coefficient — so callers
/// can tell `x` (nothing stripped) apart from `1.0*x`.
fn numeric_coeff_and_rest_expr(e: &Expr) -> (f64, Option<Expr>) {
  use crate::functions::math_ast::try_eval_to_f64_with_infinity;
  let literal = |x: &Expr| -> Option<f64> {
    match x {
      Expr::Integer(_)
      | Expr::BigInteger(_)
      | Expr::Real(_)
      | Expr::BigFloat(..) => try_eval_to_f64_with_infinity(x),
      Expr::FunctionCall { name, args }
        if name == "Rational" && args.len() == 2 =>
      {
        try_eval_to_f64_with_infinity(x)
      }
      _ => None,
    }
  };
  match e {
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => {
      let (c, r) = numeric_coeff_and_rest_expr(operand);
      (-c, Some(r.unwrap_or_else(|| (**operand).clone())))
    }
    Expr::FunctionCall { name, args } if name == "Times" && args.len() >= 2 => {
      if let Some(c) = literal(&args[0]) {
        let rest = if args.len() == 2 {
          args[1].clone()
        } else {
          Expr::FunctionCall {
            name: "Times".to_string(),
            args: args[1..].to_vec().into(),
          }
        };
        (c, Some(rest))
      } else {
        (1.0, None)
      }
    }
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => {
      if let Some(c) = literal(left) {
        let (cr, r) = numeric_coeff_and_rest_expr(right);
        (c * cr, Some(r.unwrap_or_else(|| (**right).clone())))
      } else {
        (1.0, None)
      }
    }
    _ => (1.0, None),
  }
}

/// True when `sum` is an additive expression whose highest (last) canonical
/// term is exactly the negation of `atom`. Wolfram's canonical order compares
/// sums by their highest term first, and on a symbol tie the negative
/// coefficient sorts first — so `a - s` and `2 - s` sort before `s`, while
/// `2 + s` sorts after it.
fn sum_highest_term_negates(sum: &Expr, atom: &Expr) -> bool {
  let last = match sum {
    Expr::FunctionCall { name, args } if name == "Plus" && args.len() >= 2 => {
      args.last().cloned()
    }
    Expr::BinaryOp {
      op: BinaryOperator::Minus,
      right,
      ..
    } => Some(Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand: right.clone(),
    }),
    Expr::BinaryOp {
      op: BinaryOperator::Plus,
      right,
      ..
    } => Some((**right).clone()),
    _ => None,
  };
  let Some(last) = last else { return false };
  let negated_inner = match &last {
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => Some((**operand).clone()),
    Expr::FunctionCall { name, args }
      if name == "Times"
        && args.len() == 2
        && matches!(&args[0], Expr::Integer(n) if *n < 0) =>
    {
      Some(args[1].clone())
    }
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } if matches!(left.as_ref(), Expr::Integer(n) if *n < 0) => {
      Some((**right).clone())
    }
    _ => None,
  };
  match negated_inner {
    Some(inner) => {
      crate::syntax::expr_to_string(&inner)
        == crate::syntax::expr_to_string(atom)
    }
    None => false,
  }
}

/// If `expr` is `ByteArray["<base64>"]`, decode and return its raw byte
/// payload. Used by `compare_exprs` so canonical-order comparisons walk
/// the underlying bytes rather than the base64 wrapper string.
fn decode_byte_array(expr: &Expr) -> Option<Vec<u8>> {
  if let Expr::FunctionCall { name, args } = expr
    && name == "ByteArray"
    && args.len() == 1
    && let Expr::String(b64) = &args[0]
  {
    use base64::Engine;
    let engine = base64::engine::general_purpose::STANDARD;
    return engine.decode(b64).ok();
  }
  None
}

/// Check if an expression is a Power (BinaryOp or FunctionCall)
fn is_power_expr(e: &Expr) -> bool {
  matches!(
    e,
    Expr::BinaryOp {
      op: BinaryOperator::Power,
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
    // For CurriedCall whose head is itself a FunctionCall (e.g. the
    // `Derivative[1][f]` shape stored as `CurriedCall { Derivative[1], [f] }`),
    // use the inner head name as the sort key. This keeps mixed flat
    // (`FunctionCall { Derivative, [1, f] }`) and curried forms in the
    // same sort bucket so Times canonical ordering can compare their
    // arguments structurally.
    Expr::CurriedCall { func, .. } => {
      if let Expr::FunctionCall { name, .. } = func.as_ref() {
        return name.clone();
      }
      crate::syntax::expr_to_string(e)
    }
    Expr::BinaryOp { op, left, right } => {
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

/// Wolfram canonical string comparison (returns std::cmp::Ordering)
fn wolfram_string_cmp(a: &str, b: &str) -> std::cmp::Ordering {
  match wolfram_string_order(a, b) {
    n if n > 0 => std::cmp::Ordering::Less,
    n if n < 0 => std::cmp::Ordering::Greater,
    _ => std::cmp::Ordering::Equal,
  }
}

/// Wolfram canonical string ordering: case-insensitive alphabetical, then lowercase < uppercase
pub fn wolfram_string_order(a: &str, b: &str) -> i64 {
  // Collation rank: Wolfram sorts the Nordic letters å/ä/ö/æ/ø after
  // `z` (in that order) and ñ after the plain Latin letters, not by
  // Unicode codepoint. wolframscript: Sort[{"ä", "å"}] = {å, ä}.
  fn collate(c: char) -> u32 {
    match c {
      'å' | 'Å' => 0x110000 + 27,
      'ä' | 'Ä' => 0x110000 + 28,
      'ö' | 'Ö' => 0x110000 + 29,
      'æ' | 'Æ' => 0x110000 + 30,
      'ø' | 'Ø' => 0x110000 + 31,
      'ñ' | 'Ñ' => 0x100000 + ('n' as u32) + 1,
      other => other as u32,
    }
  }
  let a_chars: Vec<char> = a.chars().collect();
  let b_chars: Vec<char> = b.chars().collect();

  // Pass 1: case-insensitive comparison over the whole strings, with a
  // shorter string sorting first on a prefix match. wolframscript:
  // Sort[{"MathML", "MAT"}] = {MAT, MathML} — the case difference at
  // the second letter must not outrank the length/letter comparison.
  for (ac, bc) in a_chars.iter().zip(b_chars.iter()) {
    let al = ac.to_lowercase().next().unwrap_or(*ac);
    let bl = bc.to_lowercase().next().unwrap_or(*bc);
    if collate(al) != collate(bl) {
      return if collate(al) < collate(bl) { 1 } else { -1 };
    }
  }
  match a_chars.len().cmp(&b_chars.len()) {
    std::cmp::Ordering::Less => return 1,
    std::cmp::Ordering::Greater => return -1,
    std::cmp::Ordering::Equal => {}
  }
  // Pass 2: case-insensitively equal strings tie-break at the first
  // case difference, lowercase first: Sort[{"Ab", "aB"}] = {aB, Ab}.
  for (ac, bc) in a_chars.iter().zip(b_chars.iter()) {
    if ac != bc {
      if ac.is_lowercase() && bc.is_uppercase() {
        return 1;
      } else if ac.is_uppercase() && bc.is_lowercase() {
        return -1;
      }
    }
  }
  0
}

/// Helper: compare two Expr values for ordering (less-or-equal)
fn expr_le(a: &Expr, b: &Expr) -> bool {
  // Use canonical_cmp for consistency with Sort
  !matches!(canonical_cmp(a, b), std::cmp::Ordering::Greater)
}
