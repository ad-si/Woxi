use crate::InterpreterError;
use crate::syntax::Expr;

/// FunctionExpand[expr] — expand special mathematical functions into simpler forms.
pub fn function_expand_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() {
    return Err(InterpreterError::EvaluationError(
      "FunctionExpand expects 1 argument".into(),
    ));
  }
  let expr = &args[0];
  let result = function_expand_inner(expr)?;
  // Evaluate the result to simplify
  crate::evaluator::evaluate_expr_to_expr(&result)
}

fn function_expand_inner(expr: &Expr) -> Result<Expr, InterpreterError> {
  match expr {
    Expr::FunctionCall { name, args } => {
      // First recursively expand arguments
      let expanded_args: Vec<Expr> = args
        .iter()
        .map(function_expand_inner)
        .collect::<Result<Vec<_>, _>>()?;

      // Then try to expand this function call
      if let Some(expanded) = try_expand_function(name, &expanded_args) {
        return Ok(expanded);
      }

      Ok(Expr::FunctionCall {
        name: name.clone(),
        args: expanded_args.into(),
      })
    }
    Expr::List(items) => {
      let expanded: Vec<Expr> = items
        .iter()
        .map(function_expand_inner)
        .collect::<Result<Vec<_>, _>>()?;
      Ok(Expr::List(expanded.into()))
    }
    Expr::BinaryOp { op, left, right } => {
      let l = function_expand_inner(left)?;
      let r = function_expand_inner(right)?;
      Ok(Expr::BinaryOp {
        op: *op,
        left: Box::new(l),
        right: Box::new(r),
      })
    }
    Expr::UnaryOp { op, operand } => {
      let o = function_expand_inner(operand)?;
      Ok(Expr::UnaryOp {
        op: *op,
        operand: Box::new(o),
      })
    }
    _ => Ok(expr.clone()),
  }
}

/// Helper to build common expressions
fn mk_call(name: &str, args: Vec<Expr>) -> Expr {
  Expr::FunctionCall {
    name: name.to_string(),
    args: args.into(),
  }
}

fn mk_id(name: &str) -> Expr {
  Expr::Identifier(name.to_string())
}

fn mk_int(n: i128) -> Expr {
  Expr::Integer(n)
}

fn mk_times(a: Expr, b: Expr) -> Expr {
  mk_call("Times", vec![a, b])
}

fn mk_plus(a: Expr, b: Expr) -> Expr {
  mk_call("Plus", vec![a, b])
}

fn mk_power(base: Expr, exp: Expr) -> Expr {
  mk_call("Power", vec![base, exp])
}

fn mk_div(a: Expr, b: Expr) -> Expr {
  mk_times(a, mk_power(b, mk_int(-1)))
}

fn mk_ratio(n: i128, d: i128) -> Expr {
  Expr::FunctionCall {
    name: "Rational".to_string(),
    args: vec![Expr::Integer(n), Expr::Integer(d)].into(),
  }
}

/// Try to expand a specific function call. Returns None if no expansion applies.
fn try_expand_function(name: &str, args: &[Expr]) -> Option<Expr> {
  match name {
    // Pochhammer[a, n] → Gamma[a + n] / Gamma[a]
    "Pochhammer" if args.len() == 2 => {
      let a = &args[0];
      let n = &args[1];
      Some(mk_div(
        mk_call("Gamma", vec![mk_plus(a.clone(), n.clone())]),
        mk_call("Gamma", vec![a.clone()]),
      ))
    }

    // Beta[a, b] → Gamma[a] * Gamma[b] / Gamma[a + b]
    "Beta" if args.len() == 2 => {
      let a = &args[0];
      let b = &args[1];
      Some(mk_div(
        mk_times(
          mk_call("Gamma", vec![a.clone()]),
          mk_call("Gamma", vec![b.clone()]),
        ),
        mk_call("Gamma", vec![mk_plus(a.clone(), b.clone())]),
      ))
    }

    // Factorial[n] (i.e. n!) → Gamma[1 + n]
    "Factorial" if args.len() == 1 => {
      Some(mk_call("Gamma", vec![mk_plus(mk_int(1), args[0].clone())]))
    }

    // Binomial[n, k]: a specific integer k expands to a polynomial; an
    // otherwise symbolic k expands to the Gamma-function form
    //   Gamma[1 + n] / (Gamma[1 + k] * Gamma[1 - k + n]).
    "Binomial" if args.len() == 2 => {
      if let Expr::Integer(k) = &args[1] {
        expand_binomial_integer_k(&args[0], *k)
      } else {
        let n = &args[0];
        let k = &args[1];
        Some(mk_div(
          mk_call("Gamma", vec![mk_plus(mk_int(1), n.clone())]),
          mk_times(
            mk_call("Gamma", vec![mk_plus(mk_int(1), k.clone())]),
            mk_call(
              "Gamma",
              vec![mk_call(
                "Plus",
                vec![mk_int(1), mk_times(mk_int(-1), k.clone()), n.clone()],
              )],
            ),
          ),
        ))
      }
    }

    // CatalanNumber[n] → (2^(2 n) Gamma[1/2 + n]) / (Sqrt[Pi] Gamma[2 + n])
    "CatalanNumber" if args.len() == 1 => {
      let n = &args[0];
      Some(mk_div(
        mk_times(
          mk_power(mk_int(2), mk_times(mk_int(2), n.clone())),
          mk_call("Gamma", vec![mk_plus(mk_ratio(1, 2), n.clone())]),
        ),
        mk_times(
          mk_call("Sqrt", vec![mk_id("Pi")]),
          mk_call("Gamma", vec![mk_plus(mk_int(2), n.clone())]),
        ),
      ))
    }

    // Subfactorial[n] → Gamma[1 + n, -1] / E
    "Subfactorial" if args.len() == 1 => {
      let n = &args[0];
      Some(mk_div(
        mk_call("Gamma", vec![mk_plus(mk_int(1), n.clone()), mk_int(-1)]),
        mk_id("E"),
      ))
    }

    // Haversine[x] → (1 - Cos[x]) / 2
    "Haversine" if args.len() == 1 => Some(mk_times(
      mk_ratio(1, 2),
      mk_plus(
        mk_int(1),
        mk_times(mk_int(-1), mk_call("Cos", vec![args[0].clone()])),
      ),
    )),

    // InverseHaversine[x] → 2 * ArcSin[Sqrt[x]]
    "InverseHaversine" if args.len() == 1 => Some(mk_times(
      mk_int(2),
      mk_call("ArcSin", vec![mk_call("Sqrt", vec![args[0].clone()])]),
    )),

    // Sinc[x] → Sin[x] / x
    "Sinc" if args.len() == 1 => Some(mk_div(
      mk_call("Sin", vec![args[0].clone()]),
      args[0].clone(),
    )),

    // LogisticSigmoid[x] → 1/(1 + E^(-x)), i.e. (1 + E^(-x))^(-1).
    "LogisticSigmoid" if args.len() == 1 => Some(mk_power(
      mk_plus(
        mk_int(1),
        mk_power(mk_id("E"), mk_times(mk_int(-1), args[0].clone())),
      ),
      mk_int(-1),
    )),

    // ChebyshevT[n, x] → Cos[n * ArcCos[x]]
    "ChebyshevT" if args.len() == 2 => Some(mk_call(
      "Cos",
      vec![mk_times(
        args[0].clone(),
        mk_call("ArcCos", vec![args[1].clone()]),
      )],
    )),

    // ChebyshevU[n, x] → Sin[(1 + n) * ArcCos[x]] / (Sqrt[1 - x] * Sqrt[1 + x])
    "ChebyshevU" if args.len() == 2 => {
      let n = &args[0];
      let x = &args[1];
      Some(mk_div(
        mk_call(
          "Sin",
          vec![mk_times(
            mk_plus(mk_int(1), n.clone()),
            mk_call("ArcCos", vec![x.clone()]),
          )],
        ),
        mk_times(
          mk_call(
            "Sqrt",
            vec![mk_plus(mk_int(1), mk_times(mk_int(-1), x.clone()))],
          ),
          mk_call("Sqrt", vec![mk_plus(mk_int(1), x.clone())]),
        ),
      ))
    }

    // Fibonacci[n] → (GoldenRatio^n - (-1/GoldenRatio)^n * Cos[n*Pi]) / Sqrt[5]
    // where GoldenRatio = (1 + Sqrt[5]) / 2
    "Fibonacci" if args.len() == 1 => {
      let n = &args[0];
      let sqrt5 = mk_call("Sqrt", vec![mk_int(5)]);
      let golden = mk_times(mk_ratio(1, 2), mk_plus(mk_int(1), sqrt5.clone()));
      let inv_golden = mk_times(
        mk_int(2),
        mk_power(mk_plus(mk_int(1), sqrt5.clone()), mk_int(-1)),
      );
      Some(mk_div(
        mk_plus(
          mk_power(golden, n.clone()),
          mk_times(
            mk_int(-1),
            mk_times(
              mk_power(inv_golden, n.clone()),
              mk_call("Cos", vec![mk_times(n.clone(), mk_id("Pi"))]),
            ),
          ),
        ),
        sqrt5,
      ))
    }

    // LucasL[n] → GoldenRatio^n + (-1/GoldenRatio)^n * Cos[n*Pi]
    "LucasL" if args.len() == 1 => {
      let n = &args[0];
      let sqrt5 = mk_call("Sqrt", vec![mk_int(5)]);
      let golden = mk_times(mk_ratio(1, 2), mk_plus(mk_int(1), sqrt5.clone()));
      let inv_golden =
        mk_times(mk_int(2), mk_power(mk_plus(mk_int(1), sqrt5), mk_int(-1)));
      Some(mk_plus(
        mk_power(golden, n.clone()),
        mk_times(
          mk_power(inv_golden, n.clone()),
          mk_call("Cos", vec![mk_times(n.clone(), mk_id("Pi"))]),
        ),
      ))
    }

    // Gamma[1/2] → Sqrt[Pi]
    "Gamma" if args.len() == 1 => {
      if let Expr::FunctionCall { name: rn, args: ra } = &args[0]
        && rn == "Rational"
        && ra.len() == 2
        && let (Expr::Integer(1), Expr::Integer(2)) = (&ra[0], &ra[1])
      {
        return Some(mk_call("Sqrt", vec![mk_id("Pi")]));
      }
      None
    }

    // Gamma[A]/Gamma[B] with A - B a positive integer expands to the rising
    // factorial Pochhammer[B, A - B] (e.g. Gamma[n+2]/Gamma[n] -> n*(1 + n)).
    "Times" => try_gamma_ratio_in_times(args),

    _ => None,
  }
}

/// If `f` is `Gamma[B]^(-1)` (in either the FunctionCall or BinaryOp Power
/// spelling), return the argument `B`.
fn reciprocal_gamma_arg(f: &Expr) -> Option<Expr> {
  let (base, exp) = match f {
    Expr::FunctionCall { name, args } if name == "Power" && args.len() == 2 => {
      (&args[0], &args[1])
    }
    Expr::BinaryOp {
      op: crate::syntax::BinaryOperator::Power,
      left,
      right,
    } => (left.as_ref(), right.as_ref()),
    _ => return None,
  };
  if !matches!(exp, Expr::Integer(-1)) {
    return None;
  }
  match base {
    Expr::FunctionCall { name, args } if name == "Gamma" && args.len() == 1 => {
      Some(args[0].clone())
    }
    _ => None,
  }
}

/// In a product of `factors`, cancel a `Gamma[A] * Gamma[B]^(-1)` pair whose
/// arguments differ by a positive integer `k = A - B`, replacing it with
/// `Pochhammer[B, k]` (which evaluates to the product B (B+1) ... (B+k-1)).
fn try_gamma_ratio_in_times(factors: &[Expr]) -> Option<Expr> {
  let mut num: Option<(usize, Expr)> = None;
  let mut den: Option<(usize, Expr)> = None;
  for (i, f) in factors.iter().enumerate() {
    if num.is_none()
      && let Expr::FunctionCall { name, args } = f
      && name == "Gamma"
      && args.len() == 1
    {
      num = Some((i, args[0].clone()));
    } else if den.is_none()
      && let Some(b) = reciprocal_gamma_arg(f)
    {
      den = Some((i, b));
    }
  }
  let (ni, a) = num?;
  let (di, b) = den?;
  // k = A - B must be an integer.
  let diff = crate::evaluator::evaluate_expr_to_expr(&mk_plus(
    a.clone(),
    mk_times(mk_int(-1), b.clone()),
  ))
  .ok()?;
  let k = match diff {
    Expr::Integer(k) => k,
    _ => return None,
  };
  // Gamma[A]/Gamma[B] = Pochhammer[B, k] for k > 0, 1/Pochhammer[A, -k] for
  // k < 0, and 1 for k == 0.
  let poch = match k.cmp(&0) {
    std::cmp::Ordering::Greater => mk_call("Pochhammer", vec![b, mk_int(k)]),
    std::cmp::Ordering::Less => {
      mk_power(mk_call("Pochhammer", vec![a, mk_int(-k)]), mk_int(-1))
    }
    std::cmp::Ordering::Equal => mk_int(1),
  };
  let mut rest: Vec<Expr> = factors
    .iter()
    .enumerate()
    .filter(|(i, _)| *i != ni && *i != di)
    .map(|(_, f)| f.clone())
    .collect();
  rest.push(poch);
  Some(if rest.len() == 1 {
    rest.remove(0)
  } else {
    mk_call("Times", rest)
  })
}

/// Expand Binomial[n, k] for specific small integer k values.
fn expand_binomial_integer_k(n: &Expr, k: i128) -> Option<Expr> {
  match k {
    0 => Some(mk_int(1)),
    1 => Some(n.clone()),
    2 => {
      // n*(n-1)/2
      Some(mk_times(
        mk_ratio(1, 2),
        mk_times(n.clone(), mk_plus(mk_int(-1), n.clone())),
      ))
    }
    3 => {
      // n*(n-1)*(n-2)/6
      Some(mk_times(
        mk_ratio(1, 6),
        mk_times(
          n.clone(),
          mk_times(
            mk_plus(mk_int(-1), n.clone()),
            mk_plus(mk_int(-2), n.clone()),
          ),
        ),
      ))
    }
    _ => None,
  }
}
