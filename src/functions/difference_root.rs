//! `DifferenceRoot[Function[{y, n}, {recurrence, initial values…}]]`: the
//! sequence defined by a linear recurrence with polynomial coefficients.
//!
//! A valid object is put into Wolfram's normal form — the formal symbols
//! `\[FormalY]`/`\[FormalN]`, the recurrence shifted to start at `y[n]` and
//! written `Σ cⱼ(n) y[n + j] + c(n) == 0` with the polynomial content
//! divided out and each coefficient factored — and `DifferenceRoot[…][k]`
//! runs the recurrence from the initial values, forwards or backwards.

use crate::InterpreterError;
use crate::evaluator::evaluate_expr_to_expr;
use crate::helpers::{call, id_expr};
use crate::syntax::{ComparisonOp, Expr};

const FORMAL_Y: &str = "\u{F818}";
const FORMAL_N: &str = "\u{F80D}";

/// A normalized recurrence `Σ coeffs[j](n) y[n + j] + inhom(n) == 0` with
/// its initial values `y[k] == v`.
struct Recurrence {
  coeffs: Vec<Expr>,
  inhom: Expr,
  inits: Vec<(i128, Expr)>,
}

fn eval(e: Expr) -> Expr {
  evaluate_expr_to_expr(&e).unwrap_or(e)
}

fn y_at(arg: Expr) -> Expr {
  Expr::FunctionCall {
    name: FORMAL_Y.to_string(),
    args: vec![arg].into(),
  }
}

fn shifted_n(j: i128) -> Expr {
  eval(call("Plus", vec![id_expr(FORMAL_N), Expr::Integer(j)]))
}

fn equal(lhs: Expr, rhs: Expr) -> Expr {
  Expr::Comparison {
    operands: vec![lhs, rhs],
    operators: vec![ComparisonOp::Equal],
  }
}

/// The two sides of an equation `lhs == rhs`.
fn equation_sides(e: &Expr) -> Option<(&Expr, &Expr)> {
  match e {
    Expr::Comparison {
      operands,
      operators,
    } if operands.len() == 2 && operators == &[ComparisonOp::Equal] => {
      Some((&operands[0], &operands[1]))
    }
    Expr::FunctionCall { name, args } if name == "Equal" && args.len() == 2 => {
      Some((&args[0], &args[1]))
    }
    _ => None,
  }
}

/// `Function[{y, n}, body]` as its two parameter names and body.
fn function_parts(f: &Expr) -> Option<(String, String, Expr)> {
  match f {
    Expr::NamedFunction { params, body, .. } if params.len() == 2 => {
      Some((params[0].clone(), params[1].clone(), (**body).clone()))
    }
    Expr::FunctionCall { name, args }
      if name == "Function" && args.len() == 2 =>
    {
      match &args[0] {
        Expr::List(ps) => match ps.as_slice() {
          [Expr::Identifier(y), Expr::Identifier(n)] => {
            Some((y.clone(), n.clone(), args[1].clone()))
          }
          _ => None,
        },
        _ => None,
      }
    }
    _ => None,
  }
}

/// Every `y[arg]` call in `e`.
fn y_calls<'a>(e: &'a Expr, out: &mut Vec<&'a Expr>) {
  if let Expr::FunctionCall { name, args } = e
    && name == FORMAL_Y
    && args.len() == 1
  {
    out.push(&args[0]);
    return;
  }
  for child in crate::syntax::expr_children(e) {
    y_calls(child, out);
  }
}

/// Whether `sym` occurs in `e`, as a symbol or as a head.
fn mentions(e: &Expr, sym: &str) -> bool {
  match e {
    Expr::Identifier(s) => s == sym,
    Expr::FunctionCall { name, .. } if name == sym => true,
    _ => crate::syntax::expr_children(e)
      .into_iter()
      .any(|c| mentions(c, sym)),
  }
}

/// The integer `k` of `y[n + k]`, if the argument has that shape.
fn shift_of(arg: &Expr) -> Option<i128> {
  match eval(call(
    "Plus",
    vec![
      arg.clone(),
      call("Times", vec![Expr::Integer(-1), id_expr(FORMAL_N)]),
    ],
  )) {
    Expr::Integer(k) => Some(k),
    _ => None,
  }
}

fn is_negative_number(e: &Expr) -> bool {
  match e {
    Expr::Integer(n) => *n < 0,
    Expr::BigInteger(n) => n.sign() == num_bigint::Sign::Minus,
    Expr::Real(x) => *x < 0.0,
    Expr::FunctionCall { name, args } if name == "Rational" => {
      args.first().is_some_and(is_negative_number)
    }
    _ => false,
  }
}

/// Whether the polynomial `c` in `n` has a negative leading coefficient
/// (numerically, or as a product led by a negative number).
fn leading_coefficient_negative(c: &Expr) -> bool {
  let expanded = eval(call("Expand", vec![c.clone()]));
  let degree =
    eval(call("Exponent", vec![expanded.clone(), id_expr(FORMAL_N)]));
  let lead = eval(call(
    "Coefficient",
    vec![expanded, id_expr(FORMAL_N), degree],
  ));
  let first_term = match &lead {
    Expr::FunctionCall { name, args } if name == "Plus" && !args.is_empty() => {
      args[0].clone()
    }
    other => other.clone(),
  };
  match &first_term {
    Expr::FunctionCall { name, args } if name == "Times" => {
      args.first().is_some_and(is_negative_number)
    }
    other => is_negative_number(other),
  }
}

/// Why a `Function[{y, n}, eqs]` defines no difference root.
#[derive(Debug, PartialEq)]
enum Invalid {
  /// Not a recurrence with polynomial coefficients (`DifferenceRoot::undef`).
  Undefined,
  /// The recurrence is not linear in `y` (`DifferenceRoot::ieqn`).
  Nonlinear,
}

/// Parse and normalize `Function[{y, n}, eqs]` into a linear recurrence
/// with polynomial coefficients.
fn normalize(f: &Expr) -> Result<Recurrence, Invalid> {
  use Invalid::{Nonlinear, Undefined};
  let (y, n, body) = function_parts(f).ok_or(Undefined)?;
  let body = crate::syntax::substitute_variables(
    &body,
    &[(&y, &id_expr(FORMAL_Y)), (&n, &id_expr(FORMAL_N))],
  );
  let evaluated = eval(body);
  let eqs = match &evaluated {
    Expr::List(items) => items.to_vec(),
    single => vec![single.clone()],
  };
  let mut recurrence: Option<Expr> = None;
  let mut inits: Vec<(i128, Expr)> = Vec::new();
  for eq in &eqs {
    let (lhs, rhs) = equation_sides(eq).ok_or(Undefined)?;
    if let Expr::FunctionCall { name, args } = lhs
      && name == FORMAL_Y
      && let [Expr::Integer(k)] = args.as_slice()
      && !mentions(rhs, FORMAL_Y)
      && !mentions(rhs, FORMAL_N)
    {
      inits.push((*k, rhs.clone()));
      continue;
    }
    if recurrence.is_some() || !mentions(eq, FORMAL_N) {
      return Err(Undefined);
    }
    recurrence = Some(eval(call(
      "Plus",
      vec![
        lhs.clone(),
        call("Times", vec![Expr::Integer(-1), rhs.clone()]),
      ],
    )));
  }
  let expr = recurrence.ok_or(Undefined)?;
  let mut args = Vec::new();
  y_calls(&expr, &mut args);
  let shifts: Vec<i128> = args
    .iter()
    .map(|a| shift_of(a))
    .collect::<Option<_>>()
    .ok_or(Undefined)?;
  let low = *shifts.iter().min().ok_or(Undefined)?;
  let high = *shifts.iter().max().ok_or(Undefined)?;
  let order = usize::try_from(high - low).map_err(|_| Undefined)?;
  if order == 0 {
    return Err(Undefined);
  }
  // Start the recurrence at y[n], clear its denominators, and read off
  // the coefficient of every shift.
  let shifted = if low == 0 {
    expr
  } else {
    eval(call(
      "ReplaceAll",
      vec![
        expr,
        Expr::Rule {
          pattern: Box::new(id_expr(FORMAL_N)),
          replacement: Box::new(shifted_n(-low)),
        },
      ],
    ))
  };
  let numerator = eval(call(
    "Expand",
    vec![eval(call(
      "Numerator",
      vec![eval(call("Together", vec![shifted]))],
    ))],
  ));
  let order_i = order as i128;
  let mut coeffs: Vec<Expr> = (0..=order_i)
    .map(|j| {
      eval(call(
        "Coefficient",
        vec![numerator.clone(), y_at(shifted_n(j))],
      ))
    })
    .collect();
  let mut inhom = eval(call(
    "ReplaceAll",
    vec![
      numerator.clone(),
      Expr::Rule {
        pattern: Box::new(y_at(Expr::Pattern {
          name: String::new(),
          head: None,
          blank_type: 1,
        })),
        replacement: Box::new(Expr::Integer(0)),
      },
    ],
  ));
  // Linear, with coefficients polynomial in n and free of y.
  let mut rebuilt = vec![inhom.clone()];
  for (j, c) in coeffs.iter().enumerate() {
    if mentions(c, FORMAL_Y) {
      return Err(Nonlinear);
    }
    if !matches!(
      eval(call("PolynomialQ", vec![c.clone(), id_expr(FORMAL_N)])),
      Expr::Identifier(ref t) if t == "True"
    ) {
      return Err(Undefined);
    }
    rebuilt.push(call("Times", vec![c.clone(), y_at(shifted_n(j as i128))]));
  }
  let residual = eval(call(
    "Expand",
    vec![call(
      "Plus",
      vec![
        call("Plus", rebuilt),
        call("Times", vec![Expr::Integer(-1), numerator.clone()]),
      ],
    )],
  ));
  if !matches!(residual, Expr::Integer(0)) || mentions(&inhom, FORMAL_Y) {
    return Err(Nonlinear);
  }
  if matches!(coeffs[0], Expr::Integer(0))
    || matches!(coeffs[order], Expr::Integer(0))
  {
    return Err(Undefined);
  }
  // Divide out the common polynomial content, make the leading
  // coefficient of the top shift positive, and factor.
  let mut parts: Vec<Expr> = coeffs.clone();
  if !matches!(inhom, Expr::Integer(0)) {
    parts.push(inhom.clone());
  }
  let gcd = eval(call("PolynomialGCD", parts));
  let sign = if leading_coefficient_negative(&coeffs[order]) {
    -1
  } else {
    1
  };
  let finish = |c: &Expr| {
    let reduced = eval(call(
      "Cancel",
      vec![call(
        "Times",
        vec![
          Expr::Integer(sign),
          c.clone(),
          call("Power", vec![gcd.clone(), Expr::Integer(-1)]),
        ],
      )],
    ));
    eval(call("Factor", vec![reduced]))
  };
  for c in &mut coeffs {
    *c = finish(c);
  }
  inhom = finish(&inhom);
  Ok(Recurrence {
    coeffs,
    inhom,
    inits,
  })
}

impl Recurrence {
  fn to_function(&self) -> Expr {
    let mut terms = vec![self.inhom.clone()];
    for (j, c) in self.coeffs.iter().enumerate() {
      terms.push(call("Times", vec![c.clone(), y_at(shifted_n(j as i128))]));
    }
    let mut eqs = vec![equal(eval(call("Plus", terms)), Expr::Integer(0))];
    for (k, v) in &self.inits {
      eqs.push(equal(y_at(Expr::Integer(*k)), v.clone()));
    }
    Expr::NamedFunction {
      params: vec![FORMAL_Y.to_string(), FORMAL_N.to_string()],
      body: Box::new(Expr::List(eqs.into())),
      bracketed: true,
    }
  }

  fn order(&self) -> i128 {
    self.coeffs.len() as i128 - 1
  }

  /// The first index of `order` consecutive initial values.
  fn start(&self) -> Option<i128> {
    let r = self.order();
    let known = |i: i128| self.inits.iter().any(|(k, _)| *k == i);
    let mut idx: Vec<i128> = self.inits.iter().map(|(k, _)| *k).collect();
    idx.sort_unstable();
    idx.into_iter().find(|&s| (s..s + r).all(known))
  }

  fn at(c: &Expr, m: i128) -> Expr {
    eval(crate::syntax::substitute_variable(
      c,
      FORMAL_N,
      &Expr::Integer(m),
    ))
  }

  /// `y[k]`, or `None` when a vanishing leading coefficient leaves it
  /// undetermined.
  fn value(&self, k: i128, start: i128) -> Option<Expr> {
    let r = self.order();
    let mut known: std::collections::HashMap<i128, Expr> =
      self.inits.iter().cloned().collect();
    let combine = |terms: Vec<Expr>, divisor: Expr| -> Option<Expr> {
      if matches!(divisor, Expr::Integer(0)) {
        return None;
      }
      Some(eval(call(
        "Together",
        vec![call(
          "Times",
          vec![
            Expr::Integer(-1),
            call("Plus", terms),
            call("Power", vec![divisor, Expr::Integer(-1)]),
          ],
        )],
      )))
    };
    if k >= start {
      for i in start + r..=k {
        if known.contains_key(&i) {
          continue;
        }
        let m = i - r;
        let mut terms = vec![Self::at(&self.inhom, m)];
        for j in 0..r {
          terms.push(call(
            "Times",
            vec![
              Self::at(&self.coeffs[j as usize], m),
              known.get(&(m + j))?.clone(),
            ],
          ));
        }
        let v = combine(terms, Self::at(&self.coeffs[r as usize], m))?;
        known.insert(i, v);
      }
    } else {
      for i in (k..start).rev() {
        if known.contains_key(&i) {
          continue;
        }
        let mut terms = vec![Self::at(&self.inhom, i)];
        for j in 1..=r {
          terms.push(call(
            "Times",
            vec![
              Self::at(&self.coeffs[j as usize], i),
              known.get(&(i + j))?.clone(),
            ],
          ));
        }
        let v = combine(terms, Self::at(&self.coeffs[0], i))?;
        known.insert(i, v);
      }
    }
    let v = known.remove(&k)?;
    if crate::functions::predicate_ast::is_numeric_q(&v) {
      Some(v)
    } else {
      Some(eval(call("Simplify", vec![v])))
    }
  }
}

fn is_normalized(f: &Expr) -> bool {
  matches!(f, Expr::NamedFunction { params, .. }
    if params.len() == 2 && params[0] == FORMAL_Y && params[1] == FORMAL_N)
}

/// `DifferenceRoot[f]`.
pub fn difference_root_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let original = crate::helpers::unevaluated("DifferenceRoot", args);
  let [f] = args else {
    crate::emit_message(&format!(
      "DifferenceRoot::argrx: DifferenceRoot called with {} arguments; 1 \
       argument is expected.",
      args.len()
    ));
    return Ok(original);
  };
  if function_parts(f).is_none() {
    crate::emit_message(&format!(
      "DifferenceRoot::pfn: {} is not a pure function.",
      crate::syntax::expr_to_message_form(f)
    ));
    return Ok(original);
  }
  if is_normalized(f) {
    return Ok(original);
  }
  match normalize(f) {
    Ok(rec) if rec.start().is_some() => {
      Ok(call("DifferenceRoot", vec![rec.to_function()]))
    }
    Err(Invalid::Nonlinear) => {
      crate::emit_message(&format!(
        "DifferenceRoot::ieqn: The supplied equation in {} is not a linear \
         difference equation with polynomial coefficients.",
        crate::syntax::expr_to_message_form(f)
      ));
      Ok(original)
    }
    _ => {
      crate::emit_message(&format!(
        "DifferenceRoot::undef: {} does not define a suitable difference \
         root object.",
        crate::syntax::expr_to_message_form(&original)
      ));
      Ok(original)
    }
  }
}

/// `DifferenceRoot[f][args]`; `None` leaves the call unevaluated.
pub fn apply_difference_root(f: &Expr, args: &[Expr]) -> Option<Expr> {
  let [k] = args else {
    return None;
  };
  if let Expr::List(items) = k {
    let mapped: Vec<Expr> = items
      .iter()
      .map(|x| {
        apply_difference_root(f, std::slice::from_ref(x)).unwrap_or_else(|| {
          Expr::CurriedCall {
            func: Box::new(call("DifferenceRoot", vec![f.clone()])),
            args: vec![x.clone()],
          }
        })
      })
      .collect();
    return Some(Expr::List(mapped.into()));
  }
  // An invalid object was reported when DifferenceRoot[f] evaluated.
  let rec = normalize(f).ok()?;
  let start = rec.start()?;
  let index = if let Expr::Integer(i) = k {
    *i
  } else {
    let message = if matches!(k, Expr::Real(_) | Expr::BigFloat(..)) {
      Some(format!(
        "DifferenceRoot::exact: Argument {} in DifferenceRoot is not an \
         exact number.",
        crate::syntax::expr_to_message_form(k)
      ))
    } else if crate::functions::predicate_ast::is_numeric_q(k) {
      Some(
        "DifferenceRoot::int: Integer expected at position 1 in \
         DifferenceRoot."
          .to_string(),
      )
    } else {
      None
    };
    if let Some(m) = message {
      crate::emit_message(&m);
    }
    return None;
  };
  rec.value(index, start)
}
