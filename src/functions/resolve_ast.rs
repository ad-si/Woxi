//! Resolve[Exists[...]/ForAll[...], Reals] — quantifier elimination for
//! univariate polynomial conditions, matching wolframscript:
//! parameter-free formulas decide via Reduce (with complex solutions
//! discarded over the reals), and the parametrized doc-example families
//! Exists[x, x^even == c] -> c >= 0 and ForAll[x, x^even + c > 0] ->
//! c > 0 return their conditions.

use crate::InterpreterError;
use crate::syntax::{
  BinaryOperator, ComparisonOp, Expr, UnaryOperator, unevaluated,
};

pub fn resolve_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = |args: &[Expr]| unevaluated("Resolve", args);
  if args.is_empty() || args.len() > 2 {
    return Ok(unevaluated(args));
  }
  // Optional domain: only Reals (or omitted, which defaults to complexes
  // in Wolfram, but the rules below are domain-agnostic for the
  // supported families)
  if args.len() == 2 && !matches!(&args[1], Expr::Identifier(d) if d == "Reals")
  {
    return Ok(unevaluated(args));
  }
  let over_reals = args.len() == 2;

  let (head, vars, cond) = match &args[0] {
    Expr::FunctionCall { name, args: qargs }
      if (name == "Exists" || name == "ForAll") && qargs.len() == 2 =>
    {
      let vars = match &qargs[0] {
        Expr::Identifier(v) => vec![v.clone()],
        Expr::List(items) => {
          let mut vs = Vec::with_capacity(items.len());
          for it in items.iter() {
            match it {
              Expr::Identifier(v) => vs.push(v.clone()),
              _ => return Ok(unevaluated(args)),
            }
          }
          vs
        }
        _ => return Ok(unevaluated(args)),
      };
      if vars.is_empty() {
        return Ok(unevaluated(args));
      }
      (name.as_str(), vars, qargs[1].clone())
    }
    _ => return Ok(unevaluated(args)),
  };

  // Several bound variables: use the separable-interval decision procedure.
  if vars.len() > 1 {
    return Ok(
      resolve_multivar(head, &vars, &cond, over_reals)
        .unwrap_or_else(|| unevaluated(args)),
    );
  }
  let var = vars[0].clone();

  let truth = |b: bool| {
    Ok(Expr::Identifier(
      if b { "True" } else { "False" }.to_string(),
    ))
  };

  // Parametrized templates
  if let Some(result) = parametric_template(head, &var, &cond) {
    return Ok(result);
  }

  // Parameter-free path: decide via Reduce
  if has_free_symbol(&cond, &var) {
    return Ok(unevaluated(args));
  }
  match head {
    "Exists" => {
      let reduced =
        crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
          name: "Reduce".to_string(),
          args: vec![cond.clone(), Expr::Identifier(var.clone())].into(),
        })?;
      match &reduced {
        Expr::Identifier(s) if s == "False" => truth(false),
        Expr::Identifier(s) if s == "True" => truth(true),
        Expr::FunctionCall { name, .. } if name == "Reduce" => {
          Ok(unevaluated(args))
        }
        solution => {
          if over_reals {
            // Discard complex solution branches
            truth(any_real_branch(solution))
          } else {
            truth(true)
          }
        }
      }
    }
    "ForAll" => {
      // ForAll[x, cond] == !Exists[x, !cond] for invertible comparisons
      let negated = match negate_comparison(&cond) {
        Some(n) => n,
        None => return Ok(unevaluated(args)),
      };
      let reduced =
        crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
          name: "Reduce".to_string(),
          args: vec![negated, Expr::Identifier(var.clone())].into(),
        })?;
      match &reduced {
        Expr::Identifier(s) if s == "False" => truth(true),
        Expr::Identifier(s) if s == "True" => truth(false),
        Expr::FunctionCall { name, .. } if name == "Reduce" => {
          Ok(unevaluated(args))
        }
        solution => {
          if over_reals {
            truth(!any_real_branch(solution))
          } else {
            truth(false)
          }
        }
      }
    }
    _ => Ok(unevaluated(args)),
  }
}

/// The documented parametrized families:
/// Exists[x, x^even == c] -> c >= 0
/// ForAll[x, x^even + c > 0] -> c > 0 (and >= 0 for GreaterEqual)
fn parametric_template(head: &str, var: &str, cond: &Expr) -> Option<Expr> {
  let (operands, op) = match cond {
    Expr::Comparison {
      operands,
      operators,
    } if operands.len() == 2 && operators.len() == 1 => {
      (operands, operators[0])
    }
    _ => return None,
  };
  let is_even_power_of_var = |e: &Expr| -> bool {
    let (base, exp) = match e {
      Expr::FunctionCall { name, args }
        if name == "Power" && args.len() == 2 =>
      {
        (&args[0], &args[1])
      }
      Expr::BinaryOp {
        op: BinaryOperator::Power,
        left,
        right,
      } => (&**left as &Expr, &**right as &Expr),
      _ => return false,
    };
    matches!(base, Expr::Identifier(v) if v == var)
      && matches!(exp, Expr::Integer(k) if *k >= 2 && k % 2 == 0)
  };
  let free_symbol = |e: &Expr| -> Option<Expr> {
    match e {
      Expr::Identifier(s) if s != var => Some(e.clone()),
      _ => None,
    }
  };

  match head {
    // Exists[x, x^even == c]
    "Exists" if op == ComparisonOp::Equal => {
      let c = if is_even_power_of_var(&operands[0]) {
        free_symbol(&operands[1])?
      } else if is_even_power_of_var(&operands[1]) {
        free_symbol(&operands[0])?
      } else {
        return None;
      };
      Some(Expr::Comparison {
        operands: vec![c, Expr::Integer(0)],
        operators: vec![ComparisonOp::GreaterEqual],
      })
    }
    // ForAll[x, x^even + c > 0] (Plus arrives canonically as c + x^even)
    "ForAll"
      if (op == ComparisonOp::Greater || op == ComparisonOp::GreaterEqual)
        && matches!(&operands[1], Expr::Integer(0)) =>
    {
      let terms: Vec<&Expr> = match &operands[0] {
        Expr::FunctionCall { name, args }
          if name == "Plus" && args.len() == 2 =>
        {
          args.iter().collect()
        }
        _ => return None,
      };
      let c = if is_even_power_of_var(terms[1]) {
        free_symbol(terms[0])?
      } else if is_even_power_of_var(terms[0]) {
        free_symbol(terms[1])?
      } else {
        return None;
      };
      Some(Expr::Comparison {
        operands: vec![c, Expr::Integer(0)],
        operators: vec![op],
      })
    }
    _ => None,
  }
}

/// Any identifier other than `var` (and protected constants)?
fn has_free_symbol(expr: &Expr, var: &str) -> bool {
  match expr {
    Expr::Identifier(s) => {
      s != var && !matches!(s.as_str(), "True" | "False" | "Pi" | "E" | "I")
    }
    Expr::Constant(_) => false,
    Expr::FunctionCall { args, .. } | Expr::List(args) => {
      args.iter().any(|a| has_free_symbol(a, var))
    }
    Expr::Comparison { operands, .. } => {
      operands.iter().any(|a| has_free_symbol(a, var))
    }
    Expr::BinaryOp { left, right, .. } => {
      has_free_symbol(left, var) || has_free_symbol(right, var)
    }
    Expr::UnaryOp { operand, .. } => has_free_symbol(operand, var),
    _ => false,
  }
}

/// Negate a simple comparison (chains and equations stay unsupported).
fn negate_comparison(cond: &Expr) -> Option<Expr> {
  match cond {
    Expr::Comparison {
      operands,
      operators,
    } if operands.len() == 2 && operators.len() == 1 => {
      let flipped = match operators[0] {
        ComparisonOp::Less => ComparisonOp::GreaterEqual,
        ComparisonOp::LessEqual => ComparisonOp::Greater,
        ComparisonOp::Greater => ComparisonOp::LessEqual,
        ComparisonOp::GreaterEqual => ComparisonOp::Less,
        _ => return None,
      };
      Some(Expr::Comparison {
        operands: operands.clone(),
        operators: vec![flipped],
      })
    }
    _ => None,
  }
}

/// Does any Or-branch of a Reduce solution avoid the imaginary unit?
fn any_real_branch(solution: &Expr) -> bool {
  let branches: Vec<&Expr> = match solution {
    Expr::FunctionCall { name, args } if name == "Or" => args.iter().collect(),
    other => vec![other],
  };
  branches.iter().any(|b| !contains_imaginary(b))
}

fn contains_imaginary(expr: &Expr) -> bool {
  match expr {
    Expr::Identifier(s) | Expr::Constant(s) => s == "I",
    Expr::FunctionCall { args, .. } | Expr::List(args) => {
      args.iter().any(contains_imaginary)
    }
    Expr::Comparison { operands, .. } => {
      operands.iter().any(contains_imaginary)
    }
    Expr::BinaryOp { left, right, .. } => {
      contains_imaginary(left) || contains_imaginary(right)
    }
    Expr::UnaryOp { operand, .. } => contains_imaginary(operand),
    _ => false,
  }
}

// ─── Multivariate, parameter-free quantifier elimination ───────────────
//
// Decides `Resolve[Exists[{x, y, ...}, cond]]` and the `ForAll` dual for a
// single polynomial comparison whose polynomial is *additively separable*:
// every bound variable occurs in exactly one monomial term, e.g.
// `x^2 + y^2 - 1`. The achievable value set of such a polynomial is the
// Minkowski sum of the per-term ranges, which is a single real interval, so
// existence/universality reduce to interval membership/containment.
//
// Semantics matched against wolframscript:
//  * inequalities (<, <=, >, >=) are always decided over the reals — the
//    order relation forces real values regardless of the stated domain;
//  * equations (==, !=) use the stated domain: the complexes by default,
//    the reals when `Reals` is given. Over the algebraically closed
//    complexes a non-constant polynomial always has zeros (and a non-empty
//    complement), so those collapse without needing separability.

fn gcd(a: i128, b: i128) -> i128 {
  let (mut a, mut b) = (a.abs(), b.abs());
  while b != 0 {
    let t = b;
    b = a % b;
    a = t;
  }
  a
}

/// Extract a rational literal (`Integer` or `Rational[n, d]`) from a leaf.
fn expr_to_rational(e: &Expr) -> Option<(i128, i128)> {
  match e {
    Expr::Integer(n) => Some((*n, 1)),
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      match (&args[0], &args[1]) {
        (Expr::Integer(n), Expr::Integer(d)) if *d != 0 => Some((*n, *d)),
        _ => None,
      }
    }
    _ => None,
  }
}

/// Normalised rational (sign kept in the numerator).
#[derive(Clone, Copy)]
struct Q {
  n: i128,
  d: i128,
}

impl Q {
  fn int(n: i128) -> Q {
    Q { n, d: 1 }
  }
  fn new(n: i128, d: i128) -> Q {
    let (mut n, mut d) = if d < 0 { (-n, -d) } else { (n, d) };
    let g = gcd(n, d);
    if g != 0 {
      n /= g;
      d /= g;
    }
    Q { n, d }
  }
  fn add(self, o: Q) -> Q {
    Q::new(self.n * o.d + o.n * self.d, self.d * o.d)
  }
  fn mul(self, o: Q) -> Q {
    Q::new(self.n * o.n, self.d * o.d)
  }
  fn neg(self) -> Q {
    Q {
      n: -self.n,
      d: self.d,
    }
  }
  fn is_zero(self) -> bool {
    self.n == 0
  }
  fn sign(self) -> i32 {
    self.n.signum() as i32
  }
  fn cmp(self, o: Q) -> std::cmp::Ordering {
    (self.n * o.d).cmp(&(o.n * self.d))
  }
}

/// A real interval. `None` on a side means unbounded (±∞); the `bool` flags
/// whether a finite endpoint is included (closed).
struct Interval {
  lo: Option<(Q, bool)>,
  hi: Option<(Q, bool)>,
}

fn resolve_multivar(
  head: &str,
  vars: &[String],
  cond: &Expr,
  over_reals: bool,
) -> Option<Expr> {
  let truth =
    |b: bool| Expr::Identifier(if b { "True" } else { "False" }.to_string());

  // Only single (non-chained) comparisons are supported.
  let (operands, op) = match cond {
    Expr::Comparison {
      operands,
      operators,
    } if operands.len() == 2 && operators.len() == 1 => {
      (operands, operators[0])
    }
    _ => return None,
  };

  // Move everything to one side: `lhs - rhs` compared against 0.
  let diff = Expr::FunctionCall {
    name: "Subtract".to_string(),
    args: vec![operands[0].clone(), operands[1].clone()].into(),
  };
  let expanded = crate::evaluator::evaluate_expr_to_expr(&Expr::FunctionCall {
    name: "Expand".to_string(),
    args: vec![diff].into(),
  })
  .ok()?;

  // Free (non-bound) symbols ⇒ a parametric multivariate form: leave it.
  if has_free_symbol_multi(&expanded, vars) {
    return None;
  }

  let is_equation = matches!(op, ComparisonOp::Equal | ComparisonOp::NotEqual);
  // Inequalities force real values; equations honour the stated domain.
  let use_reals = over_reals || !is_equation;

  // Complex equation with a non-constant polynomial: a non-constant
  // polynomial has zeros and a non-empty complement over the complexes, so
  // `Exists` is True and `ForAll` is False for both `==` and `!=`.
  if is_equation && !use_reals && contains_any_bound_var(&expanded, vars) {
    return Some(truth(head == "Exists"));
  }

  let range = separable_range(&expanded, vars)?;
  Some(truth(decide(head, op, &range)))
}

/// Range of an additively separable polynomial over the reals, as a single
/// interval. Returns `None` if the polynomial isn't separable (a variable
/// appears in more than one term, two variables share a term, a non-rational
/// coefficient, etc.) so the caller can leave the expression unevaluated.
fn separable_range(expr: &Expr, vars: &[String]) -> Option<Interval> {
  let mut terms: Vec<&Expr> = Vec::new();
  collect_plus(expr, &mut terms);

  let mut lo: Option<(Q, bool)> = Some((Q::int(0), true));
  let mut hi: Option<(Q, bool)> = Some((Q::int(0), true));
  let mut used: Vec<String> = Vec::new();

  for t in terms {
    let (coeff, vp) = monomial(t, vars)?;
    if coeff.is_zero() {
      continue;
    }
    match vp {
      None => {
        lo = add_bound(lo, Some((coeff, true)));
        hi = add_bound(hi, Some((coeff, true)));
      }
      Some((v, exp)) => {
        if used.contains(&v) {
          return None; // same variable in two terms ⇒ not separable
        }
        used.push(v);
        let (tlo, thi) = monomial_range(coeff, exp);
        lo = add_bound(lo, tlo);
        hi = add_bound(hi, thi);
      }
    }
  }
  Some(Interval { lo, hi })
}

/// Range of a single term `coeff * x^exp` (exp ≥ 1) over the reals.
fn monomial_range(
  coeff: Q,
  exp: i128,
) -> (Option<(Q, bool)>, Option<(Q, bool)>) {
  if exp % 2 == 0 {
    // x^even ranges over [0, ∞); the coefficient's sign orients it.
    if coeff.sign() > 0 {
      (Some((Q::int(0), true)), None)
    } else {
      (None, Some((Q::int(0), true)))
    }
  } else {
    // x^odd ranges over all reals.
    (None, None)
  }
}

/// Minkowski-sum two interval endpoints (both lower, or both upper). Any
/// unbounded side stays unbounded; a finite sum is closed iff both are.
fn add_bound(a: Option<(Q, bool)>, b: Option<(Q, bool)>) -> Option<(Q, bool)> {
  match (a, b) {
    (Some((av, ac)), Some((bv, bc))) => Some((av.add(bv), ac && bc)),
    _ => None,
  }
}

/// Parse a term into `(coefficient, Some((var, exponent)) | None)`.
/// `None` for the variable part marks a constant term. Returns `None`
/// overall if the term isn't a clean rational-coefficient monomial in the
/// bound variables.
fn monomial(
  term: &Expr,
  vars: &[String],
) -> Option<(Q, Option<(String, i128)>)> {
  let mut coeff = Q::int(1);
  let mut vp: Option<(String, i128)> = None;
  walk_monomial(term, vars, &mut coeff, &mut vp)?;
  Some((coeff, vp))
}

fn walk_monomial(
  e: &Expr,
  vars: &[String],
  coeff: &mut Q,
  vp: &mut Option<(String, i128)>,
) -> Option<()> {
  if let Some((n, d)) = expr_to_rational(e) {
    *coeff = coeff.mul(Q::new(n, d));
    return Some(());
  }
  match e {
    Expr::Identifier(s) => {
      if vars.iter().any(|v| v == s) {
        add_var_power(vp, s, 1)
      } else {
        None // free symbol or constant ⇒ not a bound monomial
      }
    }
    Expr::FunctionCall { name, args } if name == "Times" => {
      for a in args.iter() {
        walk_monomial(a, vars, coeff, vp)?;
      }
      Some(())
    }
    Expr::FunctionCall { name, args } if name == "Power" && args.len() == 2 => {
      power_term(&args[0], &args[1], vars, vp)
    }
    Expr::BinaryOp {
      op: BinaryOperator::Times,
      left,
      right,
    } => {
      walk_monomial(left, vars, coeff, vp)?;
      walk_monomial(right, vars, coeff, vp)
    }
    Expr::BinaryOp {
      op: BinaryOperator::Power,
      left,
      right,
    } => power_term(left, right, vars, vp),
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => {
      *coeff = coeff.neg();
      walk_monomial(operand, vars, coeff, vp)
    }
    _ => None,
  }
}

fn power_term(
  base: &Expr,
  exp: &Expr,
  vars: &[String],
  vp: &mut Option<(String, i128)>,
) -> Option<()> {
  let n = match exp {
    Expr::Integer(k) if *k >= 1 => *k,
    _ => return None,
  };
  match base {
    Expr::Identifier(v) if vars.iter().any(|x| x == v) => {
      add_var_power(vp, v, n)
    }
    _ => None,
  }
}

fn add_var_power(
  vp: &mut Option<(String, i128)>,
  v: &str,
  n: i128,
) -> Option<()> {
  match vp {
    None => {
      *vp = Some((v.to_string(), n));
      Some(())
    }
    Some((existing, e)) => {
      if existing == v {
        *e += n;
        Some(())
      } else {
        None // two distinct variables in one term ⇒ not separable
      }
    }
  }
}

fn collect_plus<'a>(e: &'a Expr, out: &mut Vec<&'a Expr>) {
  match e {
    Expr::FunctionCall { name, args } if name == "Plus" => {
      for a in args.iter() {
        collect_plus(a, out);
      }
    }
    Expr::BinaryOp {
      op: BinaryOperator::Plus,
      left,
      right,
    } => {
      collect_plus(left, out);
      collect_plus(right, out);
    }
    _ => out.push(e),
  }
}

/// Decide an `Exists`/`ForAll` over an interval `iv` for `expr OP 0`.
fn decide(head: &str, op: ComparisonOp, iv: &Interval) -> bool {
  let contains_zero = point_in(iv, Q::int(0));
  let contains_lt0 = match &iv.lo {
    None => true,
    Some((l, _)) => l.sign() < 0,
  };
  let contains_gt0 = match &iv.hi {
    None => true,
    Some((h, _)) => h.sign() > 0,
  };
  let is_point_zero = matches!(&iv.lo, Some((l, true)) if l.is_zero())
    && matches!(&iv.hi, Some((h, true)) if h.is_zero());

  match head {
    "Exists" => match op {
      ComparisonOp::Less => contains_lt0,
      ComparisonOp::LessEqual => contains_lt0 || contains_zero,
      ComparisonOp::Greater => contains_gt0,
      ComparisonOp::GreaterEqual => contains_gt0 || contains_zero,
      ComparisonOp::Equal => contains_zero,
      ComparisonOp::NotEqual => !is_point_zero,
      _ => false,
    },
    "ForAll" => match op {
      // S ⊆ (-∞, 0)
      ComparisonOp::Less => match &iv.hi {
        Some((h, c)) => h.sign() < 0 || (h.is_zero() && !*c),
        None => false,
      },
      // S ⊆ (-∞, 0]
      ComparisonOp::LessEqual => match &iv.hi {
        Some((h, _)) => h.sign() <= 0,
        None => false,
      },
      // S ⊆ (0, ∞)
      ComparisonOp::Greater => match &iv.lo {
        Some((l, c)) => l.sign() > 0 || (l.is_zero() && !*c),
        None => false,
      },
      // S ⊆ [0, ∞)
      ComparisonOp::GreaterEqual => match &iv.lo {
        Some((l, _)) => l.sign() >= 0,
        None => false,
      },
      ComparisonOp::Equal => is_point_zero,
      ComparisonOp::NotEqual => !contains_zero,
      _ => false,
    },
    _ => false,
  }
}

fn point_in(iv: &Interval, x: Q) -> bool {
  use std::cmp::Ordering;
  let lo_ok = match &iv.lo {
    None => true,
    Some((l, c)) => match x.cmp(*l) {
      Ordering::Greater => true,
      Ordering::Equal => *c,
      Ordering::Less => false,
    },
  };
  let hi_ok = match &iv.hi {
    None => true,
    Some((h, c)) => match x.cmp(*h) {
      Ordering::Less => true,
      Ordering::Equal => *c,
      Ordering::Greater => false,
    },
  };
  lo_ok && hi_ok
}

/// Any identifier that is neither a bound variable nor a protected constant.
fn has_free_symbol_multi(expr: &Expr, vars: &[String]) -> bool {
  match expr {
    Expr::Identifier(s) => {
      !vars.iter().any(|v| v == s)
        && !matches!(s.as_str(), "True" | "False" | "Pi" | "E" | "I")
    }
    Expr::Constant(_) => false,
    Expr::FunctionCall { args, .. } | Expr::List(args) => {
      args.iter().any(|a| has_free_symbol_multi(a, vars))
    }
    Expr::Comparison { operands, .. } => {
      operands.iter().any(|a| has_free_symbol_multi(a, vars))
    }
    Expr::BinaryOp { left, right, .. } => {
      has_free_symbol_multi(left, vars) || has_free_symbol_multi(right, vars)
    }
    Expr::UnaryOp { operand, .. } => has_free_symbol_multi(operand, vars),
    _ => false,
  }
}

fn contains_any_bound_var(expr: &Expr, vars: &[String]) -> bool {
  match expr {
    Expr::Identifier(s) => vars.iter().any(|v| v == s),
    Expr::FunctionCall { args, .. } | Expr::List(args) => {
      args.iter().any(|a| contains_any_bound_var(a, vars))
    }
    Expr::Comparison { operands, .. } => {
      operands.iter().any(|a| contains_any_bound_var(a, vars))
    }
    Expr::BinaryOp { left, right, .. } => {
      contains_any_bound_var(left, vars) || contains_any_bound_var(right, vars)
    }
    Expr::UnaryOp { operand, .. } => contains_any_bound_var(operand, vars),
    _ => false,
  }
}
