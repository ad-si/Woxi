#[allow(unused_imports)]
use super::*;

/// Report a `With`/`Module`/`Block` whose local variable specification is not
/// a List (`With[y, 3]`), and hand back the unevaluated call like Wolfram.
fn non_list_local_spec(head: &str, args: &[Expr]) -> Expr {
  crate::emit_message(&format!(
    "{head}::lvlist: Local variable specification {} is not a List.",
    crate::syntax::expr_to_string(&args[0])
  ));
  unevaluated(head, args)
}

/// One local of a scoping construct: the symbol it binds, its initializer
/// (`None` for a bare `x`, which only `Module` and `Block` allow) and
/// whether that initializer was delayed (`x := v`, which stays unevaluated).
struct LocalVar {
  name: String,
  init: Option<Expr>,
  delayed: bool,
}

/// A malformed entry in a local variable specification, in the vocabulary of
/// Wolfram's messages.
enum LocalSpecError {
  /// `Module[{5}, …]` — neither a symbol nor an assignment to one.
  NotSymbol(Expr),
  /// `With[{x}, …]` — `With` gives every variable a value.
  NeedsValue(Expr),
  /// `Module[{x[1] = 3}, …]` — an assignment, but not to a symbol.
  /// Boxed so this variant does not blow up the size of every `Result`.
  BadAssignment { item: Box<Expr>, target: Box<Expr> },
  /// `Module[{x, x}, …]` — the same name twice.
  Duplicate(String),
}

impl LocalSpecError {
  /// Emit the message under `head` (`Module`, `Block` or `With`), quoting the
  /// whole specification the way Wolfram does.
  fn emit(&self, head: &str, spec: &Expr) {
    let spec = expr_to_string(spec);
    let body = match self {
      Self::NotSymbol(item) => format!(
        "lvsym: Local variable specification {spec} contains {}, which is \
         not a symbol or an assignment to a symbol.",
        expr_to_string(item)
      ),
      Self::NeedsValue(item) => format!(
        "lvws: Variable {} in local variable specification {spec} requires \
         a value.",
        expr_to_string(item)
      ),
      Self::BadAssignment { item, target } => format!(
        "lvset: Local variable specification {spec} contains {}, which is \
         an assignment to {}; only assignments to symbols are allowed.",
        expr_to_string(item),
        expr_to_string(target)
      ),
      Self::Duplicate(name) => format!(
        "dup: Duplicate local variable {name} found in local variable \
         specification {spec}."
      ),
    };
    crate::emit_message(&format!("{head}::{body}"));
  }
}

/// Split `x = v` / `x := v` into target and value, whichever way the parser
/// spelled the assignment.
fn as_local_assignment(item: &Expr) -> Option<(&Expr, &Expr, bool)> {
  match item {
    Expr::FunctionCall { name, args }
      if (name == "Set" || name == "SetDelayed") && args.len() == 2 =>
    {
      Some((&args[0], &args[1], name == "SetDelayed"))
    }
    Expr::Rule {
      pattern,
      replacement,
    } => Some((pattern, replacement, false)),
    Expr::RuleDelayed {
      pattern,
      replacement,
    } => Some((pattern, replacement, true)),
    _ => None,
  }
}

/// Validate a local variable specification the way Wolfram does — left to
/// right, reporting only the first offending entry. Every entry must be a
/// symbol or an assignment to a symbol, no name may repeat, and when
/// `needs_value` is set (`With`) a bare symbol is rejected too.
fn parse_local_vars(
  items: &[Expr],
  needs_value: bool,
) -> Result<Vec<LocalVar>, LocalSpecError> {
  let mut vars: Vec<LocalVar> = Vec::new();
  for item in items {
    // `Expr::Raw` is unparsed fallback text ("x = 5") that substitution can
    // leave behind; re-parse it rather than rejecting it.
    let item = &match item {
      Expr::Raw(s) => string_to_expr(s.trim()).unwrap_or_else(|_| item.clone()),
      other => other.clone(),
    };
    let var = if let Some((target, value, delayed)) = as_local_assignment(item)
    {
      let Expr::Identifier(name) = target else {
        return Err(LocalSpecError::BadAssignment {
          item: Box::new(item.clone()),
          target: Box::new(target.clone()),
        });
      };
      LocalVar {
        name: name.clone(),
        init: Some(value.clone()),
        delayed,
      }
    } else if let Expr::Identifier(name) = item {
      if needs_value {
        return Err(LocalSpecError::NeedsValue(item.clone()));
      }
      LocalVar {
        name: name.clone(),
        init: None,
        delayed: false,
      }
    } else if needs_value {
      // `With` reports every non-assignment as a missing value, even a
      // number: `With[{3, x}, …]` says "Variable 3 … requires a value".
      return Err(LocalSpecError::NeedsValue(item.clone()));
    } else {
      return Err(LocalSpecError::NotSymbol(item.clone()));
    };
    if vars.iter().any(|v| v.name == var.name) {
      return Err(LocalSpecError::Duplicate(var.name));
    }
    vars.push(var);
  }
  Ok(vars)
}

/// AST-based Module implementation to avoid interpret() recursion
pub fn module_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(format!(
      "Module expects 2 arguments; {} given",
      args.len()
    )));
  }

  let vars_expr = &args[0];
  let body_expr = &args[1];

  // Parse variable declarations from the first argument (should be a List)
  let local_vars = match vars_expr {
    Expr::List(items) => match parse_local_vars(items, false) {
      Ok(vars) => vars,
      Err(err) => {
        err.emit("Module", vars_expr);
        return Ok(unevaluated("Module", args));
      }
    },
    _ => return Ok(non_list_local_spec("Module", args)),
  };

  // Module scopes lexically: each local is renamed to a fresh var$n
  // symbol throughout the body, so a global function called from the
  // body sees the untouched global symbol (Block, by contrast, rebinds
  // dynamically). Initializers evaluate first, in the enclosing scope
  // and before any renaming — Module[{a = 1, b = a}, ...] takes the
  // OUTER a for b.
  let mut init_values: Vec<Option<Expr>> = Vec::new();
  for var in &local_vars {
    init_values.push(match (&var.init, var.delayed) {
      // `Module[{x := f[]}, …]` keeps the right-hand side unevaluated, the
      // way SetDelayed does; it is evaluated when the body reads `x`.
      (Some(expr), false) => Some(evaluate_expr_to_expr(expr)?),
      (Some(expr), true) => Some(expr.clone()),
      (None, _) => None,
    });
  }
  let mut body_owned = body_expr.clone();
  for (var, init) in local_vars.iter().zip(init_values) {
    let var_name = &var.name;
    let fresh = crate::functions::scoping::unique_symbol(var_name);
    body_owned = crate::syntax::rename_symbol(&body_owned, var_name, &fresh);
    if let Some(value) = init {
      // The fresh symbol is globally unique, so the binding needs no
      // cleanup — and definitions or closures escaping the Module keep
      // working, like Wolfram's Temporary module variables.
      ENV.with(|e| e.borrow_mut().insert(fresh, StoredValue::ExprVal(value)));
    }
  }
  let body_expr = &body_owned;

  // Evaluate body. Wolfram leaves Return[val] symbolic at Module's
  // boundary — wrap any ReturnValue exception into a literal `Return[val]`
  // so e.g. `Module[{}, Return[1]]` evaluates to `Return[1]`. The top-level
  // display path in interpret() unwraps it back to `1` for the REPL.
  let result = match evaluate_expr_to_expr(body_expr) {
    Err(InterpreterError::ReturnValue(val)) => Ok(call1("Return", *val)),
    other => other,
  };

  // Handle Condition in body: evaluate test while locals are still in scope

  match &result {
    Ok(Expr::FunctionCall { name, args })
      if name == "Condition" && args.len() == 2 =>
    {
      match evaluate_expr_to_expr(&args[1]) {
        Ok(Expr::Identifier(ref s)) if s == "True" => {
          evaluate_expr_to_expr(&args[0])
        }
        Ok(test_val) => Ok(call("Condition", vec![args[0].clone(), test_val])),
        Err(e) => Err(e),
      }
    }
    _ => result,
  }
}

/// AST-based Block implementation - dynamic scoping (like Module but without unique symbols).
/// Block[{x = 1, y}, body] - variables are localized but use their original names.
pub fn block_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(format!(
      "Block expects 2 arguments; {} given",
      args.len()
    )));
  }

  let vars_expr = &args[0];
  let body_expr = &args[1];

  // Parse variable declarations (same as Module)
  let local_vars = match vars_expr {
    Expr::List(items) => match parse_local_vars(items, false) {
      Ok(vars) => vars,
      Err(err) => {
        err.emit("Block", vars_expr);
        return Ok(unevaluated("Block", args));
      }
    },
    _ => return Ok(non_list_local_spec("Block", args)),
  };

  // Take every value each local stands for and set up the new ones.
  //
  // Block localizes the whole symbol, not just its own value: inside
  // `Block[{f}, …]` an `f[x_] := …` defined outside is gone, `Attributes[f]`
  // is `{}`, and `f::msg` is unset — which is what lets a package temporarily
  // neutralize a function (`Block[{Simp}, SetAttributes[Simp, HoldAll]; …]`
  // in Rubi) instead of running it.
  let mut prev: Vec<crate::evaluator::symbol_values::SymbolValues> = Vec::new();

  for var in &local_vars {
    let var_name = &var.name;
    // Evaluate the initializer with the *previous* binding still active (so
    // `Block[{x = n+2, n}, ...]` sees the outer `n`). A delayed `x := v`
    // keeps its right-hand side unevaluated.
    let init = match &var.init {
      Some(expr) if var.delayed => Some(expr.clone()),
      Some(expr) => Some(evaluate_expr_to_expr(expr)?),
      None => None,
    };
    prev.push(crate::evaluator::symbol_values::take_symbol_values(
      var_name,
    ));
    if let Some(evaluated) = init {
      ENV.with(|e| {
        e.borrow_mut()
          .insert(var_name.clone(), StoredValue::ExprVal(evaluated))
      });
    }
  }

  // Evaluate body. Block, like Module, wraps any escaping Return[val]
  // into a literal `Return[val]` Expr so the symbolic value matches
  // wolframscript: `Block[{}, Return[42]]` ⇒ `Return[42]`. The top-level
  // display path unwraps it for the REPL.
  let result = match evaluate_expr_to_expr(body_expr) {
    Err(InterpreterError::ReturnValue(val)) => Ok(call1("Return", *val)),
    other => other,
  };

  // Handle Condition in body: evaluate test while locals are still in scope
  let result = match &result {
    Ok(Expr::FunctionCall { name, args })
      if name == "Condition" && args.len() == 2 =>
    {
      match evaluate_expr_to_expr(&args[1]) {
        Ok(Expr::Identifier(ref s)) if s == "True" => {
          evaluate_expr_to_expr(&args[0])
        }
        Ok(test_val) => Ok(call("Condition", vec![args[0].clone(), test_val])),
        Err(e) => Err(e),
      }
    }
    _ => result,
  };

  // Restore previous definitions (even if body returned an error), innermost
  // local first so nested Blocks on the same symbol unwind in order.
  for saved in prev.into_iter().rev() {
    crate::evaluator::symbol_values::restore_symbol_values(saved);
  }

  // The value a Block returns is evaluated once more, now that the localized
  // symbols mean what they did outside: `n = 10; Block[{n}, n]` is 10, even
  // though the body — where `n` has no value — evaluated to the bare symbol.
  // Only a result that still names a local can change, and a `Return[…]`
  // wrapper must survive as the literal value it stands for.
  match &result {
    Ok(value)
      if !matches!(value, Expr::FunctionCall { name, .. } if name == "Return")
        && local_vars.iter().any(|v| mentions_symbol(value, &v.name)) =>
    {
      evaluate_expr_to_expr(value)
    }
    _ => result,
  }
}

/// Whether `expr` mentions the symbol `name` anywhere — as a value, as a head,
/// or inside a pattern. Asked through `FreeQ`, so it counts exactly what the
/// language counts.
fn mentions_symbol(expr: &Expr, name: &str) -> bool {
  matches!(
    crate::functions::predicate_ast::free_q_ast(&[
      expr.clone(),
      Expr::Identifier(name.to_string()),
    ]),
    Ok(Expr::Identifier(ref s)) if s == "False"
  )
}

/// `BlockRandom[expr]` — evaluate `expr` with the global random generator
/// state localized, so a `SeedRandom[…]` call inside `expr` does not affect
/// the random sequence seen after `BlockRandom` returns.
///
/// `BlockRandom[expr, RandomSeeding -> spec]` additionally seeds the localized
/// generator before evaluating the body: `Inherited` (the default) keeps the
/// ambient state, `Automatic` reseeds non-reproducibly, and an integer or
/// string is handed to `SeedRandom`, making the block reproducible.
pub fn block_random_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.is_empty() {
    return Err(InterpreterError::EvaluationError(
      "BlockRandom expects at least 1 argument; 0 given".to_string(),
    ));
  }

  // Everything past the body must be an option rule (or a list of rules).
  fn rule_parts(opt: &Expr) -> Option<(&Expr, &Expr)> {
    match opt {
      Expr::Rule {
        pattern,
        replacement,
      } => Some((pattern.as_ref(), replacement.as_ref())),
      Expr::FunctionCall { name, args }
        if name == "Rule" && args.len() == 2 =>
      {
        Some((&args[0], &args[1]))
      }
      _ => None,
    }
  }
  let mut options: Vec<(String, &Expr)> = Vec::new();
  for arg in &args[1..] {
    let flat: Vec<&Expr> = match arg {
      Expr::List(items) => items.iter().collect(),
      other => vec![other],
    };
    for opt in flat {
      if let Some((Expr::Identifier(name), value)) = rule_parts(opt) {
        options.push((name.clone(), value));
      } else {
        crate::emit_message(&format!(
          "BlockRandom::nonopt: Options expected (instead of {}) beyond \
           position 1 in {}. An option must be a rule or a list of rules.",
          crate::syntax::expr_to_output(opt),
          crate::syntax::expr_to_output(&unevaluated("BlockRandom", args))
        ));
        return Ok(unevaluated("BlockRandom", args));
      }
    }
  }
  if let Some((name, _)) = options.iter().find(|(n, _)| n != "RandomSeeding") {
    crate::emit_message(&format!(
      "BlockRandom::optx: Unknown option {name} in {}.",
      crate::syntax::expr_to_output(&unevaluated("BlockRandom", args))
    ));
    return Ok(unevaluated("BlockRandom", args));
  }

  let saved = crate::snapshot_rng_state();
  // Seed the localized generator, if asked. Anything that is neither
  // Inherited/Automatic nor a SeedRandom-compatible seed falls back to the
  // default (inherited) seeding after a message, matching wolframscript.
  if let Some((_, spec)) = options.iter().find(|(n, _)| n == "RandomSeeding") {
    let seeded = match spec {
      Expr::Identifier(s) if s == "Inherited" => Ok(()),
      Expr::Identifier(s) if s == "Automatic" => {
        crate::functions::math_ast::seed_random_ast(&[]).map(|_| ())
      }
      Expr::Integer(_) | Expr::String(_) => {
        crate::functions::math_ast::seed_random_ast(std::slice::from_ref(spec))
          .map(|_| ())
      }
      _ => Err(InterpreterError::EvaluationError(String::new())),
    };
    if seeded.is_err() {
      crate::emit_message(&format!(
        "BlockRandom::seeding: The value of the option RandomSeeding -> {} is \
         not Automatic, Inherited, an integer, string or valid \
         RandomGeneratorState expression. Using the default seeding for \
         BlockRandom instead.",
        crate::syntax::expr_to_output(spec)
      ));
    }
  }
  let result = evaluate_expr_to_expr(&args[0]);
  crate::restore_rng_state(saved);
  result
}

/// Valid domain names for Element[]
const VALID_DOMAINS: &[&str] = &[
  "Primes",
  "Integers",
  "Rationals",
  "Algebraics",
  "Reals",
  "Complexes",
  "Booleans",
  "PositiveReals",
  "PositiveIntegers",
  "NonNegativeReals",
  "NonNegativeIntegers",
  "NegativeReals",
  "NegativeIntegers",
];

/// Known real-valued constants (parsed as Constant or Identifier)
const REAL_CONSTANTS: &[&str] = &[
  "Pi",
  "E",
  "Degree",
  "EulerGamma",
  "GoldenRatio",
  "Catalan",
  "Khinchin",
  "Glaisher",
];

/// Constants known to be irrational (so not in Rationals).
const IRRATIONAL_CONSTANTS: &[&str] = &["Pi", "E", "Degree", "GoldenRatio"];
/// Constants known to be transcendental (so not in Algebraics).
const TRANSCENDENTAL_CONSTANTS: &[&str] = &["Pi", "E", "Degree"];
/// Constants known to be algebraic (so in Algebraics).
const ALGEBRAIC_CONSTANTS: &[&str] = &["GoldenRatio"];

/// A literal rational number: an integer or `Rational[p, q]`.
fn is_rational_literal(e: &Expr) -> bool {
  matches!(e, Expr::Integer(_) | Expr::BigInteger(_))
    || matches!(e, Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2)
}

/// A non-integer literal rational, i.e. `Rational[p, q]` (always stored in
/// lowest terms with q != 1).
fn is_non_integer_rational(e: &Expr) -> bool {
  matches!(e, Expr::FunctionCall { name, args }
    if name == "Rational" && args.len() == 2)
}

/// Check if an expression is a member of a given domain
fn is_member_of_domain(expr: &Expr, domain: &str) -> Option<bool> {
  // Infinity, -Infinity and ComplexInfinity are DirectedInfinity objects, not
  // members of any number domain (Reals, Integers, Complexes, …). Without this
  // guard Infinity matches the real/complex constant arms below.
  if crate::functions::predicate_ast::is_directed_infinity(expr) {
    return Some(false);
  }
  match domain {
    "Integers" => match expr {
      Expr::Integer(_) | Expr::BigInteger(_) => Some(true),
      Expr::Real(f) => Some(*f == f.floor() && f.is_finite()),
      // Rational[n, d] with d != 1 is not an integer
      Expr::FunctionCall { name, args }
        if name == "Rational" && args.len() == 2 =>
      {
        match (&args[0], &args[1]) {
          (Expr::Integer(_), Expr::Integer(1)) => Some(true),
          (Expr::Integer(_), Expr::Integer(_)) => Some(false),
          _ => None,
        }
      }
      // Known constants like Pi, E are not integers
      Expr::Constant(c) if REAL_CONSTANTS.contains(&c.as_str()) => Some(false),
      Expr::Identifier(name) if name == "I" => Some(false),
      _ => None,
    },
    "Primes" => match expr {
      Expr::Integer(n) => {
        Some(*n >= 2 && crate::functions::math_ast::is_prime_i128(*n))
      }
      Expr::Real(_) => Some(false),
      Expr::Constant(_) => Some(false),
      _ => None,
    },
    "Rationals" => match expr {
      Expr::Integer(_) | Expr::BigInteger(_) => Some(true),
      Expr::FunctionCall { name, args }
        if name == "Rational" && args.len() == 2 =>
      {
        Some(true)
      }
      // Known irrational constants (Pi, E, Degree, GoldenRatio); other
      // constants (EulerGamma, Catalan, …) have open irrationality and stay
      // unevaluated.
      Expr::Constant(c) | Expr::Identifier(c)
        if IRRATIONAL_CONSTANTS.contains(&c.as_str()) =>
      {
        Some(false)
      }
      // The imaginary unit is not real, hence not rational.
      Expr::Identifier(name) if name == "I" => Some(false),
      // A surviving radical Power[rational, non-integer rational] is
      // irrational (perfect powers are already simplified to integers).
      // Radicals are stored as a Power BinaryOp.
      Expr::BinaryOp {
        op: BinaryOperator::Power,
        left,
        right,
      } if is_rational_literal(left) && is_non_integer_rational(right) => {
        Some(false)
      }
      _ => None,
    },
    "Reals" => match expr {
      Expr::Integer(_) | Expr::BigInteger(_) | Expr::Real(_) => Some(true),
      Expr::FunctionCall { name, args }
        if name == "Rational" && args.len() == 2 =>
      {
        Some(true)
      }
      // Known real constants
      Expr::Constant(c) if REAL_CONSTANTS.contains(&c.as_str()) => Some(true),
      Expr::Identifier(name) if REAL_CONSTANTS.contains(&name.as_str()) => {
        Some(true)
      }
      Expr::Identifier(name) if name == "I" => Some(false),
      _ => {
        // Exact complex with nonzero imaginary part → not real.
        if let Some((_, (im, _))) =
          crate::functions::math_ast::try_extract_complex_exact(expr)
          && im != 0
        {
          return Some(false);
        }
        // Real-valued NumericQ expressions (Sqrt[2], Log[2], Pi^2, sums of
        // reals, …) are real.
        if crate::functions::math_ast::is_real_valued(expr) {
          return Some(true);
        }
        // A numeric expression with a nonzero imaginary part (e.g.
        // Sqrt[-2] = I Sqrt[2]) is not real.
        if let Some((_, im)) =
          crate::functions::math_ast::try_extract_complex_f64(expr)
          && im != 0.0
        {
          return Some(false);
        }
        None
      }
    },
    "Complexes" => match expr {
      Expr::Integer(_)
      | Expr::BigInteger(_)
      | Expr::Real(_)
      | Expr::BigFloat(_, _) => Some(true),
      Expr::FunctionCall { name, args }
        if name == "Rational" && args.len() == 2 =>
      {
        Some(true)
      }
      // Known constants are complex numbers too
      Expr::Constant(c) if REAL_CONSTANTS.contains(&c.as_str()) => Some(true),
      Expr::Identifier(name) if REAL_CONSTANTS.contains(&name.as_str()) => {
        Some(true)
      }
      Expr::Identifier(name) if name == "I" => Some(true),
      _ => {
        // Any value that evaluates to a number — real (Sqrt[2], Log[2]) or
        // complex (I Sqrt[2]) — is a complex number.
        if crate::functions::math_ast::try_extract_complex_exact(expr).is_some()
          || crate::functions::math_ast::is_real_valued(expr)
          || crate::functions::math_ast::try_extract_complex_f64(expr)
            .is_some_and(|(re, im)| re.is_finite() && im.is_finite())
        {
          Some(true)
        } else {
          None
        }
      }
    },
    "Booleans" => match expr {
      Expr::Identifier(name) if name == "True" || name == "False" => Some(true),
      _ => Some(false),
    },
    "Algebraics" => match expr {
      Expr::Integer(_) | Expr::BigInteger(_) => Some(true),
      Expr::FunctionCall { name, args }
        if name == "Rational" && args.len() == 2 =>
      {
        Some(true)
      }
      // The imaginary unit and known algebraic constants are algebraic.
      Expr::Identifier(name) if name == "I" => Some(true),
      Expr::Constant(c) | Expr::Identifier(c)
        if ALGEBRAIC_CONSTANTS.contains(&c.as_str()) =>
      {
        Some(true)
      }
      // Known transcendental constants are not algebraic.
      Expr::Constant(c) | Expr::Identifier(c)
        if TRANSCENDENTAL_CONSTANTS.contains(&c.as_str()) =>
      {
        Some(false)
      }
      // A radical Power[rational, rational] (e.g. Sqrt[2], 2^(1/3)) is
      // algebraic. Radicals are stored as a Power BinaryOp.
      Expr::BinaryOp {
        op: BinaryOperator::Power,
        left,
        right,
      } if is_rational_literal(left) && is_rational_literal(right) => {
        Some(true)
      }
      _ => None,
    },
    "PositiveReals" => match expr {
      Expr::Integer(n) => Some(*n > 0),
      Expr::Real(f) => Some(*f > 0.0 && f.is_finite()),
      Expr::FunctionCall { name, args }
        if name == "Rational" && args.len() == 2 =>
      {
        if let (Expr::Integer(n), Expr::Integer(d)) = (&args[0], &args[1]) {
          Some((*n > 0 && *d > 0) || (*n < 0 && *d < 0))
        } else {
          None
        }
      }
      Expr::Constant(c) if REAL_CONSTANTS.contains(&c.as_str()) => Some(true),
      Expr::Identifier(name) if REAL_CONSTANTS.contains(&name.as_str()) => {
        Some(true)
      }
      _ => None,
    },
    "PositiveIntegers" => match expr {
      Expr::Integer(n) => Some(*n > 0),
      Expr::Real(f) => Some(*f > 0.0 && *f == f.floor() && f.is_finite()),
      _ => None,
    },
    "NonNegativeReals" => match expr {
      Expr::Integer(n) => Some(*n >= 0),
      Expr::Real(f) => Some(*f >= 0.0 && f.is_finite()),
      Expr::FunctionCall { name, args }
        if name == "Rational" && args.len() == 2 =>
      {
        if let (Expr::Integer(n), Expr::Integer(d)) = (&args[0], &args[1]) {
          Some((*n >= 0 && *d > 0) || (*n <= 0 && *d < 0))
        } else {
          None
        }
      }
      Expr::Constant(c) if REAL_CONSTANTS.contains(&c.as_str()) => Some(true),
      Expr::Identifier(name) if REAL_CONSTANTS.contains(&name.as_str()) => {
        Some(true)
      }
      _ => None,
    },
    "NonNegativeIntegers" => match expr {
      Expr::Integer(n) => Some(*n >= 0),
      Expr::Real(f) => Some(*f >= 0.0 && *f == f.floor() && f.is_finite()),
      _ => None,
    },
    "NegativeReals" => match expr {
      Expr::Integer(n) => Some(*n < 0),
      Expr::Real(f) => Some(*f < 0.0 && f.is_finite()),
      Expr::FunctionCall { name, args }
        if name == "Rational" && args.len() == 2 =>
      {
        if let (Expr::Integer(n), Expr::Integer(d)) = (&args[0], &args[1]) {
          Some((*n < 0 && *d > 0) || (*n > 0 && *d < 0))
        } else {
          None
        }
      }
      _ => None,
    },
    "NegativeIntegers" => match expr {
      Expr::Integer(n) => Some(*n < 0),
      Expr::Real(f) => Some(*f < 0.0 && *f == f.floor() && f.is_finite()),
      _ => None,
    },
    _ => None,
  }
}

/// Element[x, domain] - Test or assert domain membership
pub fn element_ast(x: &Expr, domain: &Expr) -> Result<Expr, InterpreterError> {
  let domain_name = match domain {
    Expr::Identifier(name) => name.as_str(),
    _ => {
      return Ok(call("Element", vec![x.clone(), domain.clone()]));
    }
  };

  // Validate domain name
  if !VALID_DOMAINS.contains(&domain_name) {
    crate::emit_message(&format!(
      "Element::bset: The second argument {domain_name} of Element should be one of: Primes, Integers, Rationals, Algebraics, Reals, Complexes or Booleans."
    ));
    return Ok(call("Element", vec![x.clone(), domain.clone()]));
  }

  // Handle Alternatives: Element[a | b | c, dom]. `a | b | c` evaluates to a
  // flat Alternatives[...] FunctionCall, but held/explicit forms may still be
  // a nested BinaryOp — accept both.
  let is_alternatives = matches!(
    x,
    Expr::BinaryOp {
      op: BinaryOperator::Alternatives,
      ..
    }
  ) || matches!(x, Expr::FunctionCall { name, .. } if name == "Alternatives");
  if is_alternatives {
    let alts = collect_alternatives(x);
    let mut remaining = Vec::new();
    for alt in &alts {
      match is_member_of_domain(alt, domain_name) {
        Some(true) => {} // Known member, skip
        Some(false) => {
          return Ok(bool_expr(false));
        }
        None => remaining.push(alt.clone()),
      }
    }
    if remaining.is_empty() {
      return Ok(bool_expr(true));
    }
    // Rebuild Alternatives from remaining
    let alt_expr = remaining
      .into_iter()
      .reduce(|acc, e| Expr::BinaryOp {
        op: BinaryOperator::Alternatives,
        left: Box::new(acc),
        right: Box::new(e),
      })
      .unwrap();
    return Ok(call("Element", vec![alt_expr, domain.clone()]));
  }

  // Handle lists: Element[{a, b, c}, dom] → Element[a | b | c, dom]
  if let Expr::List(items) = x {
    if items.is_empty() {
      return Ok(bool_expr(true));
    }
    // Convert list to Alternatives and recurse
    let alt_expr = items
      .iter()
      .cloned()
      .reduce(|acc, e| Expr::BinaryOp {
        op: BinaryOperator::Alternatives,
        left: Box::new(acc),
        right: Box::new(e),
      })
      .unwrap();
    return element_ast(&alt_expr, domain);
  }

  // Element[Plus[a, b, c, ...], Reals] / Integers / Rationals: drop any
  // summands already known to be in the domain — the remainder is in the
  // domain iff the original sum is. If everything drops out, the result is
  // True; if a single term remains, re-emit `Element[term, dom]`.
  if matches!(
    domain_name,
    "Reals" | "Integers" | "Rationals" | "Algebraics" | "Complexes"
  ) && let Expr::FunctionCall { name, args } = x
    && name == "Plus"
    && args.len() >= 2
  {
    let mut remaining: Vec<Expr> = Vec::new();
    let mut any_dropped = false;
    let mut conflict = false;
    for a in args {
      match is_member_of_domain(a, domain_name) {
        Some(true) => any_dropped = true,
        Some(false) => {
          conflict = true;
          remaining.push(a.clone());
        }
        None => remaining.push(a.clone()),
      }
    }
    if !any_dropped && !conflict {
      // No simplification possible; fall through.
    } else if remaining.is_empty() {
      return Ok(bool_expr(true));
    } else if any_dropped && !conflict {
      let reduced = if remaining.len() == 1 {
        remaining.into_iter().next().unwrap()
      } else {
        call("Plus", remaining)
      };
      return element_ast(&reduced, domain);
    }
  }

  // Simple case: check single element
  match is_member_of_domain(x, domain_name) {
    Some(true) => Ok(bool_expr(true)),
    Some(false) => Ok(bool_expr(false)),
    None => Ok(call("Element", vec![x.clone(), domain.clone()])),
  }
}

/// NotElement[x, domain] - Test non-membership of an expression in a mathematical domain
pub fn not_element_ast(
  x: &Expr,
  domain: &Expr,
) -> Result<Expr, InterpreterError> {
  let result = element_ast(x, domain)?;
  match &result {
    Expr::Identifier(name) if name == "True" => Ok(bool_expr(false)),
    Expr::Identifier(name) if name == "False" => Ok(bool_expr(true)),
    _ => {
      // Element returned unevaluated, so NotElement stays unevaluated too
      Ok(call("NotElement", vec![x.clone(), domain.clone()]))
    }
  }
}

/// Collect all alternatives from a nested Alternatives expression
fn collect_alternatives(expr: &Expr) -> Vec<Expr> {
  match expr {
    Expr::BinaryOp {
      op: BinaryOperator::Alternatives,
      left,
      right,
    } => {
      let mut result = collect_alternatives(left);
      result.extend(collect_alternatives(right));
      result
    }
    // `a | b | c` evaluates to a flat Alternatives[...] FunctionCall.
    Expr::FunctionCall { name, args } if name == "Alternatives" => {
      args.iter().flat_map(collect_alternatives).collect()
    }
    _ => vec![expr.clone()],
  }
}

/// AST-based Assuming: Assuming[assum, body]
/// Evaluates body with $Assumptions set to assum.
pub fn assuming_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 2 {
    return Err(InterpreterError::EvaluationError(format!(
      "Assuming expects 2 arguments; {} given",
      args.len()
    )));
  }

  let assumption = evaluate_expr_to_expr(&args[0])?;

  // Save current $Assumptions
  let prev = ENV.with(|e| e.borrow().get("$Assumptions").cloned());

  // Set $Assumptions to the assumption
  let val = expr_to_string(&assumption);
  ENV.with(|e| {
    e.borrow_mut()
      .insert("$Assumptions".to_string(), StoredValue::Raw(val))
  });

  // If the assumption is a simple equality `var == value` (or a List of
  // such), AND the body contains an Integrate/Sum/Product/Limit (where
  // wolframscript's ConditionalExpression handling would specialise the
  // result), substitute `var → value` in the body before evaluating.
  // Matches wolframscript: `Assuming[n == 1, Integrate[x^n, {x, 0, 1}]]`
  // returns `1/2`. Cases like `Assuming[n == 1, x^n]` keep `x^n`
  // because wolframscript also doesn't substitute there.
  let body = if contains_assumption_consumer(&args[1]) {
    apply_assumption_substitutions(&args[1], &assumption)
  } else {
    args[1].clone()
  };

  // Evaluate the body expression
  let result = evaluate_expr_to_expr(&body);

  // Restore previous $Assumptions (even if body returned an error)
  ENV.with(|e| {
    let mut env = e.borrow_mut();
    if let Some(v) = prev {
      env.insert("$Assumptions".to_string(), v);
    } else {
      env.remove("$Assumptions");
    }
  });

  result
}

/// Does `expr` contain a function whose result depends on
/// `$Assumptions` (Integrate / Sum / Product / Limit)? Used as a guard
/// before specialising the body of `Assuming`, so `Assuming[n == 1, x^n]`
/// keeps `x^n` (matching wolframscript) while
/// `Assuming[n == 1, Integrate[x^n, ...]]` substitutes.
fn contains_assumption_consumer(expr: &Expr) -> bool {
  fn is_consumer(name: &str) -> bool {
    matches!(name, "Integrate" | "Sum" | "Product" | "Limit")
  }
  match expr {
    Expr::FunctionCall { name, args } => {
      is_consumer(name) || args.iter().any(contains_assumption_consumer)
    }
    Expr::List(items) => items.iter().any(contains_assumption_consumer),
    Expr::BinaryOp { left, right, .. } => {
      contains_assumption_consumer(left) || contains_assumption_consumer(right)
    }
    Expr::UnaryOp { operand, .. } => contains_assumption_consumer(operand),
    Expr::Comparison { operands, .. } => {
      operands.iter().any(contains_assumption_consumer)
    }
    _ => false,
  }
}

/// Walk `assumption` looking for equality assumptions of the form
/// `Equal[symbol, value]` (parsed as Comparison or FunctionCall, with
/// optional List wrapper for conjunctions) and substitute each
/// `symbol → value` in `body`. Used by `assuming_ast` to specialise an
/// integration / sum / etc. before evaluating it.
fn apply_assumption_substitutions(body: &Expr, assumption: &Expr) -> Expr {
  fn extract_equalities(a: &Expr, out: &mut Vec<(String, Expr)>) {
    match a {
      // List of assumptions: walk each entry.
      Expr::List(items) => {
        for item in items {
          extract_equalities(item, out);
        }
      }
      // And[…] (sometimes from `&&`): walk each clause.
      Expr::FunctionCall { name, args } if name == "And" => {
        for arg in args {
          extract_equalities(arg, out);
        }
      }
      // FullForm Equal[var, value]
      Expr::FunctionCall { name, args }
        if name == "Equal" && args.len() == 2 =>
      {
        if let Expr::Identifier(var) = &args[0] {
          out.push((var.clone(), args[1].clone()));
        }
      }
      // Parsed `var == value` is `Comparison { operands: [var, value],
      // operators: [Equal] }` (not a FunctionCall).
      Expr::Comparison {
        operands,
        operators,
      } if operands.len() == 2
        && operators.len() == 1
        && operators[0] == ComparisonOp::Equal =>
      {
        if let Expr::Identifier(var) = &operands[0] {
          out.push((var.clone(), operands[1].clone()));
        }
      }
      _ => {}
    }
  }
  let mut pairs: Vec<(String, Expr)> = Vec::new();
  extract_equalities(assumption, &mut pairs);
  let mut result = body.clone();
  for (var, value) in &pairs {
    result = crate::syntax::substitute_variable(&result, var, value);
  }
  result
}

/// FilterRules[{rules...}, keys] - filter rules by matching keys
pub fn filter_rules_ast(
  rules: &Expr,
  keys: &Expr,
) -> Result<Expr, InterpreterError> {
  let Expr::List(rule_list) = rules else {
    return Ok(call("FilterRules", vec![rules.clone(), keys.clone()]));
  };

  // Build set of key names to keep
  let key_names: Vec<String> = match keys {
    Expr::List(items) => items.iter().map(expr_to_string).collect(),
    _ => vec![expr_to_string(keys)],
  };

  let mut result = Vec::new();
  for rule in rule_list {
    let rule_key = match rule {
      Expr::Rule { pattern, .. } | Expr::RuleDelayed { pattern, .. } => {
        expr_to_string(pattern)
      }
      Expr::FunctionCall { name, args }
        if (name == "Rule" || name == "RuleDelayed") && !args.is_empty() =>
      {
        expr_to_string(&args[0])
      }
      _ => continue,
    };
    if key_names.contains(&rule_key) {
      result.push(rule.clone());
    }
  }

  Ok(Expr::List(result.into()))
}

/// AST-based For loop: For[init, test, incr, body]
pub fn for_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() < 3 || args.len() > 4 {
    return Err(InterpreterError::EvaluationError(format!(
      "For expects 3 or 4 arguments; {} given",
      args.len()
    )));
  }

  let init = &args[0];
  let test = &args[1];
  let incr = &args[2];
  let body = args.get(3);

  // Mirror the While safety cap. Wolframscript has no For iteration limit;
  // we set a very high one so practical scripts run unhindered while a true
  // infinite loop still terminates eventually rather than hanging the host.
  const MAX_ITERATIONS: usize = 1_000_000_000;

  // Evaluate the initialization
  evaluate_expr_to_expr(init)?;

  let mut iterations = 0;
  loop {
    // Evaluate the test condition
    let test_result = evaluate_expr_to_expr(test)?;
    match test_result {
      Expr::Identifier(ref s) if s == "True" => {}
      Expr::Identifier(ref s) if s == "False" => break,
      _ => break,
    }

    // Evaluate the body (if provided). Return[val] inside the body
    // exits the loop and yields the literal `Return[val]` symbolic
    // expression — wolframscript renders it as `Return[val]` in
    // InputForm; interpret()'s top-level display unwraps it.
    if let Some(body) = body {
      match evaluate_expr_to_expr(body) {
        Ok(_) => {}
        Err(InterpreterError::BreakSignal) => break,
        Err(InterpreterError::ContinueSignal) => {}
        Err(InterpreterError::ReturnValue(val)) => {
          return Ok(call1("Return", *val));
        }
        Err(e) => return Err(e),
      }
    }

    // Evaluate the increment
    evaluate_expr_to_expr(incr)?;

    iterations += 1;
    if iterations >= MAX_ITERATIONS {
      return Err(InterpreterError::EvaluationError(
        "For: maximum iterations exceeded".into(),
      ));
    }
  }

  Ok(Expr::Identifier("Null".to_string()))
}

/// AST-based With implementation - substitutes bindings into body before evaluation.
/// With[{x = val, y = val2}, body] replaces x and y in body with evaluated values.
pub fn with_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  // With[{x = 1}, {y = x + 1}, …, body] scopes each specification inside the
  // ones before it, so it is exactly the nested form
  // With[{x = 1}, With[{y = x + 1}, …, body]]. Desugaring keeps a single
  // implementation for every arity, and makes the unevaluated form of a
  // malformed specification nest the way wolframscript's does.
  if args.len() > 2 {
    let inner = call("With", args[1..].to_vec());
    return with_ast(&[args[0].clone(), inner]);
  }

  let vars_expr = &args[0];
  let body_expr = &args[1];

  // Parse variable declarations from the first argument (should be a List)
  let bindings: Vec<(String, Expr)> = match vars_expr {
    Expr::List(items) => {
      let locals = match parse_local_vars(items, true) {
        Ok(vars) => vars,
        Err(err) => {
          err.emit("With", vars_expr);
          return Ok(unevaluated("With", args));
        }
      };
      let mut vars = Vec::new();
      for var in locals {
        // `With[{x := v}, …]` substitutes `v` unevaluated.
        let init = var.init.unwrap_or(Expr::Identifier("Null".to_string()));
        let val = if var.delayed {
          init
        } else {
          evaluate_expr_to_expr(&init)?
        };
        vars.push((var.name, val));
      }
      vars
    }
    _ => return Ok(non_list_local_spec("With", args)),
  };

  // Substitute all bindings into the body simultaneously
  // to prevent variable name leakage across bindings
  let binding_refs: Vec<(&str, &Expr)> = bindings
    .iter()
    .map(|(name, val)| (name.as_str(), val))
    .collect();
  let substituted =
    crate::syntax::substitute_variables(body_expr, &binding_refs);

  // Evaluate the substituted body. Like Module and Block, With leaves a
  // Return that reaches its boundary standing as `Return[val]`.
  crate::evaluator::evaluate_value(&substituted)
}

/// Recursively set a value at a path of indices within an Expr.
/// Supports lists and FunctionCall arguments (e.g., Grid[[1, row, col]]).
/// Supports Span indices (e.g., `A[[;;, 2]] = {6, 7}`).
pub fn set_part_deep(
  expr: &mut Expr,
  indices: &[Expr],
  value: &Expr,
) -> Result<(), InterpreterError> {
  if indices.is_empty() {
    *expr = value.clone();
    return Ok(());
  }

  // Handle `All` index (a[[All]] = ...) — equivalent to Span[1, All].
  // Threads the assignment over every position in the current list.
  if matches!(&indices[0], Expr::Identifier(s) | Expr::Constant(s) if s == "All")
  {
    let len = match expr {
      Expr::List(items) => items.len(),
      Expr::FunctionCall { args, .. } => args.len(),
      _ => {
        return Err(InterpreterError::EvaluationError(
          "Part assignment: cannot apply All to this expression".into(),
        ));
      }
    };
    let rhs_items = match value {
      Expr::List(items) if items.len() == len => Some(items.clone()),
      _ => None,
    };
    for i in 0..len {
      let elem_value = match &rhs_items {
        Some(items) => items[i].clone(),
        None => value.clone(),
      };
      let inner = match expr {
        Expr::List(items) => &mut items[i],
        Expr::FunctionCall { args, .. } => &mut args[i],
        _ => unreachable!(),
      };
      set_part_deep(inner, &indices[1..], &elem_value)?;
    }
    return Ok(());
  }

  // Handle Span index (e.g., 1;;n, ;;, 1;;-1) by threading the assignment
  // over each selected position in the current list.
  if let Expr::FunctionCall { name, args } = &indices[0]
    && name == "Span"
  {
    let len = match expr {
      Expr::List(items) => items.len() as i64,
      Expr::FunctionCall { args, .. } => args.len() as i64,
      _ => {
        return Err(InterpreterError::EvaluationError(
          "Part assignment: cannot apply Span to this expression".into(),
        ));
      }
    };
    let positions = resolve_span(args, len)?;
    // Match wolframscript: a List of the same length as the selection
    // distributes element-wise; any other RHS (scalar, or list with
    // different length) is broadcast as a whole to each position.
    let rhs_items = match value {
      Expr::List(items) if items.len() == positions.len() => {
        Some(items.clone())
      }
      _ => None,
    };
    for (i, &pos) in positions.iter().enumerate() {
      let actual_idx = (pos - 1) as usize;
      let elem_value = match &rhs_items {
        Some(items) => &items[i],
        None => value,
      };
      let inner = match expr {
        Expr::List(items) => &mut items[actual_idx],
        Expr::FunctionCall { args, .. } => &mut args[actual_idx],
        _ => unreachable!(),
      };
      set_part_deep(inner, &indices[1..], elem_value)?;
    }
    return Ok(());
  }

  // Handle List index (e.g., a[[{1, 3}]] = ...): assign to each selected
  // position. When the RHS is a List of the same length, distribute its
  // elements; otherwise broadcast the entire RHS to every position.
  if let Expr::List(index_items) = &indices[0] {
    let len = match expr {
      Expr::List(items) => items.len() as i64,
      Expr::FunctionCall { args, .. } => args.len() as i64,
      _ => {
        return Err(InterpreterError::EvaluationError(
          "Part assignment: cannot apply list index to this expression".into(),
        ));
      }
    };
    let mut positions = Vec::with_capacity(index_items.len());
    for item in index_items {
      let n = match item {
        Expr::Integer(n) => *n as i64,
        Expr::BigInteger(n) => {
          use num_traits::ToPrimitive;
          n.to_i64().ok_or_else(|| {
            InterpreterError::EvaluationError(
              "Part assignment: index too large".into(),
            )
          })?
        }
        _ => {
          return Err(InterpreterError::EvaluationError(
            "Part assignment: index must be an integer".into(),
          ));
        }
      };
      let actual_idx = if n < 0 { len + n } else { n - 1 };
      if actual_idx < 0 || actual_idx >= len {
        return Err(InterpreterError::EvaluationError(format!(
          "Part::partw: Part {n} of list does not exist."
        )));
      }
      positions.push(actual_idx as usize);
    }
    let distribute =
      matches!(value, Expr::List(items) if items.len() == positions.len());
    for (i, &actual_idx) in positions.iter().enumerate() {
      let elem_value: Expr = if distribute {
        if let Expr::List(items) = value {
          items[i].clone()
        } else {
          unreachable!()
        }
      } else {
        value.clone()
      };
      let inner = match expr {
        Expr::List(items) => &mut items[actual_idx],
        Expr::FunctionCall { args, .. } => &mut args[actual_idx],
        _ => unreachable!(),
      };
      set_part_deep(inner, &indices[1..], &elem_value)?;
    }
    return Ok(());
  }

  // Association assignment: an integer position selects the n-th value
  // (mirroring the Part read semantics), while a key selects the value for
  // that key (appending a new entry when the key is absent).
  if let Expr::Association(pairs) = expr {
    use num_traits::ToPrimitive;
    let pos = match &indices[0] {
      Expr::Integer(n) => Some(*n as i64),
      Expr::BigInteger(n) => n.to_i64(),
      _ => None,
    };
    if let Some(n) = pos {
      let len = pairs.len() as i64;
      let actual_idx = if n < 0 { len + n } else { n - 1 };
      if actual_idx < 0 || actual_idx >= len {
        return Err(InterpreterError::EvaluationError(format!(
          "Part::partw: Part {n} of association does not exist."
        )));
      }
      return set_part_deep(
        &mut pairs[actual_idx as usize].1,
        &indices[1..],
        value,
      );
    }
    // Key-based index: match on the string form (quote-insensitive).
    let key_cmp = expr_to_string(&indices[0]).trim_matches('"').to_string();
    if let Some(pair) = pairs
      .iter_mut()
      .find(|(k, _)| expr_to_string(k).trim_matches('"') == key_cmp)
    {
      return set_part_deep(&mut pair.1, &indices[1..], value);
    }
    // Absent key: append it when this is the final index; descending deeper
    // into a key that does not exist is an error (matching wolframscript).
    if indices.len() == 1 {
      pairs.push((indices[0].clone(), value.clone()));
      return Ok(());
    }
    return Err(InterpreterError::EvaluationError(format!(
      "Part::partw: Part {key_cmp} of association does not exist."
    )));
  }

  let idx = match &indices[0] {
    Expr::Integer(n) => *n as i64,
    Expr::BigInteger(n) => {
      use num_traits::ToPrimitive;
      n.to_i64().ok_or_else(|| {
        InterpreterError::EvaluationError(
          "Part assignment: index too large".into(),
        )
      })?
    }
    _ => {
      return Err(InterpreterError::EvaluationError(
        "Part assignment: index must be an integer".into(),
      ));
    }
  };

  match expr {
    Expr::List(items) => {
      let len = items.len() as i64;
      let actual_idx = if idx < 0 { len + idx } else { idx - 1 };
      if actual_idx < 0 || actual_idx >= len {
        return Err(InterpreterError::EvaluationError(format!(
          "Part::partw: Part {idx} of list does not exist."
        )));
      }
      set_part_deep(&mut items[actual_idx as usize], &indices[1..], value)
    }
    Expr::FunctionCall { args, .. } => {
      // Part 0 is the head, Part 1.. are arguments (1-indexed)
      if idx == 0 {
        return Err(InterpreterError::EvaluationError(
          "Cannot set Part 0 (head) of a function call".into(),
        ));
      }
      let actual_idx = (idx - 1) as usize;
      if actual_idx >= args.len() {
        return Err(InterpreterError::EvaluationError(format!(
          "Part::partw: Part {idx} of expression does not exist."
        )));
      }
      set_part_deep(&mut args[actual_idx], &indices[1..], value)
    }
    _ => Err(InterpreterError::EvaluationError(
      "Part assignment: cannot index into this expression".into(),
    )),
  }
}

/// Resolve a Span[start, end] (or Span[start, end, step]) over a sequence of
/// `len` elements into the list of 1-based positions it selects.
fn resolve_span(args: &[Expr], len: i64) -> Result<Vec<i64>, InterpreterError> {
  let to_pos = |e: &Expr, default: i64| -> Result<i64, InterpreterError> {
    match e {
      Expr::Integer(n) => {
        let n = *n as i64;
        Ok(if n < 0 { len + n + 1 } else { n })
      }
      Expr::Identifier(s) if s == "All" => Ok(default),
      _ => Err(InterpreterError::EvaluationError(
        "Part assignment: unsupported Span endpoint".into(),
      )),
    }
  };
  let (start, end, step) = match args.len() {
    2 => (to_pos(&args[0], 1)?, to_pos(&args[1], len)?, 1i64),
    3 => {
      let step_val = match &args[2] {
        Expr::Integer(n) => *n as i64,
        _ => {
          return Err(InterpreterError::EvaluationError(
            "Part assignment: unsupported Span step".into(),
          ));
        }
      };
      (to_pos(&args[0], 1)?, to_pos(&args[1], len)?, step_val)
    }
    _ => {
      return Err(InterpreterError::EvaluationError(
        "Part assignment: malformed Span".into(),
      ));
    }
  };
  if step == 0 {
    return Err(InterpreterError::EvaluationError(
      "Part assignment: Span step cannot be zero".into(),
    ));
  }
  let mut positions = Vec::new();
  let mut p = start;
  if step > 0 {
    while p <= end {
      if p < 1 || p > len {
        return Err(InterpreterError::EvaluationError(format!(
          "Part::partw: Part {p} of list does not exist."
        )));
      }
      positions.push(p);
      p += step;
    }
  } else {
    while p >= end {
      if p < 1 || p > len {
        return Err(InterpreterError::EvaluationError(format!(
          "Part::partw: Part {p} of list does not exist."
        )));
      }
      positions.push(p);
      p += step;
    }
  }
  Ok(positions)
}
