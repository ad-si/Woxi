#[allow(unused_imports)]
use super::*;
use crate::syntax::{BinaryOperator, UnaryOperator};
use std::cell::Cell;

thread_local! {
  /// When set, `set_delayed_ast` appends new rules to FUNC_DEFS in source
  /// order instead of inserting by pattern specificity. Used by
  /// `set_downvalues_from_rules` so `DownValues[f] := {...}` preserves the
  /// caller's ordering, matching Wolfram.
  static SUPPRESS_SPECIFICITY_SORT: Cell<bool> = const { Cell::new(false) };

  /// Names of forms registered via `Format[expr, FORM] := …` that should
  /// appear in `$PrintForms` (and `$OutputForms`) in addition to the
  /// builtin defaults. Insertion order is preserved.
  pub static USER_PRINT_FORMS: std::cell::RefCell<Vec<String>> =
    const { std::cell::RefCell::new(Vec::new()) };

  /// Format rules registered via `Format[pat, FORM] := body`, keyed by the
  /// head symbol of `pat`. Stored as `(form_name, lhs_pattern, rhs_body)`
  /// triples in source order.
  pub static FORMAT_VALUES: std::cell::RefCell<
    std::collections::HashMap<String, Vec<(String, Expr, Expr)>>,
  > = std::cell::RefCell::new(std::collections::HashMap::new());

  /// SubValue rules registered via `f[a][b] := …` (and deeper curried
  /// forms), keyed by the outermost head symbol `f`. Stored as
  /// `(lhs_curried_call, rhs_body)` pairs in source order.
  pub static SUB_VALUES: std::cell::RefCell<
    std::collections::HashMap<String, Vec<(Expr, Expr)>>,
  > = std::cell::RefCell::new(std::collections::HashMap::new());

  /// NValue rules registered via `N[sym, …] := …` / `N[sym, …] = …`,
  /// keyed by the symbol that the `N` head wraps. Stored as
  /// `(lhs_pattern, rhs_body)` pairs in source order, where
  /// `lhs_pattern` is the canonical
  /// `N[sym, {MachinePrecision, MachinePrecision}]` form Wolfram
  /// reports under `NValues`.
  pub static N_VALUES: std::cell::RefCell<
    std::collections::HashMap<String, Vec<(Expr, Expr)>>,
  > = std::cell::RefCell::new(std::collections::HashMap::new());
}

/// If `expr` is a pattern with a head constraint (e.g. `_Q`, `x_Q`, `Blank[Q]`,
/// `Pattern[x, Blank[Q]]`), return that head symbol so it can be used as the
/// tag in an UpSet/UpSetDelayed assignment. Returns None otherwise.
fn pattern_head_tag(expr: &Expr) -> Option<String> {
  match expr {
    Expr::Pattern { head: Some(h), .. } => Some(h.clone()),
    Expr::FunctionCall { name, args }
      if (name == "Blank"
        || name == "BlankSequence"
        || name == "BlankNullSequence")
        && args.len() == 1 =>
    {
      if let Expr::Identifier(h) = &args[0] {
        Some(h.clone())
      } else {
        None
      }
    }
    Expr::FunctionCall { name, args }
      if name == "Pattern" && args.len() == 2 =>
    {
      pattern_head_tag(&args[1])
    }
    _ => None,
  }
}

/// Return the head symbol of a `Format` LHS pattern so the rule can be keyed
/// in `FORMAT_VALUES`. For `F[x_]` returns `Some("F")`; for a bare symbol
/// returns the symbol name. Returns None for unrecognised shapes.
fn format_pattern_head(pat: &Expr) -> Option<String> {
  match pat {
    Expr::Identifier(s) => Some(s.clone()),
    Expr::FunctionCall { name, .. } => Some(name.clone()),
    _ => None,
  }
}

/// Register a user-defined print form (e.g. via `Format[expr, FORM] := …`).
/// No-op if already registered or if `name` is one of the builtin forms.
pub fn register_user_print_form(name: &str) {
  const BUILTIN: &[&str] = &[
    "InputForm",
    "OutputForm",
    "TextForm",
    "CForm",
    "FortranForm",
    "ScriptForm",
    "MathMLForm",
    "TeXForm",
    "StandardForm",
    "TraditionalForm",
  ];
  if BUILTIN.contains(&name) {
    return;
  }
  USER_PRINT_FORMS.with(|v| {
    let mut v = v.borrow_mut();
    if !v.iter().any(|s| s == name) {
      v.push(name.to_string());
    }
  });
}

fn suppress_specificity_sort() -> bool {
  SUPPRESS_SPECIFICITY_SORT.with(|c| c.get())
}

/// Walk an expression and wrap each `Rule`/`RuleDelayed` pattern (LHS) in
/// `HoldPattern[…]` so subsequent evaluation doesn't strip the LHS shape.
/// Used by NValues/UpValues/etc. assignment to mirror Wolfram's behavior of
/// implicitly wrapping rule LHSs (Wolfram returns `{HoldPattern[N[b, MachinePrecision]] :> 2}`
/// for `NValues[b] := {N[b, MachinePrecision] :> 2}`).
fn wrap_rule_lhs_in_holdpattern(expr: &Expr) -> Expr {
  fn already_held(p: &Expr) -> bool {
    matches!(p, Expr::FunctionCall { name, args }
      if (name == "HoldPattern" || name == "Verbatim") && args.len() == 1)
  }
  fn wrap_pattern(p: &Expr) -> Expr {
    if already_held(p) {
      p.clone()
    } else {
      Expr::FunctionCall {
        name: "HoldPattern".to_string(),
        args: vec![p.clone()].into(),
      }
    }
  }
  match expr {
    Expr::Rule {
      pattern,
      replacement,
    } => Expr::Rule {
      pattern: Box::new(wrap_pattern(pattern)),
      replacement: replacement.clone(),
    },
    Expr::RuleDelayed {
      pattern,
      replacement,
    } => Expr::RuleDelayed {
      pattern: Box::new(wrap_pattern(pattern)),
      replacement: replacement.clone(),
    },
    Expr::List(items) => {
      Expr::List(items.iter().map(wrap_rule_lhs_in_holdpattern).collect())
    }
    _ => expr.clone(),
  }
}

/// Resolve a function name through the environment to pick up Module-scoped
/// unique symbols.  When Module[{f}, …] creates a unique symbol like `f$1` and
/// stores it in the environment as `ENV["f"] = Raw("f$1")`, DownValue
/// definitions on `f` (i.e. `f[x_] := …`) must be stored under the resolved
/// name `f$1` so that subsequent calls (which evaluate `f` → `f$1`) can find
/// them.
fn resolve_func_name(name: &str) -> String {
  ENV.with(|e| {
    if let Some(StoredValue::Raw(val)) = e.borrow().get(name) {
      // Only resolve if the stored value looks like a Module-generated
      // unique symbol (contains '$')
      if val.contains('$') {
        return val.clone();
      }
    }
    name.to_string()
  })
}

/// Returns true when a rule body carries a whole-rule `/;` guard, i.e. it is
/// stored as a `Condition[body, test]` wrapper (`f[...] := body /; test`).
fn body_has_condition(body: &Expr) -> bool {
  matches!(
    body,
    Expr::FunctionCall { name, args } if name == "Condition" && args.len() == 2
  )
}

/// Whether a stored per-parameter condition marks a genuine literal-argument
/// definition (e.g. `f[0]`, stored as `param === value`). The synthetic
/// `Length[_lp] === N` guard emitted for list patterns also uses `SameQ` but
/// is NOT a literal-argument marker, so it is excluded — otherwise an
/// all-blank list rule like `f[{a_, b_}]` would be treated as a literal
/// definition and wrongly jump ahead of more specific overloads.
fn condition_is_literal_arg(c: &Expr) -> bool {
  if let Expr::Comparison {
    operators,
    operands,
  } = c
  {
    let is_length_check = matches!(
      operands.first(),
      Some(Expr::FunctionCall { name, .. }) if name == "Length"
    );
    !is_length_check
      && operators
        .iter()
        .any(|op| matches!(op, crate::syntax::ComparisonOp::SameQ))
  } else {
    false
  }
}

/// Compute a specificity score for a pattern rule based on its conditions,
/// blank types, and head constraints. Lower score = more specific = should be
/// tried first. Ordering: literal (SameQ) > head-constrained Blank > Blank >
/// BlankSequence > BlankNullSequence.
///
/// `body` is the rule's RHS; a whole-rule `/;` guard (stored as a
/// `Condition[body, test]` wrapper) makes the rule more specific, matching
/// Wolfram, where a guarded rule like `f[n_Integer, _] := 0 /; n < 0` is tried
/// before an unguarded but otherwise-more-specific rule like
/// `f[n_Integer, r_Integer] := …`.
fn pattern_specificity_score(
  blank_types: &[u8],
  heads: &[Option<String>],
  conditions: &[Option<Expr>],
  body: &Expr,
) -> u32 {
  // Literal-argument definitions (e.g. `f[0]`) are most specific.
  let is_literal = conditions.iter().flatten().any(condition_is_literal_arg);
  if is_literal {
    return 0;
  }
  // Use the maximum blank_type as primary score (higher = less specific)
  let max_blank = blank_types.iter().copied().max().unwrap_or(1) as u32;
  // Head constraints make a pattern more specific (subtract 1 for each)
  let head_bonus = heads.iter().filter(|h| h.is_some()).count() as u32;
  // Each stored constraint clause makes the rule more specific. A plain guard
  // or `PatternTest` slot counts once; a list pattern bundles several real
  // constraints (per-element `MatchQ[Part[_lp, i], …]` guards + any `/;`
  // guard) into one `And[…]` slot, so we count the meaningful clauses inside
  // it — the synthetic `Length[_lp] === N` check does not count. This makes a
  // `{0, x_}` rule more specific than an all-blank `{a_, b_}` rule regardless
  // of definition order, matching Wolfram.
  let cond_bonus: u32 = conditions
    .iter()
    .flatten()
    .map(count_specificity_clauses)
    .sum();
  // A whole-rule `/;` guard left on the body (non-list rules) adds specificity.
  let rule_cond_bonus = u32::from(body_has_condition(body));
  // Score: higher blank_type dominates, head/condition constraints reduce score
  (max_blank * 10).saturating_sub(head_bonus + cond_bonus + rule_cond_bonus)
}

/// Split a rule's stored arrays into its argument positions (head + blank type)
/// and its guard expressions, for partial-order comparison. Appended guard-only
/// slots (empty param name, line ~2149) contribute a guard but not a position;
/// a real position whose condition slot holds a `/;` guard contributes both a
/// position and a guard. `__StructuralPattern__` markers are structural, not
/// guards. A whole-rule `Condition[body, test]` wrapper also contributes a guard.
fn rule_positions_and_guards(
  params: &[String],
  heads: &[Option<String>],
  blank_types: &[u8],
  conditions: &[Option<Expr>],
  defaults: &[Option<Expr>],
  body: &Expr,
) -> (Vec<(Option<String>, u8, bool)>, Vec<Expr>) {
  let mut positions = Vec::new();
  let mut guards = Vec::new();
  for i in 0..heads.len() {
    let cond = conditions.get(i).and_then(|c| c.as_ref());
    let is_guard_slot = params.get(i).map(|p| p.is_empty()).unwrap_or(false);
    if is_guard_slot {
      if let Some(c) = cond {
        guards.push(c.clone());
      }
      continue;
    }
    let is_optional = defaults.get(i).map(|d| d.is_some()).unwrap_or(false);
    positions.push((heads[i].clone(), blank_types[i], is_optional));
    if let Some(c) = cond {
      let is_structural_marker = matches!(
        c,
        Expr::FunctionCall { name, .. } if name == "__StructuralPattern__"
      );
      if !is_structural_marker {
        guards.push(c.clone());
      }
    }
  }
  if let Expr::FunctionCall { name, args } = body
    && name == "Condition"
    && args.len() == 2
  {
    guards.push(args[1].clone());
  }
  (positions, guards)
}

/// Partial order on rules: returns true when rule `a`'s match set is a strict
/// subset of rule `b`'s, i.e. `a` is strictly more specific and must be tried
/// before `b`. Returns false when the two rules are incomparable (neither match
/// set contains the other), in which case the caller preserves definition
/// order — matching Wolfram, where e.g. `f[n_Integer, _] := 0 /; n < 0` and
/// `f[n_Integer, r_Integer] := …` are incomparable (a guarded but structurally
/// looser rule vs an unguarded but structurally tighter one) and fire in the
/// order they were entered.
#[allow(clippy::too_many_arguments)]
pub fn rule_dominates(
  a_params: &[String],
  a_heads: &[Option<String>],
  a_bt: &[u8],
  a_conds: &[Option<Expr>],
  a_defaults: &[Option<Expr>],
  a_body: &Expr,
  b_params: &[String],
  b_heads: &[Option<String>],
  b_bt: &[u8],
  b_conds: &[Option<Expr>],
  b_defaults: &[Option<Expr>],
  b_body: &Expr,
) -> bool {
  // Some patterns encode their constraints as opaque conditions the
  // position/guard split below cannot compare structurally:
  //   * list-destructuring patterns (`f[{a_, b_}]`) → `MatchQ`/`Length`
  //     conditions on a single `_lp` param;
  //   * nested structural patterns (`f[g[x_]]`) → a `__StructuralPattern__`
  //     marker condition on a `__sp` param.
  // For those, fall back to the linear specificity score (`score(a) < score(b)`
  // ⇔ `a` is more specific), preserving their established ordering.
  let needs_score_fallback = |params: &[String], conds: &[Option<Expr>]| {
    params
      .iter()
      .any(|p| p.starts_with("_lp") || p.starts_with("__sp"))
      || conds.iter().flatten().any(|c| {
        matches!(c, Expr::FunctionCall { name, .. } if name == "__StructuralPattern__")
      })
  };
  if needs_score_fallback(a_params, a_conds)
    || needs_score_fallback(b_params, b_conds)
  {
    return pattern_specificity_score(a_bt, a_heads, a_conds, a_body)
      < pattern_specificity_score(b_bt, b_heads, b_conds, b_body);
  }
  let (a_pos, a_guards) = rule_positions_and_guards(
    a_params, a_heads, a_bt, a_conds, a_defaults, a_body,
  );
  let (b_pos, b_guards) = rule_positions_and_guards(
    b_params, b_heads, b_bt, b_conds, b_defaults, b_body,
  );
  // Optional (defaulted) trailing positions let a rule match a range of
  // arities: `[required, total]`. `a`'s match set is a subset of `b`'s only if
  // every arity `a` accepts, `b` also accepts — i.e. `b` requires no more than
  // `a` (`b_req <= a_req`) and tolerates at least as many (`a_tot <= b_tot`).
  // A bare `f[x_]` (1..1) is then strictly more specific than `f[x_, y_:0]`
  // (1..2), so it is tried first — matching Wolfram.
  let a_req = a_pos.iter().filter(|(_, _, opt)| !opt).count();
  let b_req = b_pos.iter().filter(|(_, _, opt)| !opt).count();
  let (a_tot, b_tot) = (a_pos.len(), b_pos.len());
  if b_req > a_req || a_tot > b_tot {
    return false;
  }
  // `b` matches a strictly wider arity range ⇒ `a` is strictly more specific.
  let mut strictly_tighter = b_req < a_req || a_tot < b_tot;
  for ((ah, abt, _), (bh, bbt, _)) in a_pos.iter().zip(b_pos.iter()) {
    // Head: if `b` constrains the head, `a` must constrain it identically;
    // otherwise `a` could match a head `b` rejects → not a subset.
    match (ah, bh) {
      (_, None) => {
        if ah.is_some() {
          strictly_tighter = true;
        }
      }
      (Some(a), Some(b)) if a == b => {}
      _ => return false,
    }
    // Blank type: a looser blank (larger type) on `a` breaks the subset.
    if abt > bbt {
      return false;
    }
    if abt < bbt {
      strictly_tighter = true;
    }
  }
  // Every guard of `b` must also guard `a`, else `a` accepts inputs `b` rejects.
  // Compare guards by their string form (Expr has no PartialEq).
  let a_guard_strs: Vec<String> =
    a_guards.iter().map(crate::syntax::expr_to_string).collect();
  let b_guard_strs: Vec<String> =
    b_guards.iter().map(crate::syntax::expr_to_string).collect();
  for g in &b_guard_strs {
    if !a_guard_strs.contains(g) {
      return false;
    }
  }
  let a_has_extra_guard =
    a_guard_strs.iter().any(|g| !b_guard_strs.contains(g));
  // Strict subset only: identical rules don't dominate, so order is preserved.
  strictly_tighter || a_has_extra_guard
}

/// Count the meaningful constraint clauses inside a stored condition, recursing
/// through `And[…]`. The synthetic `Length[_lp] === N` length check emitted for
/// list patterns is structural, not a specificity constraint, so it is
/// excluded; everything else (element `MatchQ` guards, `/;` guard expressions,
/// `PatternTest` checks) counts once.
fn count_specificity_clauses(c: &Expr) -> u32 {
  match c {
    Expr::FunctionCall { name, args } if name == "And" => {
      args.iter().map(count_specificity_clauses).sum()
    }
    // A list-element `MatchQ[Part[_lp, i], pat]` guard only adds specificity
    // when `pat` actually constrains (literal / head / nested); a bare-blank
    // guard is always True and is emitted purely to preserve the name.
    Expr::FunctionCall { name, args }
      if name == "MatchQ" && args.len() == 2 =>
    {
      u32::from(element_needs_match_check(&args[1]))
    }
    Expr::Comparison {
      operators,
      operands,
    } if operators
      .iter()
      .any(|op| matches!(op, crate::syntax::ComparisonOp::SameQ))
      && matches!(
        operands.first(),
        Some(Expr::FunctionCall { name, .. }) if name == "Length"
      ) =>
    {
      0
    }
    // A nested structural pattern (`f[g[x_]]`) is stored as
    // `__StructuralPattern__[param, patternAST]`. Score it by how constrained
    // the pattern AST is, so a tighter inner pattern (`g[x_Integer]`) outranks a
    // looser one (`g[x_]`). Never drop below 1, the historical flat count, so a
    // structural pattern stays more specific than a bare blank.
    Expr::FunctionCall { name, args }
      if name == "__StructuralPattern__" && args.len() == 2 =>
    {
      count_pattern_specificity(&args[1]).max(1)
    }
    _ => 1,
  }
}

/// Count the constraining features of a (structural) pattern AST: each fixed
/// head symbol, each head-typed / tested pattern node, and each literal atom
/// makes the pattern more specific. Used to rank nested structural patterns
/// (`g[x_]` vs `g[x_Integer]`) against each other.
fn count_pattern_specificity(pat: &Expr) -> u32 {
  match pat {
    // A bare blank constrains nothing; a head-typed blank constrains the head.
    Expr::Pattern { head, .. } | Expr::PatternOptional { head, .. } => {
      u32::from(head.is_some())
    }
    // `x_?test` carries a test (and possibly a head).
    Expr::PatternTest { head, .. } => 1 + u32::from(head.is_some()),
    // A fixed head symbol (`g[…]`) is itself a constraint; recurse into args.
    Expr::FunctionCall { args, .. } => {
      1 + args.iter().map(count_pattern_specificity).sum::<u32>()
    }
    Expr::BinaryOp { left, right, .. } => {
      1 + count_pattern_specificity(left) + count_pattern_specificity(right)
    }
    Expr::UnaryOp { operand, .. } => 1 + count_pattern_specificity(operand),
    Expr::List(items) => {
      1 + items.iter().map(count_pattern_specificity).sum::<u32>()
    }
    // A bare blank (`_`, `__`, `___`, or a pattern placeholder) constrains
    // nothing; a concrete literal or symbol is maximally specific at that slot.
    Expr::Identifier(s) => u32::from(!s.starts_with('_')),
    _ => 1,
  }
}

/// Collect all operands for an associative binary operator (Plus, Times, Alternatives),
/// flattening nested applications of the same operator.
fn collect_binary_children(
  expr: &Expr,
  target_op: &BinaryOperator,
) -> Vec<Expr> {
  match expr {
    Expr::BinaryOp { op, left, right } if op == target_op => {
      let mut parts = collect_binary_children(left, target_op);
      parts.extend(collect_binary_children(right, target_op));
      parts
    }
    _ => vec![expr.clone()],
  }
}

/// Collect all pattern variable names from an expression.
/// Returns tuples of (name, head, is_optional) for each Pattern/PatternOptional node found.
fn collect_pattern_vars(
  expr: &Expr,
) -> Vec<(String, Option<String>, bool, u8)> {
  let mut vars = Vec::new();
  collect_pattern_vars_inner(expr, &mut vars);
  vars
}

fn collect_pattern_vars_inner(
  expr: &Expr,
  vars: &mut Vec<(String, Option<String>, bool, u8)>,
) {
  match expr {
    Expr::Pattern {
      name,
      head,
      blank_type,
    } if !vars.iter().any(|(n, _, _, _)| n == name) => {
      vars.push((name.clone(), head.clone(), false, *blank_type));
    }
    Expr::PatternOptional {
      name,
      head,
      default,
    } => {
      if !vars.iter().any(|(n, _, _, _)| n == name) {
        vars.push((name.clone(), head.clone(), true, 1));
      }
      if let Some(d) = default {
        collect_pattern_vars_inner(d, vars);
      }
    }
    Expr::PatternTest {
      name,
      test,
      blank_type,
      ..
    } => {
      if !vars.iter().any(|(n, _, _, _)| n == name) {
        vars.push((name.clone(), None, false, *blank_type));
      }
      collect_pattern_vars_inner(test, vars);
    }
    Expr::BinaryOp { left, right, .. } => {
      collect_pattern_vars_inner(left, vars);
      collect_pattern_vars_inner(right, vars);
    }
    Expr::UnaryOp { operand, .. } => {
      collect_pattern_vars_inner(operand, vars);
    }
    Expr::FunctionCall { args, .. } => {
      for a in args {
        collect_pattern_vars_inner(a, vars);
      }
    }
    Expr::List(items) => {
      for item in items {
        collect_pattern_vars_inner(item, vars);
      }
    }
    _ => {}
  }
}

/// Replace pattern variables with placeholder identifiers in an expression.
/// Returns the substituted expression.
fn replace_patterns_with_placeholders(
  expr: &Expr,
  vars: &[(String, Option<String>, bool, u8)],
) -> Expr {
  match expr {
    Expr::Pattern { name, .. } | Expr::PatternOptional { name, .. } => {
      if vars.iter().any(|(n, _, _, _)| n == name) {
        Expr::Identifier(format!("__patvar{}__", name))
      } else {
        expr.clone()
      }
    }
    Expr::PatternTest { name, .. } => {
      if vars.iter().any(|(n, _, _, _)| n == name) {
        Expr::Identifier(format!("__patvar{}__", name))
      } else {
        expr.clone()
      }
    }
    Expr::BinaryOp { op, left, right } => Expr::BinaryOp {
      op: *op,
      left: Box::new(replace_patterns_with_placeholders(left, vars)),
      right: Box::new(replace_patterns_with_placeholders(right, vars)),
    },
    Expr::UnaryOp { op, operand } => Expr::UnaryOp {
      op: *op,
      operand: Box::new(replace_patterns_with_placeholders(operand, vars)),
    },
    Expr::FunctionCall { name, args } => Expr::FunctionCall {
      name: name.clone(),
      args: args
        .iter()
        .map(|a| replace_patterns_with_placeholders(a, vars))
        .collect(),
    },
    Expr::List(items) => Expr::List(
      items
        .iter()
        .map(|a| replace_patterns_with_placeholders(a, vars))
        .collect(),
    ),
    _ => expr.clone(),
  }
}

/// Replace placeholder identifiers back with Pattern or PatternOptional nodes.
fn replace_placeholders_with_patterns(
  expr: &Expr,
  vars: &[(String, Option<String>, bool, u8)],
) -> Expr {
  match expr {
    Expr::Identifier(name) => {
      if let Some(stripped) = name
        .strip_prefix("__patvar")
        .and_then(|s| s.strip_suffix("__"))
        && let Some((pat_name, head, is_optional, blank_type)) =
          vars.iter().find(|(n, _, _, _)| n == stripped)
      {
        if *is_optional {
          return Expr::PatternOptional {
            name: pat_name.clone(),
            head: head.clone(),
            default: None, // system-determined default
          };
        } else {
          return Expr::Pattern {
            name: pat_name.clone(),
            head: head.clone(),
            blank_type: *blank_type,
          };
        }
      }
      expr.clone()
    }
    Expr::BinaryOp { op, left, right } => Expr::BinaryOp {
      op: *op,
      left: Box::new(replace_placeholders_with_patterns(left, vars)),
      right: Box::new(replace_placeholders_with_patterns(right, vars)),
    },
    Expr::UnaryOp { op, operand } => Expr::UnaryOp {
      op: *op,
      operand: Box::new(replace_placeholders_with_patterns(operand, vars)),
    },
    Expr::FunctionCall { name, args } => Expr::FunctionCall {
      name: name.clone(),
      args: args
        .iter()
        .map(|a| replace_placeholders_with_patterns(a, vars))
        .collect(),
    },
    Expr::List(items) => Expr::List(
      items
        .iter()
        .map(|a| replace_placeholders_with_patterns(a, vars))
        .collect(),
    ),
    _ => expr.clone(),
  }
}

/// Normalize a structural pattern by evaluating it with placeholder variables.
/// E.g., `1/x_` (BinaryOp Divide) → `Power[x_, -1]` (canonical form).
fn normalize_structural_pattern(pattern: &Expr) -> Expr {
  let vars = collect_pattern_vars(pattern);
  if vars.is_empty() {
    return pattern.clone();
  }
  let with_placeholders = replace_patterns_with_placeholders(pattern, &vars);
  match evaluate_expr_to_expr(&with_placeholders) {
    Ok(evaluated) => {
      // Convert BinaryOp::Divide to canonical Times[..., Power[..., -1]] form
      // so that patterns match regardless of how the expression was written
      // (e.g., 1/(a*b) vs (a*b)^-1 should both match the same pattern).
      // This is done before replacing placeholders back, since the arithmetic
      // functions (power_two, times_ast) need plain symbols, not pattern nodes.
      let evaluated = canonicalize_divide_in_expr(&evaluated);
      let result = replace_placeholders_with_patterns(&evaluated, &vars);
      // For Orderless functions (Times, Plus), reorder top-level args so
      // PatternOptional args come last. This ensures non-optional patterns
      // match earlier canonical args (e.g., numbers before symbols),
      // following Wolfram's convention for Orderless matching.
      reorder_orderless_pattern_args(result)
    }
    Err(_) => pattern.clone(), // fallback to raw pattern
  }
}

/// Recursively convert BinaryOp::Divide to canonical Times[num, Power[den, -1]]
/// form. Uses power_two/times_ast so nested structures are properly distributed
/// (e.g., 1/(a*Sqrt[b]) → Times[Power[a,-1], Power[b,-1/2]]).
pub fn canonicalize_divide_in_expr(expr: &Expr) -> Expr {
  match expr {
    Expr::BinaryOp {
      op: BinaryOperator::Divide,
      left,
      right,
    } => {
      let left = canonicalize_divide_in_expr(left);
      let right = canonicalize_divide_in_expr(right);
      match crate::functions::math_ast::power_two(&right, &Expr::Integer(-1)) {
        Ok(den_inv) => {
          if matches!(left, Expr::Integer(1)) {
            den_inv
          } else {
            crate::functions::math_ast::times_ast(&[left.clone(), den_inv])
              .unwrap_or_else(|_| Expr::BinaryOp {
                op: BinaryOperator::Divide,
                left: Box::new(left),
                right: Box::new(right),
              })
          }
        }
        Err(_) => Expr::BinaryOp {
          op: BinaryOperator::Divide,
          left: Box::new(left),
          right: Box::new(right),
        },
      }
    }
    Expr::FunctionCall { name, args } => Expr::FunctionCall {
      name: name.clone(),
      args: args.iter().map(canonicalize_divide_in_expr).collect(),
    },
    Expr::BinaryOp { op, left, right } => Expr::BinaryOp {
      op: *op,
      left: Box::new(canonicalize_divide_in_expr(left)),
      right: Box::new(canonicalize_divide_in_expr(right)),
    },
    Expr::UnaryOp { op, operand } => Expr::UnaryOp {
      op: *op,
      operand: Box::new(canonicalize_divide_in_expr(operand)),
    },
    Expr::List(items) => {
      Expr::List(items.iter().map(canonicalize_divide_in_expr).collect())
    }
    other => other.clone(),
  }
}

/// For FunctionCall patterns of Orderless functions, move PatternOptional
/// args to the end so non-optional patterns get matched first.
fn reorder_orderless_pattern_args(pattern: Expr) -> Expr {
  if let Expr::FunctionCall { name, args } = &pattern {
    let is_orderless = crate::evaluator::listable::is_builtin_orderless(name);
    if is_orderless
      && args.len() >= 2
      && args
        .iter()
        .any(|a| matches!(a, Expr::PatternOptional { .. }))
    {
      let mut sorted_args = args.clone();
      sorted_args.sort_by_key(|a| {
        if matches!(a, Expr::PatternOptional { .. }) {
          1
        } else {
          0
        }
      });
      return Expr::FunctionCall {
        name: name.clone(),
        args: sorted_args,
      };
    }
  }
  pattern
}

/// Helper for Attributes[f] = value / Attributes[f] := value
/// Extracts attribute symbols from value, validates, and sets them on the symbol.
fn set_attributes_from_value(
  sym_name: &str,
  rhs_value: &Expr,
) -> Result<Expr, InterpreterError> {
  // Check if symbol is locked
  let is_locked = crate::FUNC_ATTRS.with(|m| {
    m.borrow()
      .get(sym_name)
      .is_some_and(|attrs| attrs.contains(&"Locked".to_string()))
  });
  if is_locked {
    crate::emit_message(&format!(
      "Attributes::locked: Symbol {} is locked.",
      sym_name
    ));
    return Ok(rhs_value.clone());
  }

  // Extract attribute names from the value
  let attr_exprs = match rhs_value {
    Expr::List(items) => items.clone(),
    Expr::Identifier(_) => vec![rhs_value.clone()].into(),
    _ => vec![rhs_value.clone()].into(),
  };

  let mut valid_attrs = Vec::new();
  let mut has_error = false;
  for attr_expr in &attr_exprs {
    if let Expr::Identifier(attr_name) = attr_expr {
      valid_attrs.push(attr_name.clone());
    } else {
      // Non-symbol attribute — emit warning
      let attr_str = expr_to_string(attr_expr);
      crate::emit_message(&format!(
        "Attributes::attnf: {} is not a known attribute.",
        attr_str
      ));
      has_error = true;
    }
  }

  if has_error {
    return Ok(Expr::Identifier("$Failed".to_string()));
  }

  // Replace all user-defined attributes for this symbol
  crate::FUNC_ATTRS.with(|m| {
    m.borrow_mut().insert(sym_name.to_string(), valid_attrs);
  });

  Ok(rhs_value.clone())
}

/// Helper for `DownValues[f] = rules` / `DownValues[f] := rules` (and the
/// SubValues variant): unpack each Rule/RuleDelayed in `rhs` (`HoldPattern`
/// wrappers are stripped) and replay it as an individual `lhs := rhs` so
/// the rules get installed in FUNC_DEFS via the regular SetDelayed path.
/// Iteration order is preserved (no specificity sorting), matching Wolfram.
fn set_downvalues_from_rules(rhs: &Expr) -> Result<Expr, InterpreterError> {
  let rules: Vec<Expr> = match rhs {
    Expr::List(items) => items.to_vec(),
    other => vec![other.clone()],
  };
  // Collect the head symbols touched by these rules and clear their existing
  // FUNC_DEFS so the new list replaces (rather than augments) old definitions.
  let mut touched_heads: Vec<String> = Vec::new();
  for rule in &rules {
    let (pat, _) = match rule {
      Expr::Rule {
        pattern,
        replacement,
      }
      | Expr::RuleDelayed {
        pattern,
        replacement,
      } => ((**pattern).clone(), (**replacement).clone()),
      _ => continue,
    };
    let unwrapped = match &pat {
      Expr::FunctionCall { name, args }
        if name == "HoldPattern" && args.len() == 1 =>
      {
        args[0].clone()
      }
      _ => pat.clone(),
    };
    if let Expr::FunctionCall { name, .. } = &unwrapped
      && !touched_heads.contains(name)
    {
      touched_heads.push(name.clone());
    }
  }
  for head in &touched_heads {
    crate::FUNC_DEFS.with(|m| {
      m.borrow_mut().remove(head);
    });
  }

  // Replay each rule in given order with specificity sorting disabled so
  // FUNC_DEFS preserves the caller's listing order.
  let prev = SUPPRESS_SPECIFICITY_SORT.with(|c| c.replace(true));
  let result = (|| -> Result<(), InterpreterError> {
    for rule in &rules {
      let (pat, body, is_delayed) = match rule {
        Expr::Rule {
          pattern,
          replacement,
        } => ((**pattern).clone(), (**replacement).clone(), false),
        Expr::RuleDelayed {
          pattern,
          replacement,
        } => ((**pattern).clone(), (**replacement).clone(), true),
        _ => continue,
      };
      let lhs = match &pat {
        Expr::FunctionCall { name, args }
          if name == "HoldPattern" && args.len() == 1 =>
        {
          args[0].clone()
        }
        _ => pat.clone(),
      };
      // Rule (`->`) on the RHS replays as Set (literal-match LHS); only
      // RuleDelayed (`:>`) replays as SetDelayed (pattern-match LHS).
      // This matters when the LHS contains plain identifiers that should
      // remain literal (e.g. `Default[g] -> 3` in `DefaultValues[g] = …`).
      if is_delayed {
        set_delayed_ast(&lhs, &body)?;
      } else {
        set_ast(&lhs, &body)?;
      }
    }
    Ok(())
  })();
  SUPPRESS_SPECIFICITY_SORT.with(|c| c.set(prev));
  result?;
  Ok(rhs.clone())
}

/// Helper for Options[f] = value — set options for symbol f
fn set_options_from_value(
  sym_name: &str,
  rhs_value: &Expr,
) -> Result<Expr, InterpreterError> {
  // Extract rules from the value
  let rules = match rhs_value {
    Expr::List(items) => items.clone(),
    _ => vec![rhs_value.clone()].into(),
  };

  crate::FUNC_OPTIONS.with(|m| {
    m.borrow_mut().insert(sym_name.to_string(), rules.to_vec());
  });

  Ok(rhs_value.clone())
}

/// Returns true if `expr` mentions the symbol `name` anywhere in its tree.
/// Used by the `var = var <> rhs` fast path to refuse in-place mutation
/// when the right-hand side reads the variable being assigned (which
/// would otherwise observe the half-mutated state).
fn expr_references_identifier(expr: &Expr, name: &str) -> bool {
  match expr {
    Expr::Identifier(s) | Expr::Constant(s) => s == name,
    Expr::FunctionCall { args, .. } => {
      args.iter().any(|a| expr_references_identifier(a, name))
    }
    Expr::BinaryOp { left, right, .. } => {
      expr_references_identifier(left, name)
        || expr_references_identifier(right, name)
    }
    Expr::UnaryOp { operand, .. } => expr_references_identifier(operand, name),
    Expr::List(items) => {
      items.iter().any(|a| expr_references_identifier(a, name))
    }
    Expr::CurriedCall { func, args } => {
      expr_references_identifier(func, name)
        || args.iter().any(|a| expr_references_identifier(a, name))
    }
    Expr::Part { expr, index } => {
      expr_references_identifier(expr, name)
        || expr_references_identifier(index, name)
    }
    _ => false,
  }
}

/// AST-based Set implementation to handle Part assignment on associations and lists
/// `Symbol["x"]` on the left-hand side of an assignment denotes the symbol
/// `x` itself, so rewrite it to a plain `Identifier` before storing. This
/// lets `Set[Symbol["x"], v]` (and `Symbol["x"] = v`) bind the same `x` that a
/// later bare `x` lookup resolves, matching wolframscript.
fn normalize_symbol_lhs(lhs: &Expr) -> Expr {
  if let Expr::FunctionCall { name, args } = lhs
    && name == "Symbol"
    && args.len() == 1
    && let Expr::String(s) = &args[0]
  {
    return Expr::Identifier(s.clone());
  }
  lhs.clone()
}

pub fn set_ast(lhs: &Expr, rhs: &Expr) -> Result<Expr, InterpreterError> {
  let lhs = &normalize_symbol_lhs(lhs);
  // Unwrap Condition on LHS: f[x_] /; test = body is parsed as
  // Set[Condition[f[x_], test], body]. Extract the pattern and condition.
  let (lhs, _lhs_condition) = if let Expr::FunctionCall { name, args } = lhs
    && name == "Condition"
    && args.len() == 2
  {
    (&args[0], Some(&args[1]))
  } else {
    (lhs, None)
  };

  // Handle Entity property mutation: Entity["type", "name"]["property"] = value
  if let Expr::CurriedCall { func, args } = lhs
    && let Expr::FunctionCall {
      name,
      args: entity_args,
    } = func.as_ref()
    && name == "Entity"
    && entity_args.len() == 2
  {
    return crate::functions::entity_ast::entity_property_set(
      entity_args,
      args,
      rhs,
    );
  }

  // Single-bracket Part assignment on an existing Association:
  //   a[k] = v   when `a` is bound to an Association
  // Wolfram allows this as an alias for `a[[k]] = v` (mutate-by-key).
  // We match only when `a` already holds an Association so that
  // SubValue / DownValue definitions like `f[k_] := ...` still work.
  if let Expr::FunctionCall {
    name: head_name,
    args: head_args,
  } = lhs
    && head_args.len() == 1
  {
    let is_assoc = crate::ENV.with(|e| {
      let env = e.borrow();
      matches!(env.get(head_name), Some(StoredValue::Association(_)))
    });
    if is_assoc {
      let key_expr = evaluate_expr_to_expr(&head_args[0])?;
      let rhs_value = evaluate_expr_to_expr(rhs)?;
      let key = expr_to_string(&key_expr);
      crate::ENV.with(|e| {
        let mut env = e.borrow_mut();
        if let Some(StoredValue::Association(pairs)) = env.get_mut(head_name) {
          if let Some(pair) = pairs.iter_mut().find(|(k, _)| k == &key) {
            pair.1 = expr_to_string(&rhs_value);
          } else {
            pairs.push((key, expr_to_string(&rhs_value)));
          }
        }
      });
      return Ok(rhs_value);
    }
  }

  // Handle Part assignment: var[[indices]] = value
  if let Expr::Part { .. } = lhs {
    // Flatten nested Part to get base variable and list of indices
    let mut indices = Vec::new();
    let mut current = lhs;
    while let Expr::Part { expr, index } = current {
      indices.push(index.as_ref().clone());
      current = expr.as_ref();
    }
    indices.reverse();

    let var_name = match current {
      Expr::Identifier(name) => name.clone(),
      _ => {
        return Err(InterpreterError::EvaluationError(
          "Part assignment requires a variable name".into(),
        ));
      }
    };

    // Check Protected attribute
    if is_symbol_protected(&var_name) {
      let rhs_value = evaluate_expr_to_expr(rhs)?;
      crate::emit_message(&format!(
        "Part::wrsym: Symbol {} is Protected.",
        var_name
      ));
      return Ok(rhs_value);
    }

    // Evaluate indices
    let mut eval_indices = Vec::new();
    for idx in &indices {
      eval_indices.push(evaluate_expr_to_expr(idx)?);
    }

    // Evaluate the RHS
    let rhs_value = evaluate_expr_to_expr(rhs)?;

    // Part assignment into a variable bound to an Association (stored in the
    // string-keyed `StoredValue::Association` form). Convert it to an
    // `Expr::Association`, apply the assignment via `set_part_deep` — which
    // handles integer positions, keys, and deeply-nested targets uniformly —
    // then store the result back. This covers `myHash[["key"]] = v`,
    // `myHash[[1]] = v` (n-th value by position), and deep paths such as
    // `a[["x", "n"]] = v`.
    let is_assoc = crate::ENV.with(|e| {
      let env = e.borrow();
      matches!(env.get(&var_name), Some(StoredValue::Association(_)))
    });
    if is_assoc {
      let mut assoc_expr = crate::ENV.with(|e| {
        let env = e.borrow();
        match env.get(&var_name) {
          Some(StoredValue::Association(items)) => {
            let pairs: Vec<(Expr, Expr)> = items
              .iter()
              .map(|(k, v)| {
                (
                  string_to_expr(k).unwrap_or_else(|_| Expr::String(k.clone())),
                  string_to_expr(v).unwrap_or_else(|_| Expr::String(v.clone())),
                )
              })
              .collect();
            Expr::Association(pairs)
          }
          _ => unreachable!("is_assoc guarantees a stored Association"),
        }
      });
      set_part_deep(&mut assoc_expr, &eval_indices, &rhs_value)?;
      let new_value = match &assoc_expr {
        Expr::Association(items) => StoredValue::Association(
          items
            .iter()
            .map(|(k, v)| (expr_to_string(k), expr_to_string(v)))
            .collect(),
        ),
        // The whole association was replaced by a non-association value.
        other => StoredValue::ExprVal(other.clone()),
      };
      crate::ENV.with(|e| e.borrow_mut().insert(var_name.clone(), new_value));
      return Ok(rhs_value);
    }

    // General Part assignment: modify in-place if ExprVal, otherwise parse/modify/store
    let modified_in_place = crate::ENV.with(|e| {
      let mut env = e.borrow_mut();
      if let Some(StoredValue::ExprVal(expr)) = env.get_mut(&var_name) {
        // Modify directly in place — no clone needed
        set_part_deep(expr, &eval_indices, &rhs_value)
      } else {
        Err(InterpreterError::EvaluationError("not ExprVal".into()))
      }
    });
    if modified_in_place.is_ok() {
      return Ok(rhs_value);
    }

    // Fallback: parse stored string, modify, store back as ExprVal
    let stored_str = crate::ENV.with(|e| {
      let env = e.borrow();
      match env.get(&var_name) {
        Some(StoredValue::Raw(s)) => Some(s.clone()),
        _ => None,
      }
    });
    if let Some(stored_str) = stored_str {
      let mut stored_expr =
        string_to_expr(&stored_str).unwrap_or(Expr::Raw(stored_str));
      set_part_deep(&mut stored_expr, &eval_indices, &rhs_value)?;
      crate::ENV.with(|e| {
        e.borrow_mut()
          .insert(var_name, StoredValue::ExprVal(stored_expr))
      });
      return Ok(rhs_value);
    }

    // Matching wolframscript: if the target has no value at all, emit a
    // 'Set::noval' warning and return the RHS (the assignment is a no-op).
    crate::emit_message(&format!(
      "Set::noval: Symbol {} in part assignment does not have an immediate value.",
      var_name
    ));
    return Ok(rhs_value);
  }

  // Handle simple identifier assignment: x = value
  // Constants like Pi/E parse as `Expr::Constant`, but at the top level
  // they should still be assignable (subject to the Protected check),
  // matching wolframscript's `Pi = 4` → emits Set::wrsym and returns 4.
  if let Expr::Identifier(var_name) | Expr::Constant(var_name) = lhs {
    // Fast path for `var = var <> rhs` when the RHS evaluates to a string
    // and `var` already holds a string. Mutates the stored String in
    // place so a tight `Do[s = s <> c, …]` accumulator stays linear in
    // total work instead of paying an O(N) copy on every iteration.
    if let Expr::BinaryOp {
      op: BinaryOperator::StringJoin,
      left,
      right,
    } = rhs
      && let Expr::Identifier(left_name) = left.as_ref()
      && left_name == var_name
      && !expr_references_identifier(right, var_name)
    {
      let evaluated_right = evaluate_expr_to_expr(right)?;
      if let Expr::String(rhs_str) = &evaluated_right {
        let mutated = ENV.with(|e| -> Option<Expr> {
          let mut env = e.borrow_mut();
          if let Some(StoredValue::ExprVal(Expr::String(current))) =
            env.get_mut(var_name)
          {
            current.push_str(rhs_str);
            Some(Expr::String(current.clone()))
          } else {
            None
          }
        });
        if let Some(result) = mutated {
          return Ok(result);
        }
        // Variable wasn't yet bound to a String — fall through to the
        // generic path so the right-hand side is computed normally and
        // assigned (e.g. first-time assignment from "" <> "x").
      }
    }

    let rhs_value = evaluate_expr_to_expr(rhs)?;

    // Check Protected attribute
    if is_symbol_protected(var_name) {
      crate::emit_message(&format!(
        "Set::wrsym: Symbol {} is Protected.",
        var_name
      ));
      return Ok(rhs_value);
    }

    // Check if RHS is an association
    if let Expr::Association(items) = &rhs_value {
      let pairs: Vec<(String, String)> = items
        .iter()
        .map(|(k, v)| {
          // Use expr_to_string for keys to preserve type info
          // (e.g. Expr::String("x") → "\"x\"", Expr::Identifier("x") → "x")
          (expr_to_string(k), expr_to_string(v))
        })
        .collect();
      ENV.with(|e| {
        e.borrow_mut()
          .insert(var_name.clone(), StoredValue::Association(pairs))
      });
    } else if matches!(
      &rhs_value,
      Expr::List(_)
        | Expr::FunctionCall { .. }
        | Expr::String(_)
        | Expr::Function { .. }
        | Expr::NamedFunction { .. }
        | Expr::Image { .. }
        | Expr::Graphics { .. }
        | Expr::Rule { .. }
        | Expr::RuleDelayed { .. }
    ) {
      // Store lists, function calls, functions, strings, and rules as
      // ExprVal for faithful roundtrip. Rules in particular must not be
      // stored as Raw — `(p /; c) :> b` would reparse without the parens
      // as `Condition[p, RuleDelayed[c, b]]` because `:>` binds tighter
      // than `/;`.
      ENV.with(|e| {
        e.borrow_mut()
          .insert(var_name.clone(), StoredValue::ExprVal(rhs_value.clone()))
      });
    } else {
      ENV.with(|e| {
        e.borrow_mut().insert(
          var_name.clone(),
          StoredValue::Raw(expr_to_string(&rhs_value)),
        )
      });
    }

    return Ok(rhs_value);
  }

  // Handle Options[f] = value — set options on symbol f
  if let Expr::FunctionCall {
    name: func_name,
    args: lhs_args,
  } = lhs
    && func_name == "Options"
    && lhs_args.len() == 1
    && let Expr::Identifier(sym_name) = &lhs_args[0]
  {
    let rhs_value = evaluate_expr_to_expr(rhs)?;
    let result = set_options_from_value(sym_name, &rhs_value);
    // Plain `=` clears any prior `:=` marker so Definition emits the
    // matching operator on subsequent reads.
    crate::FUNC_OPTIONS_DELAYED.with(|m| {
      m.borrow_mut().remove(sym_name);
    });
    return result;
  }

  // Handle `N[sym, …] = value` (and `N[sym] = value`): store under
  // the symbol's NValues with the canonical LHS Wolfram reports.
  // `N[sym] = v` uses `N[sym, {MachinePrecision, MachinePrecision}]`,
  // `N[sym, p] = v` (numeric p) uses `N[sym, {p., Infinity}]`. Falls
  // through if sym isn't a bare Identifier.
  if let Expr::FunctionCall {
    name: func_name,
    args: lhs_args,
  } = lhs
    && func_name == "N"
    && (lhs_args.len() == 1 || lhs_args.len() == 2)
    && let Expr::Identifier(sym_name) = &lhs_args[0]
  {
    let rhs_value = evaluate_expr_to_expr(rhs)?;
    let prec_part = if lhs_args.len() == 2 {
      let prec_eval = evaluate_expr_to_expr(&lhs_args[1])?;
      let as_real = match &prec_eval {
        Expr::Integer(n) => Some(Expr::Real(*n as f64)),
        Expr::Real(_) => Some(prec_eval.clone()),
        _ => None,
      };
      if let Some(prec_real) = as_real {
        Expr::List(
          vec![prec_real, Expr::Identifier("Infinity".to_string())].into(),
        )
      } else {
        // Non-numeric precision (`N[a, p_?test] := …` style) — keep
        // the user's spec verbatim, mirroring Wolfram's HoldPattern
        // round-trip.
        prec_eval
      }
    } else {
      Expr::List(
        vec![
          Expr::Identifier("MachinePrecision".to_string()),
          Expr::Identifier("MachinePrecision".to_string()),
        ]
        .into(),
      )
    };
    let canonical_lhs = Expr::FunctionCall {
      name: "N".to_string(),
      args: vec![Expr::Identifier(sym_name.clone()), prec_part].into(),
    };
    N_VALUES.with(|m| {
      let mut map = m.borrow_mut();
      let entries = map.entry(sym_name.clone()).or_default();
      entries.retain(|(lhs_p, _)| {
        !crate::evaluator::pattern_matching::expr_equal(lhs_p, &canonical_lhs)
      });
      entries.push((canonical_lhs, rhs_value.clone()));
    });
    return Ok(rhs_value);
  }

  // Handle Attributes[f] = value — set attributes on symbol f
  if let Expr::FunctionCall {
    name: func_name,
    args: lhs_args,
  } = lhs
    && func_name == "Attributes"
    && lhs_args.len() == 1
    && let Expr::Identifier(sym_name) = &lhs_args[0]
  {
    let rhs_value = evaluate_expr_to_expr(rhs)?;
    return set_attributes_from_value(sym_name, &rhs_value);
  }

  // Handle DownValues[sym] = rules / SubValues[sym] = rules: replay each
  // rule as an individual `lhs := rhs` so the global FUNC_DEFS table picks
  // up the function definitions used during dispatch.
  if let Expr::FunctionCall {
    name: func_name,
    args: lhs_args,
  } = lhs
    && (func_name == "DownValues" || func_name == "SubValues")
    && lhs_args.len() == 1
    && matches!(&lhs_args[0], Expr::Identifier(_))
  {
    let rhs_value = evaluate_expr_to_expr(rhs)?;
    return set_downvalues_from_rules(&rhs_value);
  }

  // Handle DefaultValues[sym] = rules — same machinery as DownValues but
  // the rules' LHS is `Default[sym, ...]`. set_downvalues_from_rules
  // already replays each Rule/RuleDelayed via set_delayed_ast, which
  // routes Default[sym, n] = v through the regular Default DownValue
  // path so OptionValue / Default[sym] lookups find it.
  if let Expr::FunctionCall {
    name: func_name,
    args: lhs_args,
  } = lhs
    && func_name == "DefaultValues"
    && lhs_args.len() == 1
    && matches!(&lhs_args[0], Expr::Identifier(_))
  {
    let rhs_value = evaluate_expr_to_expr(rhs)?;
    return set_downvalues_from_rules(&rhs_value);
  }

  // Handle Messages[sym] = rules — replace all message DownValues for
  // `sym` (entries on `MessageName` whose first slot SameQ-matches sym)
  // with the rules in the RHS list. Each rule like `sym::tag :> "text"`
  // (i.e. `RuleDelayed[MessageName[sym, tag], "text"]`) is replayed via
  // SetDelayed so it lands as a regular MessageName DownValue.
  if let Expr::FunctionCall {
    name: func_name,
    args: lhs_args,
  } = lhs
    && func_name == "Messages"
    && lhs_args.len() == 1
    && let Expr::Identifier(sym_name) = &lhs_args[0]
  {
    let rhs_value = evaluate_expr_to_expr(rhs)?;
    // Filter MessageName DownValues, keeping only entries whose slot-0
    // SameQ literal does NOT equal `sym_name`.
    crate::FUNC_DEFS.with(|m| {
      let mut map = m.borrow_mut();
      if let Some(defs) = map.get_mut("MessageName") {
        defs.retain(|(params, conds, _defaults, _heads, _blanks, _body)| {
          let slot0_literal = conds.iter().find_map(|c| {
            if let Some(Expr::Comparison {
              operands,
              operators,
            }) = c
              && operators.len() == 1
              && matches!(operators[0], crate::syntax::ComparisonOp::SameQ)
              && operands.len() == 2
              && let Expr::Identifier(name) = &operands[0]
              && !params.is_empty()
              && name == &params[0]
            {
              Some(operands[1].clone())
            } else {
              None
            }
          });
          !matches!(&slot0_literal,
            Some(Expr::Identifier(s)) if s == sym_name)
        });
      }
    });
    // Replay each rule as an individual SetDelayed/Set so the rule LHS
    // re-installs as a MessageName DownValue.
    let rule_items: Vec<Expr> = match &rhs_value {
      Expr::List(items) => items.to_vec(),
      _ => vec![],
    };
    for item in &rule_items {
      let (pat, body, is_delayed) = match item {
        Expr::Rule {
          pattern,
          replacement,
        } => ((**pattern).clone(), (**replacement).clone(), false),
        Expr::RuleDelayed {
          pattern,
          replacement,
        } => ((**pattern).clone(), (**replacement).clone(), true),
        _ => continue,
      };
      let unwrapped = match &pat {
        Expr::FunctionCall { name: pn, args: pa }
          if pn == "HoldPattern" && pa.len() == 1 =>
        {
          pa[0].clone()
        }
        _ => pat.clone(),
      };
      if is_delayed {
        set_delayed_ast(&unwrapped, &body)?;
      } else {
        set_ast(&unwrapped, &body)?;
      }
    }
    return Ok(rhs_value);
  }

  // Handle NValues[sym] = rules / NValues[sym] := rules — store
  // the rule list directly into N_VALUES under sym, stripping any
  // outer HoldPattern wrappers so the LHS round-trips. Each rule's
  // pattern is kept verbatim (Wolfram does the same: arbitrary
  // user-written LHS patterns flow through `NValues[sym] = …`).
  if let Expr::FunctionCall {
    name: func_name,
    args: lhs_args,
  } = lhs
    && func_name == "NValues"
    && lhs_args.len() == 1
    && let Expr::Identifier(sym_name) = &lhs_args[0]
  {
    let wrapped_rhs = wrap_rule_lhs_in_holdpattern(rhs);
    let rhs_value = evaluate_expr_to_expr(&wrapped_rhs)?;
    let rule_items: Vec<Expr> = match &rhs_value {
      Expr::List(items) => items.to_vec(),
      _ => vec![],
    };
    let mut entries: Vec<(Expr, Expr)> = Vec::new();
    for item in &rule_items {
      let (pat, rep) = match item {
        Expr::Rule {
          pattern,
          replacement,
        }
        | Expr::RuleDelayed {
          pattern,
          replacement,
        } => ((**pattern).clone(), (**replacement).clone()),
        _ => continue,
      };
      let unwrapped_pat = match &pat {
        Expr::FunctionCall { name: pn, args: pa }
          if pn == "HoldPattern" && pa.len() == 1 =>
        {
          pa[0].clone()
        }
        _ => pat.clone(),
      };
      entries.push((unwrapped_pat, rep));
    }
    N_VALUES.with(|m| {
      let mut map = m.borrow_mut();
      map.insert(sym_name.clone(), entries);
    });
    return Ok(rhs_value);
  }

  // Handle DownValues: f[val1, val2, ...] = rhs
  // Store as a function definition with literal-match conditions
  if let Expr::FunctionCall {
    name: func_name,
    args: lhs_args,
  } = lhs
  {
    // Resolve Module-scoped unique symbols (e.g. f → f$1)
    let func_name = &resolve_func_name(func_name);
    let rhs_value = evaluate_expr_to_expr(rhs)?;

    // Check user-defined Protected attribute for DownValues
    // (builtin Protected is not checked for DownValues, matching wolframscript behavior)
    let is_user_protected = crate::FUNC_ATTRS.with(|m| {
      m.borrow()
        .get(func_name.as_str())
        .is_some_and(|attrs| attrs.contains(&"Protected".to_string()))
    });
    if is_user_protected {
      crate::emit_message(&format!(
        "Set::wrsym: Symbol {} is Protected.",
        func_name
      ));
      return Ok(rhs_value);
    }

    // Build param names and conditions for each argument
    let mut params = Vec::new();
    let mut conditions: Vec<Option<Expr>> = Vec::new();
    let mut defaults = Vec::new();
    let mut heads = Vec::new();
    let mut blank_types: Vec<u8> = Vec::new();

    for (i, arg) in lhs_args.iter().enumerate() {
      let param_name = format!("_dv{}", i);

      // Check if arg is a pattern (Blank, BlankSequence, named pattern, etc.)
      let is_pattern = match arg {
        Expr::Pattern { .. }
        | Expr::PatternOptional { .. }
        | Expr::PatternTest { .. } => true,
        Expr::Identifier(name) => name.contains('_'),
        _ => crate::evaluator::pattern_matching::contains_pattern(arg),
      };

      if is_pattern {
        let (pat_name, head, blank_type) = extract_pattern_info(arg);
        let final_name = if pat_name.is_empty() {
          param_name
        } else {
          pat_name
        };
        params.push(final_name);
        conditions.push(None);
        heads.push(head);
        blank_types.push(blank_type);
      } else {
        // Evaluate the literal argument value
        let eval_arg = evaluate_expr_to_expr(arg)?;
        // Condition: _dvN === eval_arg (using SameQ for exact matching)
        conditions.push(Some(Expr::Comparison {
          operands: vec![Expr::Identifier(param_name.clone()), eval_arg],
          operators: vec![crate::syntax::ComparisonOp::SameQ],
        }));
        params.push(param_name);
        heads.push(None);
        blank_types.push(1);
      }
      defaults.push(None);
    }

    // Check if all args are literal (non-pattern) — if so, insert at beginning
    // for priority over general patterns (matching Mathematica specificity ordering)
    let has_literal_conditions =
      conditions.iter().flatten().any(condition_is_literal_arg);

    // Pure literal-argument definition `f[lit, …] = value` (every argument is
    // a literal matched by SameQ, no pattern blanks) — the memoization idiom
    // `f[x_] := f[x] = …`. Store it in the O(1) MEMO_VALUES cache rather than
    // accumulating thousands of FUNC_DEFS overloads (which made memoized
    // recursion O(n²) under linear dispatch). DownValues merges these back.
    // Only divert genuine user-defined functions to the memoization cache.
    // Builtin heads (NumericQ[a]=True, MessageName/`f::usage`, Format, Default,
    // Options, N, …) keep their literal definitions in FUNC_DEFS, where the
    // dedicated builtin lookups read them directly; routing those aside would
    // break NumericQ value declarations, usage messages, etc. The memoization
    // idiom (`f[x_] := f[x] = …`) only ever applies to user functions.
    let is_builtin_head =
      !crate::evaluator::attributes::get_builtin_attributes(func_name.as_str())
        .is_empty();
    let all_literal_sameq = !is_builtin_head
      && !conditions.is_empty()
      && conditions.iter().all(|c| {
        matches!(c, Some(Expr::Comparison { operators, .. })
          if operators.len() == 1
            && operators[0] == crate::syntax::ComparisonOp::SameQ)
      });
    if all_literal_sameq {
      let mut arg_exprs = Vec::with_capacity(conditions.len());
      for c in &conditions {
        if let Some(Expr::Comparison { operands, .. }) = c
          && operands.len() == 2
        {
          arg_exprs.push(operands[1].clone());
        }
      }
      if arg_exprs.len() == conditions.len() {
        let key = arg_exprs
          .iter()
          .map(crate::syntax::expr_to_string)
          .collect::<Vec<_>>()
          .join("\u{1}");
        crate::MEMO_VALUES.with(|m| {
          m.borrow_mut()
            .entry(func_name.clone())
            .or_default()
            .insert(key, (arg_exprs, rhs_value.clone()));
        });
        return Ok(rhs_value);
      }
    }

    crate::FUNC_DEFS.with(|m| {
      let mut defs = m.borrow_mut();
      let entry = defs.entry(func_name.clone()).or_insert_with(Vec::new);
      if has_literal_conditions {
        // Literal-match definitions go before pattern definitions but after
        // existing literal definitions (preserving definition order).
        // Re-defining the same literal pattern (e.g. NumericQ[a] = True
        // followed by NumericQ[a] = False) replaces the prior definition
        // rather than shadowing it — match Wolfram's "second Set wins"
        // semantics. Compare on params + conditions structurally.
        let cond_strs: Vec<Option<String>> = conditions
          .iter()
          .map(|c| c.as_ref().map(crate::syntax::expr_to_string))
          .collect();
        entry.retain(|(p, c, _, _, _, _)| {
          if p != &params {
            return true;
          }
          let other_strs: Vec<Option<String>> = c
            .iter()
            .map(|c| c.as_ref().map(crate::syntax::expr_to_string))
            .collect();
          other_strs != cond_strs
        });
        let pos = entry
          .iter()
          .position(|(_, c, _, _, _, _)| {
            // Find the first non-literal (pattern) definition
            !c.iter().flatten().any(condition_is_literal_arg)
          })
          .unwrap_or(entry.len());
        entry.insert(
          pos,
          (
            params,
            conditions,
            defaults,
            heads,
            blank_types,
            rhs_value.clone(),
          ),
        );
      } else {
        // Insert by the rule partial order: place the new rule before the first
        // existing rule it strictly dominates (is more specific than). Rules it
        // does not dominate — including incomparable ones — keep definition
        // order, matching Wolfram.
        let pos = entry
          .iter()
          .position(|(p, c, d, h, bt, b)| {
            rule_dominates(
              &params,
              &heads,
              &blank_types,
              &conditions,
              &defaults,
              &rhs_value,
              p,
              h,
              bt,
              c,
              d,
              b,
            )
          })
          .unwrap_or(entry.len());
        entry.insert(
          pos,
          (
            params,
            conditions,
            defaults,
            heads,
            blank_types,
            rhs_value.clone(),
          ),
        );
      }
    });

    return Ok(rhs_value);
  }

  // Handle destructuring assignment: {a, b, c} = {1, 2, 3}
  if let Expr::List(lhs_items) = lhs {
    let rhs_value = evaluate_expr_to_expr(rhs)?;
    if let Expr::List(rhs_items) = &rhs_value {
      if lhs_items.len() == rhs_items.len() {
        for (l, r) in lhs_items.iter().zip(rhs_items.iter()) {
          set_ast(l, r)?;
        }
        // Re-evaluate the RHS in the post-assignment environment so
        // references to the freshly-bound variables resolve to their
        // new values. Wolfram returns the RHS-after-substitution:
        //   {a, b, {c, {d}}} = {1, 2, {{c1, c2}, {a}}}
        //     → {1, 2, {{c1, c2}, {1}}}
        // (the trailing `{a}` becomes `{1}` because `a` was just
        // bound to 1).
        return evaluate_expr_to_expr(&rhs_value);
      } else {
        crate::emit_message(&format!(
          "Set::shape: Lists {} and {} are not the same shape.",
          expr_to_string(lhs),
          expr_to_string(&rhs_value)
        ));
      }
    } else {
      crate::emit_message(&format!(
        "Set::shape: Lists {} and {} are not the same shape.",
        expr_to_string(lhs),
        expr_to_string(&rhs_value)
      ));
    }
    return Ok(rhs_value);
  }

  // SubValue form: `f[a][b] = rhs` (also deeper nestings). Mathematica
  // stores these under SubValues[f]; Woxi's `set_delayed_ast` already
  // accepts `:=` here and returns Null without actually installing a
  // subvalue rule. Mirror that for `=` so `f[a][b] = 3` doesn't error
  // out — matching wolframscript's surface behaviour.
  if let Expr::CurriedCall { func, .. } = lhs {
    let mut inner = func.as_ref();
    loop {
      match inner {
        Expr::CurriedCall { func: f2, .. } => inner = f2.as_ref(),
        Expr::FunctionCall { .. } => {
          let rhs_value = evaluate_expr_to_expr(rhs)?;
          return Ok(rhs_value);
        }
        _ => break,
      }
    }
  }

  Err(InterpreterError::EvaluationError(
    "First argument of Set must be an identifier, part extract, or function call".into(),
  ))
}

/// Handle SetDelayed[f[patterns...], body] — stores a function definition.
/// This handles cases that the PEG FunctionDefinition rule doesn't parse,
/// such as list-pattern arguments: f[{x_Integer, y_Integer}] := body.
pub fn set_delayed_ast(
  lhs: &Expr,
  body: &Expr,
) -> Result<Expr, InterpreterError> {
  let lhs = &normalize_symbol_lhs(lhs);
  // Early reject: `a + b := c` / `a * b := c` / `a^b := c` etc.
  // attempt to install DownValues on the corresponding built-in
  // (`Plus`, `Times`, `Power`, …) which are Protected.
  // wolframscript emits `SetDelayed::write: Tag <op> in <lhs> is
  // Protected.` and returns `$Failed`. Detect the BinaryOp head
  // here so the message and return value match.
  if let Expr::BinaryOp { op, .. } = lhs {
    use BinaryOperator as B;
    let tag = match op {
      B::Plus | B::Minus => Some("Plus"),
      B::Times | B::Divide => Some("Times"),
      B::Power => Some("Power"),
      _ => None,
    };
    if let Some(t) = tag
      && crate::evaluator::get_builtin_attributes(t).contains(&"Protected")
    {
      crate::emit_message(&format!(
        "SetDelayed::write: Tag {} in {} is Protected.",
        t,
        crate::syntax::expr_to_string(lhs)
      ));
      return Ok(Expr::Identifier("$Failed".to_string()));
    }
  }

  // Unwrap nested Conditions on LHS:
  //   f[x_] /; a /; b := body → SetDelayed[Condition[Condition[f[x_], a], b], body]
  // Collect each condition and keep unwrapping until the LHS is no longer a Condition.
  let mut lhs_stripped = lhs;
  let mut lhs_conditions: Vec<Expr> = Vec::new();
  while let Expr::FunctionCall { name, args } = lhs_stripped {
    if name == "Condition" && args.len() == 2 {
      lhs_conditions.push(args[1].clone());
      lhs_stripped = &args[0];
    } else {
      break;
    }
  }
  // The conditions appear in reverse order (outermost first), so reverse to
  // get the left-to-right source order.
  lhs_conditions.reverse();
  let lhs = lhs_stripped;

  // Unwrap nested Conditions on RHS:
  //   f[x_] := body /; a /; b → SetDelayed[f[x_], Condition[Condition[body, a], b]]
  let mut body_stripped = body;
  let mut rhs_conditions: Vec<Expr> = Vec::new();
  while let Expr::FunctionCall { name, args } = body_stripped {
    if name == "Condition" && args.len() == 2 {
      rhs_conditions.push(args[1].clone());
      body_stripped = &args[0];
    } else {
      break;
    }
  }
  rhs_conditions.reverse();
  let body = body_stripped;

  // Combine all conditions into a single And[..] expression.
  let mut all_conditions: Vec<Expr> = Vec::new();
  all_conditions.extend(lhs_conditions);
  all_conditions.extend(rhs_conditions);
  let combined = match all_conditions.len() {
    0 => None,
    1 => Some(all_conditions.into_iter().next().unwrap()),
    _ => Some(Expr::FunctionCall {
      name: "And".to_string(),
      args: all_conditions.into(),
    }),
  };
  let body_condition = combined.as_ref();

  // Handle Attributes[f] := value — set attributes on symbol f
  if let Expr::FunctionCall {
    name: func_name,
    args: lhs_args,
  } = lhs
    && func_name == "Attributes"
    && lhs_args.len() == 1
    && let Expr::Identifier(sym_name) = &lhs_args[0]
  {
    // SetDelayed still evaluates the RHS for Attributes
    let rhs_value = evaluate_expr_to_expr(body)?;
    let result = set_attributes_from_value(sym_name, &rhs_value)?;
    // On success, SetDelayed returns Null (no visible output) — matching
    // wolframscript. Error paths (`$Failed`) are preserved as-is.
    if matches!(&result, Expr::Identifier(s) if s == "$Failed") {
      return Ok(result);
    }
    return Ok(Expr::Identifier("Null".to_string()));
  }

  // Handle `NValues[sym] := rules` — same store as the Set form,
  // but SetDelayed returns Null at top level.
  if let Expr::FunctionCall {
    name: func_name,
    args: lhs_args,
  } = lhs
    && func_name == "NValues"
    && lhs_args.len() == 1
    && let Expr::Identifier(sym_name) = &lhs_args[0]
  {
    let wrapped_body = wrap_rule_lhs_in_holdpattern(body);
    let rhs_value = evaluate_expr_to_expr(&wrapped_body)?;
    let rule_items: Vec<Expr> = match &rhs_value {
      Expr::List(items) => items.to_vec(),
      _ => vec![],
    };
    let mut entries: Vec<(Expr, Expr)> = Vec::new();
    for item in &rule_items {
      let (pat, rep) = match item {
        Expr::Rule {
          pattern,
          replacement,
        }
        | Expr::RuleDelayed {
          pattern,
          replacement,
        } => ((**pattern).clone(), (**replacement).clone()),
        _ => continue,
      };
      let unwrapped_pat = match &pat {
        Expr::FunctionCall { name: pn, args: pa }
          if pn == "HoldPattern" && pa.len() == 1 =>
        {
          pa[0].clone()
        }
        _ => pat.clone(),
      };
      entries.push((unwrapped_pat, rep));
    }
    N_VALUES.with(|m| {
      let mut map = m.borrow_mut();
      map.insert(sym_name.clone(), entries);
    });
    return Ok(Expr::Identifier("Null".to_string()));
  }

  // `Format[expr, FORM] := …` (or its UpSet/UpSetDelayed cousins) registers
  // FORM in `$PrintForms`/`$OutputForms` and stores the rule under
  // `FormatValues[head]` where `head` is the head symbol of `expr`.
  // The 1-arg form `Format[expr] := …` stores under `FormatValues[head]`
  // with the empty form name `""`, which `box_subexpr_via_user_rules`
  // treats as "applies to every form".
  if let Expr::FunctionCall {
    name: func_name,
    args: lhs_args,
  } = lhs
    && func_name == "Format"
    && (lhs_args.len() == 1 || lhs_args.len() == 2)
  {
    let form_name = if lhs_args.len() == 2 {
      if let Expr::Identifier(name) = &lhs_args[1] {
        Some(name.clone())
      } else {
        None
      }
    } else {
      Some(String::new())
    };
    if let Some(form_name) = form_name {
      if !form_name.is_empty() {
        register_user_print_form(&form_name);
      }
      if let Some(head) = format_pattern_head(&lhs_args[0]) {
        FORMAT_VALUES.with(|m| {
          m.borrow_mut().entry(head).or_default().push((
            form_name,
            lhs_args[0].clone(),
            body.clone(),
          ));
        });
      }
      return Ok(Expr::Identifier("Null".to_string()));
    }
  }

  // Handle DownValues[sym] := rules / SubValues[sym] := rules: replay each
  // rule via set_delayed_ast so dispatch picks them up.
  if let Expr::FunctionCall {
    name: func_name,
    args: lhs_args,
  } = lhs
    && (func_name == "DownValues" || func_name == "SubValues")
    && lhs_args.len() == 1
    && matches!(&lhs_args[0], Expr::Identifier(_))
  {
    let rhs_value = evaluate_expr_to_expr(body)?;
    set_downvalues_from_rules(&rhs_value)?;
    return Ok(Expr::Identifier("Null".to_string()));
  }

  // Handle Options[f] := rules — same as Options[f] = rules (SetDelayed
  // still evaluates the RHS to extract the rule list, which doesn't depend
  // on f's later state). OptionValue then looks up via FUNC_OPTIONS.
  if let Expr::FunctionCall {
    name: func_name,
    args: lhs_args,
  } = lhs
    && func_name == "Options"
    && lhs_args.len() == 1
    && let Expr::Identifier(sym_name) = &lhs_args[0]
  {
    let rhs_value = evaluate_expr_to_expr(body)?;
    set_options_from_value(sym_name, &rhs_value)?;
    crate::FUNC_OPTIONS_DELAYED.with(|m| {
      m.borrow_mut().insert(sym_name.to_string());
    });
    return Ok(Expr::Identifier("Null".to_string()));
  }

  // Handle UpValues[sym] := rules: replay each rule as a TagSetDelayed on
  // `sym` so the upvalue dispatch (UPVALUES) picks them up. The tag is
  // the symbol named by `sym`; each rule's pattern becomes the LHS of
  // `sym /: lhs := rhs`.
  if let Expr::FunctionCall {
    name: func_name,
    args: lhs_args,
  } = lhs
    && func_name == "UpValues"
    && lhs_args.len() == 1
    && let Expr::Identifier(sym) = &lhs_args[0]
  {
    let rhs_value = evaluate_expr_to_expr(body)?;
    let rules: Vec<Expr> = match &rhs_value {
      Expr::List(items) => items.to_vec(),
      other => vec![other.clone()],
    };
    let tag = Expr::Identifier(sym.clone());
    for rule in &rules {
      let (pat, body) = match rule {
        Expr::Rule {
          pattern,
          replacement,
        }
        | Expr::RuleDelayed {
          pattern,
          replacement,
        } => ((**pattern).clone(), (**replacement).clone()),
        _ => continue,
      };
      let pattern_lhs = match &pat {
        Expr::FunctionCall { name, args }
          if name == "HoldPattern" && args.len() == 1 =>
        {
          args[0].clone()
        }
        _ => pat.clone(),
      };
      tag_set_delayed_ast(&tag, &pattern_lhs, &body, false)?;
    }
    return Ok(Expr::Identifier("Null".to_string()));
  }

  if let Expr::FunctionCall {
    name: func_name,
    args: lhs_args,
  } = lhs
  {
    // Resolve Module-scoped unique symbols (e.g. f → f$1)
    let func_name = &resolve_func_name(func_name);
    // Check user-defined Protected attribute for DownValues, and
    // also the built-in Protected attribute for system symbols
    // like `Sin`, `Plus`, etc. wolframscript emits
    // `SetDelayed::write` (not `::wrsym`) and returns `$Failed`
    // for protected built-ins. Special case: `N[sym, …] := body`
    // stores an NValue on `sym`, so allow even though `N` is
    // Protected.
    let is_user_protected = crate::FUNC_ATTRS.with(|m| {
      m.borrow()
        .get(func_name.as_str())
        .is_some_and(|attrs| attrs.contains(&"Protected".to_string()))
    });
    // Heads with a redirected per-symbol storage mechanism —
    // wolframscript permits these even though the head is
    // Protected (NValues, Messages, Format rules, etc.).
    let is_n_value_assignment = matches!(
      func_name.as_str(),
      "N" | "MessageName" | "Format" | "Default" | "Options"
    );
    let was_unprotected = crate::FUNC_ATTRS_REMOVED.with(|m| {
      m.borrow()
        .get(func_name.as_str())
        .is_some_and(|attrs| attrs.contains(&"Protected".to_string()))
    });
    let is_builtin_protected = !is_n_value_assignment
      && !was_unprotected
      && crate::evaluator::attributes::get_builtin_attributes(func_name)
        .contains(&"Protected");
    if is_user_protected || is_builtin_protected {
      let (msg_tag, ret) = if is_builtin_protected && !is_user_protected {
        ("SetDelayed::write", Expr::Identifier("$Failed".to_string()))
      } else {
        ("SetDelayed::wrsym", Expr::Identifier("Null".to_string()))
      };
      let lhs_str = crate::syntax::expr_to_string(lhs);
      crate::emit_message(&format!(
        "{}: Tag {} in {} is Protected.",
        msg_tag, func_name, lhs_str
      ));
      return Ok(ret);
    }

    let mut params = Vec::new();
    let mut conditions: Vec<Option<Expr>> = Vec::new();
    let mut defaults: Vec<Option<Expr>> = Vec::new();
    let mut heads: Vec<Option<String>> = Vec::new();
    let mut blank_types: Vec<u8> = Vec::new();
    // List-pattern destructuring substitutions: each leaf pattern variable maps
    // to the `Part[…]` / `Sequence@@Drop[…]` expression that extracts its value
    // from the bound list parameter. Nested list patterns contribute deeper
    // `Part` paths (see `collect_list_pattern_bindings`).
    let mut body_substitutions: Vec<(String, Expr)> = Vec::new();
    let mut inline_opts_defaults: Option<Vec<Expr>> = None;

    for (i, arg) in lhs_args.iter().enumerate() {
      match arg {
        // OptionsPattern[] or OptionsPattern[{defaults...}] — matches zero or more Rule arguments
        Expr::FunctionCall {
          name: fn_name,
          args: op_args,
        } if fn_name == "OptionsPattern" => {
          let param_name = format!("__opts{}", i);
          params.push(param_name);
          conditions.push(None);
          defaults.push(None);
          heads.push(None);
          blank_types.push(3); // BlankNullSequence - matches 0 or more args
          // Extract inline defaults from OptionsPattern[{a -> a0, ...}]
          if op_args.len() == 1
            && let Expr::List(rules) = &op_args[0]
          {
            inline_opts_defaults = Some(rules.to_vec());
          }
        }
        // List pattern: {x_Integer, y_Integer} — destructure a list argument.
        // Supports a trailing BlankSequence (`__`) or BlankNullSequence (`___`)
        // element, which relaxes the length check and (for named sequence
        // elements) binds the sequence variable to the matching tail.
        Expr::List(patterns) => {
          let param_name = format!("_lp{}", i);
          // The combined condition checks the list length and that each
          // constrained element (literal, head-constrained, or nested) matches
          // its sub-pattern; the bindings extract each leaf variable.
          let (combined_cond, bindings) =
            build_list_pattern_match(patterns, &param_name);
          conditions.push(Some(combined_cond));
          body_substitutions.extend(bindings);
          params.push(param_name);
          defaults.push(None);
          heads.push(Some("List".to_string()));
          blank_types.push(1);
        }
        // PatternTest: x_?test or x_Head?test — store as structural pattern
        // to preserve the test condition during dispatch.
        // Use the original PatternTest expression directly: round-tripping
        // it through normalize_structural_pattern drops the `?test` part
        // (collect_pattern_vars only records name/head/optional, not tests).
        Expr::PatternTest {
          name,
          head,
          blank_type,
          ..
        } => {
          let param_name = if name.is_empty() {
            format!("__sp{}", i)
          } else {
            name.clone()
          };
          conditions.push(Some(Expr::FunctionCall {
            name: "__StructuralPattern__".to_string(),
            args: vec![Expr::Identifier(param_name.clone()), arg.clone()]
              .into(),
          }));
          params.push(param_name);
          defaults.push(None);
          heads.push(head.clone());
          blank_types.push(*blank_type);
        }
        // `Pattern[name, body]` (long-form named pattern, e.g.
        // `Pattern[levelspec, _?LevelQ]` ≡ `levelspec_?LevelQ`). Re-attach
        // the name onto the inner pattern body so PatternTest/Blank handling
        // sees it as a named pattern rather than an anonymous structural one.
        Expr::FunctionCall {
          name: pat_fn,
          args: pat_args,
        } if pat_fn == "Pattern"
          && pat_args.len() == 2
          && matches!(&pat_args[0], Expr::Identifier(_)) =>
        {
          let pname = match &pat_args[0] {
            Expr::Identifier(s) => s.clone(),
            _ => unreachable!(),
          };
          match &pat_args[1] {
            Expr::PatternTest {
              head,
              blank_type,
              test,
              ..
            } => {
              let normalized = Expr::PatternTest {
                name: pname.clone(),
                head: head.clone(),
                blank_type: *blank_type,
                test: test.clone(),
              };
              conditions.push(Some(Expr::FunctionCall {
                name: "__StructuralPattern__".to_string(),
                args: vec![Expr::Identifier(pname.clone()), normalized].into(),
              }));
              params.push(pname);
              defaults.push(None);
              heads.push(head.clone());
              blank_types.push(*blank_type);
            }
            // `Pattern[name, OptionsPattern[…]]` — named OptionsPattern
            // (e.g. `opt:OptionsPattern[]`) matches zero or more Rule
            // arguments. Use the `__opts{i}` synthetic name so the
            // option-bindings collector recognises it; the user-visible
            // `name` is not currently bound (rare in practice — most
            // bodies reach options via `OptionValue[…]`, not the bound
            // sequence symbol).
            Expr::FunctionCall {
              name: opn,
              args: op_args,
            } if opn == "OptionsPattern" => {
              let _ = pname; // user-visible name not bound
              let param_name = format!("__opts{}", i);
              params.push(param_name);
              conditions.push(None);
              defaults.push(None);
              heads.push(None);
              blank_types.push(3); // BlankNullSequence — 0 or more
              if op_args.len() == 1
                && let Expr::List(rules) = &op_args[0]
              {
                inline_opts_defaults = Some(rules.to_vec());
              }
            }
            // `Pattern[name, {p1, p2, ...}]` — named list pattern.
            // Bind `name` to the entire list (via the param itself) AND
            // destructure inner elements.
            Expr::List(patterns) => {
              let (combined_cond, bindings) =
                build_list_pattern_match(patterns, &pname);
              conditions.push(Some(combined_cond));
              body_substitutions.extend(bindings);
              params.push(pname);
              defaults.push(None);
              heads.push(Some("List".to_string()));
              blank_types.push(1);
            }
            // `Pattern[name, _]` / `Pattern[name, _Head]` — plain named
            // blank, fast path.
            Expr::Identifier(n) if n.starts_with('_') => {
              let (_, head, blank_type) = extract_pattern_info(arg);
              params.push(pname);
              conditions.push(None);
              defaults.push(None);
              heads.push(head);
              blank_types.push(blank_type);
            }
            Expr::FunctionCall { name: bn, .. }
              if bn == "Blank"
                || bn == "BlankSequence"
                || bn == "BlankNullSequence" =>
            {
              let (_, head, blank_type) = extract_pattern_info(arg);
              params.push(pname);
              conditions.push(None);
              defaults.push(None);
              heads.push(head);
              blank_types.push(blank_type);
            }
            // Any other body (Alternatives, Except, nested patterns, …)
            // must keep its constraint: store the whole Pattern[name,
            // body] as a structural pattern matched at dispatch time —
            // previously the constraint was silently dropped, so
            // s[x : (_Integer | _Real)] matched any argument at all.
            _ => {
              conditions.push(Some(Expr::FunctionCall {
                name: "__StructuralPattern__".to_string(),
                args: vec![Expr::Identifier(pname.clone()), arg.clone()].into(),
              }));
              params.push(pname);
              defaults.push(None);
              heads.push(None);
              blank_types.push(1);
            }
          }
        }
        // Simple pattern: x_ or x_Head
        _ => {
          let (pat_name, head, blank_type) = extract_pattern_info(arg);
          // Check for anonymous pattern identifiers (_, __, ___)
          let is_anonymous_pattern = pat_name.is_empty()
            && head.is_none()
            && matches!(arg, Expr::Identifier(name) if name.starts_with('_'));
          if is_anonymous_pattern {
            let param_name = format!("_dv{}", i);
            params.push(param_name);
            conditions.push(None);
            blank_types.push(blank_type);
          } else if pat_name.is_empty() && head.is_none() {
            if crate::evaluator::pattern_matching::contains_pattern(arg) {
              // Structural pattern (e.g., 1/x_, a_ + b_) — normalize and store
              // the pattern AST in a __StructuralPattern__ marker for dispatch-time matching.
              let param_name = format!("__sp{}", i);
              let normalized = normalize_structural_pattern(arg);
              conditions.push(Some(Expr::FunctionCall {
                name: "__StructuralPattern__".to_string(),
                args: vec![Expr::Identifier(param_name.clone()), normalized]
                  .into(),
              }));
              params.push(param_name);
              blank_types.push(1);
            } else {
              // Literal value (not a pattern) — create a SameQ condition
              // e.g., f[1] := ... should only match when arg === 1
              let param_name = format!("_dv{}", i);
              let eval_arg = evaluate_expr_to_expr(arg)?;
              conditions.push(Some(Expr::Comparison {
                operands: vec![Expr::Identifier(param_name.clone()), eval_arg],
                operators: vec![crate::syntax::ComparisonOp::SameQ],
              }));
              params.push(param_name);
              blank_types.push(1);
            }
          } else {
            params.push(pat_name);
            conditions.push(None);
            blank_types.push(blank_type);
          }
          // Carry inline default through; if `arg` is `x_.` (PatternOptional
          // with no inline default), store `Default[func]` as a deferred
          // placeholder so dispatch can pull a user-set Default[f] at call
          // time (matching Wolfram's `f[x_.] := ...` + `Default[f] = c`).
          let default_for_slot = match arg {
            Expr::PatternOptional {
              default: Some(d), ..
            } => Some((**d).clone()),
            Expr::PatternOptional { default: None, .. } => {
              Some(Expr::FunctionCall {
                name: "Default".to_string(),
                args: vec![
                  Expr::Identifier(func_name.clone()),
                  Expr::Integer((i + 1) as i128),
                ]
                .into(),
              })
            }
            _ => None,
          };
          defaults.push(default_for_slot);
          heads.push(head);
        }
      }
    }

    // Apply all list-destructuring substitutions to an expression: each
    // element name is replaced with `Part[param, idx+1]`, except a named
    // trailing sequence pattern (`y___` or `y__`) which becomes
    // `Sequence@@Drop[param, idx]` so the body can splice in the tail.
    // Used for both the rule body AND any body-level `/;` guard, so a guard
    // like `… /; n1 > n2` over destructured elements is checked against the
    // bound `Part[_lp, i]` values rather than unbound symbols.
    let apply_list_substitutions = |expr: &Expr| -> Expr {
      let mut out = expr.clone();
      for (name, replacement) in &body_substitutions {
        out = crate::syntax::substitute_variable(&out, name, replacement);
      }
      out
    };
    let final_body = apply_list_substitutions(body);
    let body_condition_sub = body_condition.map(apply_list_substitutions);

    // If there's a body-level condition (from /;), attach it to a condition slot
    if let Some(body_cond) = body_condition_sub.as_ref() {
      let mut attached = false;
      for c in conditions.iter_mut() {
        if c.is_none() {
          *c = Some(body_cond.clone());
          attached = true;
          break;
        }
      }
      if !attached && !conditions.is_empty() {
        // All slots have conditions - combine with first non-structural-pattern using And
        let combine_idx = conditions.iter().position(|c| {
          !matches!(
            c,
            Some(Expr::FunctionCall { name, .. }) if name == "__StructuralPattern__"
          )
        });
        if let Some(idx) = combine_idx {
          let existing = conditions[idx].take().unwrap();
          conditions[idx] = Some(Expr::FunctionCall {
            name: "And".to_string(),
            args: vec![existing, body_cond.clone()].into(),
          });
        } else {
          // All conditions are structural patterns — append as extra condition
          conditions.push(Some(body_cond.clone()));
          params.push(String::new());
          defaults.push(None);
          heads.push(None);
          blank_types.push(1);
        }
      }
    }

    // Check if all args are literal (non-pattern) — if so, insert at beginning
    // for priority over general patterns (matching Mathematica specificity ordering)
    let has_literal_conditions =
      conditions.iter().flatten().any(condition_is_literal_arg);

    crate::FUNC_DEFS.with(|m| {
      let mut defs = m.borrow_mut();
      let entry = defs.entry(func_name.clone()).or_insert_with(Vec::new);
      let insert_pos = if suppress_specificity_sort() {
        // DownValues[f] := {...} replay: append in source order so the
        // caller's listing is preserved.
        entry.len()
      } else if has_literal_conditions {
        // Literal-match definitions go before pattern definitions but after
        // existing literal definitions (preserving definition order).
        entry
          .iter()
          .position(|(_, c, _, _, _, _)| {
            !c.iter().flatten().any(condition_is_literal_arg)
          })
          .unwrap_or(entry.len())
      } else {
        // Insert by the rule partial order: place the new rule before the first
        // existing rule it strictly dominates (is more specific than). Rules it
        // does not dominate — including incomparable ones — keep definition
        // order, matching Wolfram.
        entry
          .iter()
          .position(|(p, c, d, h, bt, b)| {
            rule_dominates(
              &params,
              &heads,
              &blank_types,
              &conditions,
              &defaults,
              &final_body,
              p,
              h,
              bt,
              c,
              d,
              b,
            )
          })
          .unwrap_or(entry.len())
      };
      entry.insert(
        insert_pos,
        (params, conditions, defaults, heads, blank_types, final_body),
      );
      // Store inline OptionsPattern defaults, keeping in sync with FUNC_DEFS entries
      crate::FUNC_OPTS_INLINE.with(|oi| {
        let mut inline_map = oi.borrow_mut();
        let inline_entry =
          inline_map.entry(func_name.clone()).or_insert_with(Vec::new);
        // Ensure the inline_entry has the same length as the FUNC_DEFS entry
        while inline_entry.len() < entry.len() {
          inline_entry.push(None);
        }
        // Set the inline defaults for this overload at the correct position
        if let Some(ref opts) = inline_opts_defaults {
          inline_entry[insert_pos] = Some(opts.clone());
        }
      });
    });

    return Ok(Expr::Identifier("Null".to_string()));
  }

  // Handle simple identifier assignment: a := expr (OwnValues)
  if let Expr::Identifier(var_name) = lhs {
    if is_symbol_protected(var_name) {
      crate::emit_message(&format!(
        "SetDelayed::wrsym: Symbol {} is Protected.",
        var_name
      ));
      return Ok(Expr::Identifier("Null".to_string()));
    }
    // If a `/; cond` clause was stripped from the LHS (or RHS), wrap the
    // body in `Condition[body, cond]` so the lookup can re-check the
    // guard at access time and skip the rule when it fails (matching
    // Wolfram's `a /; b > 0 := 3` semantics).
    let stored_body = match body_condition {
      Some(cond) => Expr::FunctionCall {
        name: "Condition".to_string(),
        args: vec![body.clone(), cond.clone()].into(),
      },
      None => body.clone(),
    };
    // Store the unevaluated body — it will be re-evaluated each time the symbol is accessed
    ENV.with(|e| {
      e.borrow_mut()
        .insert(var_name.clone(), StoredValue::ExprVal(stored_body))
    });
    return Ok(Expr::Identifier("Null".to_string()));
  }

  // SubValue form: f[a][b] := rhs (also deeper nestings like f[a][b][c])
  // Mathematica stores these under SubValues[f] and they fire when exactly
  // f[a][b] is evaluated. Record the rule keyed by the outermost head so
  // `SubValues[f]` can return them. Dispatch (`f[1][5]` → `5`) is not yet
  // wired up here.
  if let Expr::CurriedCall { func, .. } = lhs {
    let mut inner = func.as_ref();
    let outer_head = loop {
      match inner {
        Expr::CurriedCall { func: f2, .. } => inner = f2.as_ref(),
        Expr::FunctionCall { name, .. } => break Some(name.clone()),
        _ => break None,
      }
    };
    if let Some(head) = outer_head {
      SUB_VALUES.with(|m| {
        m.borrow_mut()
          .entry(head)
          .or_default()
          .push((lhs.clone(), body.clone()));
      });
      return Ok(Expr::Identifier("Null".to_string()));
    }
  }

  // Fallback: return symbolic form
  Ok(Expr::FunctionCall {
    name: "SetDelayed".to_string(),
    args: vec![lhs.clone(), body.clone()].into(),
  })
}

/// Whether a list-pattern element needs an explicit `MatchQ` guard at
/// dispatch. A bare blank (`x_`, `_`, `x__`, `x___`) with no head constraint
/// matches anything, so no guard is needed. Anything else — a head-constrained
/// pattern (`x_Integer`), a literal (`0`, `"s"`), a `PatternTest`, or a nested
/// structural pattern — must be verified against the actual element.
fn element_needs_match_check(pat: &Expr) -> bool {
  match pat {
    Expr::Pattern { head, .. } => head.is_some(),
    Expr::PatternOptional { head, .. } => head.is_some(),
    _ => true,
  }
}

/// The `Part[base, idx+1]` / `Sequence@@Drop[base, idx]` expression that
/// extracts list element `idx` (0-based) from `base`. A trailing
/// `BlankSequence`/`BlankNullSequence` element binds to the spliced tail.
fn list_element_accessor(
  base: &Expr,
  idx: usize,
  is_trailing_seq: bool,
) -> Expr {
  if is_trailing_seq {
    Expr::FunctionCall {
      name: "Apply".to_string(),
      args: vec![
        Expr::Identifier("Sequence".to_string()),
        Expr::FunctionCall {
          name: "Drop".to_string(),
          args: vec![base.clone(), Expr::Integer(idx as i128)].into(),
        },
      ]
      .into(),
    }
  } else {
    Expr::FunctionCall {
      name: "Part".to_string(),
      args: vec![base.clone(), Expr::Integer((idx + 1) as i128)].into(),
    }
  }
}

/// Recursively collect the variable bindings for a list pattern. `base` is the
/// expression that yields the current (sub)list — the param identifier at the
/// top level, a `Part[…]` deeper down. Each leaf named pattern maps to the
/// accessor expression that extracts its value; a nested list pattern recurses
/// so `{{a_, b_}, c_}` binds `a`, `b`, `c`. Structural validation is handled
/// separately by the element `MatchQ` guards, so this only produces bindings.
fn collect_list_pattern_bindings(
  patterns: &[Expr],
  base: &Expr,
  out: &mut Vec<(String, Expr)>,
) {
  let n = patterns.len();
  for (idx, pat) in patterns.iter().enumerate() {
    let (_, _, bt) = extract_pattern_info(pat);
    let is_trailing_seq = idx + 1 == n && bt >= 2;
    let accessor = list_element_accessor(base, idx, is_trailing_seq);
    if let Expr::List(inner) = pat
      && !is_trailing_seq
    {
      collect_list_pattern_bindings(inner, &accessor, out);
      continue;
    }
    let (name, _, _) = extract_pattern_info(pat);
    if !name.is_empty() {
      out.push((name, accessor));
    }
  }
}

/// Build the dispatch condition and body bindings for a list pattern argument
/// bound to the parameter `base_name`. The condition verifies the list length
/// plus a `MatchQ[Part[base, i], …]` guard for every constrained element
/// (literal, head-constrained, or nested), and the bindings extract each leaf
/// pattern variable (recursing into nested lists).
fn build_list_pattern_match(
  patterns: &[Expr],
  base_name: &str,
) -> (Expr, Vec<(String, Expr)>) {
  let base = Expr::Identifier(base_name.to_string());
  // Detect a trailing sequence pattern (BlankSequence or BlankNullSequence).
  let (trailing_seq, trailing_seq_blank) = patterns
    .last()
    .map(|p| {
      let (_, _, bt) = extract_pattern_info(p);
      (bt >= 2, bt)
    })
    .unwrap_or((false, 0));
  // Length condition:
  // - All single-element patterns (`x_`): Length === N.
  // - Trailing `__` (BlankSequence): Length >= N (consumes ≥1).
  // - Trailing `___` (BlankNullSequence): Length >= N-1 (consumes ≥0).
  let len_cmp = if !trailing_seq {
    (crate::syntax::ComparisonOp::SameQ, patterns.len() as i128)
  } else if trailing_seq_blank == 2 {
    (
      crate::syntax::ComparisonOp::GreaterEqual,
      patterns.len() as i128,
    )
  } else {
    (
      crate::syntax::ComparisonOp::GreaterEqual,
      (patterns.len() - 1) as i128,
    )
  };
  let mut rule_conds: Vec<Expr> = vec![Expr::Comparison {
    operands: vec![
      Expr::FunctionCall {
        name: "Length".to_string(),
        args: vec![base.clone()].into(),
      },
      Expr::Integer(len_cmp.1),
    ],
    operators: vec![len_cmp.0],
  }];
  let last_pat_idx = patterns.len().saturating_sub(1);
  for (eidx, pat) in patterns.iter().enumerate() {
    // The trailing sequence element matches the (length-checked) tail; it has
    // no single Part to test against.
    if trailing_seq && eidx == last_pat_idx {
      continue;
    }
    // Emit a `MatchQ` guard for every element. For a constrained element
    // (literal/head/nested) it enforces the match; for a bare blank it is
    // always True but keeps the element's surface pattern (and name) on record
    // so `DownValues` can reconstruct the original `{…}` pattern. Specificity
    // scoring (`count_specificity_clauses`) ignores the always-true ones.
    rule_conds.push(Expr::FunctionCall {
      name: "MatchQ".to_string(),
      args: vec![list_element_accessor(&base, eidx, false), pat.clone()].into(),
    });
  }
  let combined = if rule_conds.len() == 1 {
    rule_conds.pop().unwrap()
  } else {
    Expr::FunctionCall {
      name: "And".to_string(),
      args: rule_conds.into(),
    }
  };
  let mut bindings = Vec::new();
  collect_list_pattern_bindings(patterns, &base, &mut bindings);
  (combined, bindings)
}

/// Replace every subexpression structurally equal to `from` with `to`. Used to
/// turn the lowered `Part[_lp, i]` element accessors back into their original
/// names when reconstructing a list-pattern rule for display.
fn replace_subexpr(expr: &Expr, from: &Expr, to: &Expr) -> Expr {
  if crate::evaluator::pattern_matching::expr_equal(expr, from) {
    return to.clone();
  }
  match expr {
    Expr::List(items) => {
      Expr::List(items.iter().map(|e| replace_subexpr(e, from, to)).collect())
    }
    Expr::FunctionCall { name, args } => Expr::FunctionCall {
      name: name.clone(),
      args: args
        .iter()
        .map(|e| replace_subexpr(e, from, to))
        .collect::<Vec<_>>()
        .into(),
    },
    Expr::BinaryOp { op, left, right } => Expr::BinaryOp {
      op: *op,
      left: Box::new(replace_subexpr(left, from, to)),
      right: Box::new(replace_subexpr(right, from, to)),
    },
    Expr::UnaryOp { op, operand } => Expr::UnaryOp {
      op: *op,
      operand: Box::new(replace_subexpr(operand, from, to)),
    },
    Expr::Comparison {
      operands,
      operators,
    } => Expr::Comparison {
      operands: operands
        .iter()
        .map(|e| replace_subexpr(e, from, to))
        .collect(),
      operators: operators.clone(),
    },
    Expr::Part { expr: e, index } => Expr::Part {
      expr: Box::new(replace_subexpr(e, from, to)),
      index: Box::new(replace_subexpr(index, from, to)),
    },
    Expr::CompoundExpr(es) => Expr::CompoundExpr(
      es.iter().map(|e| replace_subexpr(e, from, to)).collect(),
    ),
    Expr::Rule {
      pattern,
      replacement,
    } => Expr::Rule {
      pattern: Box::new(replace_subexpr(pattern, from, to)),
      replacement: Box::new(replace_subexpr(replacement, from, to)),
    },
    Expr::RuleDelayed {
      pattern,
      replacement,
    } => Expr::RuleDelayed {
      pattern: Box::new(replace_subexpr(pattern, from, to)),
      replacement: Box::new(replace_subexpr(replacement, from, to)),
    },
    Expr::Association(items) => Expr::Association(
      items
        .iter()
        .map(|(k, v)| {
          (replace_subexpr(k, from, to), replace_subexpr(v, from, to))
        })
        .collect(),
    ),
    other => other.clone(),
  }
}

/// Map each leaf pattern variable in `pat` to the `Part[…]` accessor `path`
/// that extracts it, recursing through nested list patterns. Used to undo the
/// element-accessor substitution when displaying a list-pattern rule.
fn map_part_names(pat: &Expr, path: &Expr, out: &mut Vec<(Expr, String)>) {
  let part = |p: &Expr, j: usize| Expr::FunctionCall {
    name: "Part".to_string(),
    args: vec![p.clone(), Expr::Integer(j as i128)].into(),
  };
  match pat {
    Expr::Pattern { name, .. }
    | Expr::PatternTest { name, .. }
    | Expr::PatternOptional { name, .. }
      if !name.is_empty() =>
    {
      out.push((path.clone(), name.clone()));
    }
    Expr::List(inner) => {
      for (j, ip) in inner.iter().enumerate() {
        map_part_names(ip, &part(path, j + 1), out);
      }
    }
    _ => {}
  }
}

/// Reconstruct the surface list pattern (`{p1, p2, …}`), the element-accessor →
/// name map, and any `/;` guard clauses for a lowered list-pattern parameter
/// `param` from its stored dispatch condition `cond`.
fn reconstruct_list_param(
  param: &str,
  cond: Option<&Expr>,
) -> (Expr, Vec<(Expr, String)>, Vec<Expr>) {
  let fallback = || Expr::Pattern {
    name: param.to_string(),
    head: Some("List".to_string()),
    blank_type: 1,
  };
  let Some(cond) = cond else {
    return (fallback(), vec![], vec![]);
  };
  // Flatten nested And[…] into individual clauses.
  fn flatten<'a>(e: &'a Expr, out: &mut Vec<&'a Expr>) {
    if let Expr::FunctionCall { name, args } = e
      && name == "And"
    {
      for a in args.iter() {
        flatten(a, out);
      }
    } else {
      out.push(e);
    }
  }
  let mut clauses = Vec::new();
  flatten(cond, &mut clauses);

  let mut length: Option<i128> = None;
  let mut fixed = true;
  let mut elem_pats: Vec<(usize, Expr)> = Vec::new();
  let mut guards: Vec<Expr> = Vec::new();
  for c in clauses {
    match c {
      Expr::Comparison {
        operands,
        operators,
      } if matches!(
        operands.first(),
        Some(Expr::FunctionCall { name, .. }) if name == "Length"
      ) =>
      {
        if let Some(Expr::Integer(n)) = operands.get(1) {
          length = Some(*n);
          fixed = operators
            .iter()
            .any(|o| matches!(o, crate::syntax::ComparisonOp::SameQ));
        }
      }
      Expr::FunctionCall { name, args }
        if name == "MatchQ"
          && args.len() == 2
          && matches!(&args[0],
            Expr::FunctionCall { name: pn, args: pargs }
              if pn == "Part"
                && matches!(pargs.first(), Some(Expr::Identifier(id)) if id == param)
                && matches!(pargs.get(1), Some(Expr::Integer(_)))) =>
      {
        if let Expr::FunctionCall { args: pargs, .. } = &args[0]
          && let Some(Expr::Integer(idx)) = pargs.get(1)
        {
          elem_pats.push((*idx as usize, args[1].clone()));
        }
      }
      other => guards.push(other.clone()),
    }
  }

  let mut part_names = Vec::new();
  for (idx, pat) in &elem_pats {
    let path = Expr::FunctionCall {
      name: "Part".to_string(),
      args: vec![
        Expr::Identifier(param.to_string()),
        Expr::Integer(*idx as i128),
      ]
      .into(),
    };
    map_part_names(pat, &path, &mut part_names);
  }

  // Surface elements: a MatchQ-guarded element uses its recorded sub-pattern;
  // a gap is an unconstrained blank. A non-fixed length (trailing sequence)
  // appends one `__`/`___` element whose name is not recoverable.
  let n_fixed = elem_pats.iter().map(|(i, _)| *i).max().unwrap_or(0);
  let mut elems: Vec<Expr> = Vec::new();
  for i in 1..=n_fixed {
    if let Some((_, p)) = elem_pats.iter().find(|(idx, _)| *idx == i) {
      elems.push(p.clone());
    } else {
      elems.push(Expr::Pattern {
        name: String::new(),
        head: None,
        blank_type: 1,
      });
    }
  }
  if !fixed {
    let bt = if length == Some(n_fixed as i128 + 1) {
      2
    } else {
      3
    };
    elems.push(Expr::Pattern {
      name: String::new(),
      head: None,
      blank_type: bt,
    });
  }
  (Expr::List(elems.into()), part_names, guards)
}

/// Rebuild a displayable DownValue (pattern args + body) for a stored rule that
/// uses one or more lowered list-pattern parameters (`_lp{i}`), turning them
/// back into surface `{…}` patterns and un-substituting the `Part[_lp, i]`
/// element accessors in the body/guard. Returns `None` when no list-pattern
/// parameter is present (the caller keeps its normal reconstruction).
pub fn reconstruct_list_downvalue(
  params: &[String],
  conditions: &[Option<Expr>],
  heads: &[Option<String>],
  blank_types: &[u8],
  body: &Expr,
) -> Option<(Vec<Expr>, Expr)> {
  if !params.iter().any(|p| p.starts_with("_lp")) {
    return None;
  }
  let mut display_body = body.clone();
  let mut guards: Vec<Expr> = Vec::new();
  let mut pattern_args: Vec<Expr> = Vec::with_capacity(params.len());
  for (i, p) in params.iter().enumerate() {
    let cond = conditions.get(i).and_then(|c| c.as_ref());
    if p.starts_with("_lp") {
      let (list_pat, part_names, mut g) = reconstruct_list_param(p, cond);
      for (path, name) in &part_names {
        let name_expr = Expr::Identifier(name.clone());
        display_body = replace_subexpr(&display_body, path, &name_expr);
        for gg in g.iter_mut() {
          *gg = replace_subexpr(gg, path, &name_expr);
        }
      }
      guards.append(&mut g);
      pattern_args.push(list_pat);
    } else if let Some(Expr::Comparison {
      operands,
      operators,
    }) = cond
      && operators
        .iter()
        .any(|op| matches!(op, crate::syntax::ComparisonOp::SameQ))
      && let Some(lit) = operands.get(1)
    {
      pattern_args.push(lit.clone());
    } else {
      pattern_args.push(Expr::Pattern {
        name: p.clone(),
        head: heads.get(i).and_then(|h| h.clone()),
        blank_type: blank_types.get(i).copied().unwrap_or(1),
      });
    }
  }
  let final_body = if guards.is_empty() {
    display_body
  } else {
    let guard = if guards.len() == 1 {
      guards.pop().unwrap()
    } else {
      Expr::FunctionCall {
        name: "And".to_string(),
        args: guards.into(),
      }
    };
    Expr::FunctionCall {
      name: "Condition".to_string(),
      args: vec![display_body, guard].into(),
    }
  };
  Some((pattern_args, final_body))
}

/// Extract a pattern name, optional head constraint, and blank type from a pattern expression.
/// Returns (name, head, blank_type) where blank_type is:
///   1 = Blank (_), 2 = BlankSequence (__), 3 = BlankNullSequence (___)
/// e.g., x_Integer -> ("x", Some("Integer"), 1), x__ -> ("x", None, 2)
pub fn extract_pattern_info(expr: &Expr) -> (String, Option<String>, u8) {
  match expr {
    // AST Pattern node: Pattern { name: "x", head: Some("Integer"), blank_type: 1 }
    Expr::Pattern {
      name,
      head,
      blank_type,
    } => (name.clone(), head.clone(), *blank_type),
    // AST PatternOptional node: x_:default or x_Head:default
    Expr::PatternOptional { name, head, .. } => (name.clone(), head.clone(), 1),
    // AST PatternTest node: x_?test or x_Head?test
    Expr::PatternTest {
      name,
      head,
      blank_type,
      ..
    } => (name.clone(), head.clone(), *blank_type),
    Expr::Identifier(name) => {
      // Could be a pattern like "x_Integer", "x_", "x__", "x___" in text form
      if let Some(pos) = name.find('_') {
        let pat_name = name[..pos].to_string();
        let rest = &name[pos..];
        // Count consecutive underscores (1=Blank, 2=BlankSequence, 3=BlankNullSequence)
        let num_underscores = rest.chars().take_while(|c| *c == '_').count();
        let blank_type = num_underscores.min(3) as u8;
        let head_str = &rest[num_underscores..];
        if head_str.is_empty() {
          (pat_name, None, blank_type)
        } else {
          (pat_name, Some(head_str.to_string()), blank_type)
        }
      } else {
        (name.clone(), None, 1)
      }
    }
    Expr::FunctionCall { name, args }
      if name == "Pattern" && args.len() == 2 =>
    {
      // Pattern[name, Blank[head]], Pattern[name, BlankSequence[head]], Pattern[name, BlankNullSequence[head]]
      if let Expr::Identifier(pat_name) = &args[0]
        && let Expr::FunctionCall {
          name: blank_name,
          args: blank_args,
        } = &args[1]
      {
        let blank_type = match blank_name.as_str() {
          "Blank" => 1,
          "BlankSequence" => 2,
          "BlankNullSequence" => 3,
          _ => 1,
        };
        let head = blank_args.first().and_then(|a| {
          if let Expr::Identifier(h) = a {
            Some(h.clone())
          } else {
            None
          }
        });
        return (pat_name.clone(), head, blank_type);
      }
      (String::new(), None, 1)
    }
    _ => {
      // Structural patterns (e.g., BinaryOp containing patterns) are not
      // simple named patterns — return empty to signal special handling.
      (String::new(), None, 1)
    }
  }
}

/// Handle TagSetDelayed[tag, lhs, rhs] — stores an upvalue definition.
/// When evaluate_rhs is true, acts as TagSet (evaluates the RHS first).
pub fn tag_set_delayed_ast(
  tag: &Expr,
  lhs: &Expr,
  body: &Expr,
  evaluate_rhs: bool,
) -> Result<Expr, InterpreterError> {
  let tag_name = match tag {
    Expr::Identifier(s) => s.clone(),
    _ => {
      return Err(InterpreterError::EvaluationError(
        "TagSetDelayed: first argument must be a symbol".into(),
      ));
    }
  };

  let body = if evaluate_rhs {
    evaluate_expr_to_expr(body)?
  } else {
    body.clone()
  };

  // Extract Condition from body: Condition[actual_body, test] → (actual_body, Some(test))
  let (body, body_condition) = if let Expr::FunctionCall { name, args } = &body
    && name == "Condition"
    && args.len() == 2
  {
    (args[0].clone(), Some(args[1].clone()))
  } else {
    (body, None)
  };

  // Unwrap Condition from LHS: x + y_ /; y > -2 is parsed as
  // Condition[Plus[x, y_], Greater[y, -2]]. Extract the inner LHS and condition.
  // Keep original_lhs for UPVALUES display (includes the Condition wrapper).
  let original_lhs = lhs;
  let (lhs, lhs_condition) = if let Expr::FunctionCall { name, args } = lhs
    && name == "Condition"
    && args.len() == 2
  {
    (&args[0], Some(&args[1]))
  } else {
    (lhs, None)
  };

  // Extract outer function name and args from the LHS
  // Handles FunctionCall directly, and converts BinaryOp/UnaryOp/Comparison
  // to their canonical function call form (e.g. Plus[a, b] for a + b).
  let (outer_func, lhs_args): (String, Vec<Expr>) = match lhs {
    Expr::FunctionCall { name, args } => (name.clone(), args.to_vec()),
    Expr::BinaryOp { op, left, right } => {
      let (name, args) = match op {
        BinaryOperator::Plus => {
          ("Plus".to_string(), collect_binary_children(lhs, op))
        }
        BinaryOperator::Times => {
          ("Times".to_string(), collect_binary_children(lhs, op))
        }
        BinaryOperator::Alternatives => {
          ("Alternatives".to_string(), collect_binary_children(lhs, op))
        }
        BinaryOperator::Minus => (
          "Plus".to_string(),
          vec![
            left.as_ref().clone(),
            Expr::BinaryOp {
              op: BinaryOperator::Times,
              left: Box::new(Expr::Integer(-1)),
              right: right.clone(),
            },
          ],
        ),
        BinaryOperator::Divide => (
          "Times".to_string(),
          vec![
            left.as_ref().clone(),
            Expr::BinaryOp {
              op: BinaryOperator::Power,
              left: right.clone(),
              right: Box::new(Expr::Integer(-1)),
            },
          ],
        ),
        BinaryOperator::Power => (
          "Power".to_string(),
          vec![left.as_ref().clone(), right.as_ref().clone()],
        ),
        BinaryOperator::And => (
          "And".to_string(),
          vec![left.as_ref().clone(), right.as_ref().clone()],
        ),
        BinaryOperator::Or => (
          "Or".to_string(),
          vec![left.as_ref().clone(), right.as_ref().clone()],
        ),
        BinaryOperator::StringJoin => (
          "StringJoin".to_string(),
          vec![left.as_ref().clone(), right.as_ref().clone()],
        ),
      };
      (name, args)
    }
    Expr::UnaryOp { op, operand } => {
      let (name, args) = match op {
        UnaryOperator::Minus => (
          "Times".to_string(),
          vec![Expr::Integer(-1), operand.as_ref().clone()],
        ),
        UnaryOperator::Not => {
          ("Not".to_string(), vec![operand.as_ref().clone()])
        }
      };
      (name, args)
    }
    _ => {
      return Err(InterpreterError::EvaluationError(
        "TagSetDelayed: second argument must be a function call".into(),
      ));
    }
  };

  // Process each argument in the LHS to extract patterns
  let mut params = Vec::new();
  let mut conditions: Vec<Option<Expr>> = Vec::new();
  let mut defaults: Vec<Option<Expr>> = Vec::new();
  let mut heads: Vec<Option<String>> = Vec::new();
  let mut final_body = body.clone();
  // Track first occurrence of each pattern variable for SameQ conditions
  // on repeated pattern variables (e.g. v_ appearing in multiple args).
  let mut seen_pattern_vars: std::collections::HashMap<String, Expr> =
    std::collections::HashMap::new();
  // Extra conditions for repeated pattern variables
  let mut extra_conditions: Vec<Expr> = Vec::new();

  for (i, arg) in lhs_args.iter().enumerate() {
    match arg {
      Expr::FunctionCall {
        name: arg_func_name,
        args: inner_args,
      } => {
        let param_name = format!("_up{}", i);
        heads.push(Some(arg_func_name.clone()));

        if !inner_args.is_empty() {
          conditions.push(Some(Expr::Comparison {
            operands: vec![
              Expr::FunctionCall {
                name: "Length".to_string(),
                args: vec![Expr::Identifier(param_name.clone())].into(),
              },
              Expr::Integer(inner_args.len() as i128),
            ],
            operators: vec![crate::syntax::ComparisonOp::SameQ],
          }));
        } else {
          conditions.push(None);
        }

        for (j, inner_arg) in inner_args.iter().enumerate() {
          let (pat_name, _pat_head, _blank_type) =
            extract_pattern_info(inner_arg);
          if !pat_name.is_empty() {
            let part_expr = Expr::FunctionCall {
              name: "Part".to_string(),
              args: vec![
                Expr::Identifier(param_name.clone()),
                Expr::Integer((j + 1) as i128),
              ]
              .into(),
            };
            // If this pattern variable was already seen, add a SameQ
            // condition to ensure both occurrences match the same value.
            if let Some(prev_expr) = seen_pattern_vars.get(&pat_name) {
              extra_conditions.push(Expr::Comparison {
                operands: vec![prev_expr.clone(), part_expr.clone()],
                operators: vec![crate::syntax::ComparisonOp::SameQ],
              });
            } else {
              seen_pattern_vars.insert(pat_name.clone(), part_expr.clone());
            }
            final_body = crate::syntax::substitute_variable(
              &final_body,
              &pat_name,
              &part_expr,
            );
          }
        }

        params.push(param_name);
        defaults.push(None);
      }
      _ => {
        // Check if this argument is actually a pattern (contains _ or is a Pattern node)
        // Plain identifiers like `x` are literal symbols, not patterns.
        let is_pattern = match arg {
          Expr::Identifier(name) => name.contains('_'),
          Expr::Pattern { .. } | Expr::PatternOptional { .. } => true,
          _ => crate::evaluator::pattern_matching::contains_pattern(arg),
        };
        if is_pattern {
          let (pat_name, head, _blank_type) = extract_pattern_info(arg);
          if pat_name.is_empty() && head.is_none() {
            // Anonymous pattern — use generated name
            let param_name = format!("_up{}", i);
            params.push(param_name);
          } else {
            params.push(pat_name);
          }
          conditions.push(None);
          heads.push(head);
        } else {
          // Literal argument — must match exactly via SameQ
          let param_name = format!("_up{}", i);
          let eval_arg = evaluate_expr_to_expr(arg)?;
          conditions.push(Some(Expr::Comparison {
            operands: vec![Expr::Identifier(param_name.clone()), eval_arg],
            operators: vec![crate::syntax::ComparisonOp::SameQ],
          }));
          params.push(param_name);
          defaults.push(None);
          heads.push(None);
          continue;
        }
        defaults.push(None);
      }
    }
  }

  // Add extra conditions for repeated pattern variables to the last
  // parameter's condition slot (merging with any existing condition).
  if !extra_conditions.is_empty() && !conditions.is_empty() {
    let last_idx = conditions.len() - 1;
    let mut all_conds = extra_conditions;
    if let Some(existing) = conditions[last_idx].take() {
      all_conds.insert(0, existing);
    }
    // Combine all conditions with And
    let combined = if all_conds.len() == 1 {
      all_conds.remove(0)
    } else {
      Expr::FunctionCall {
        name: "And".to_string(),
        args: all_conds.into(),
      }
    };
    conditions[last_idx] = Some(combined);
  }

  // Attach any conditions from the LHS (/;) or body to condition slots
  for extra_cond in lhs_condition.into_iter().chain(body_condition.as_ref()) {
    let mut attached = false;
    for c in conditions.iter_mut() {
      if c.is_none() {
        *c = Some(extra_cond.clone());
        attached = true;
        break;
      }
    }
    if !attached && !conditions.is_empty() {
      // All slots have conditions - combine with first non-SameQ one using And
      let combine_idx = conditions.iter().position(|c| {
        !matches!(
          c,
          Some(Expr::Comparison { operators, .. })
          if operators.iter().any(|op| matches!(op, crate::syntax::ComparisonOp::SameQ))
        )
      }).unwrap_or(0);
      let existing = conditions[combine_idx].take().unwrap();
      conditions[combine_idx] = Some(Expr::FunctionCall {
        name: "And".to_string(),
        args: vec![existing, extra_cond.clone()].into(),
      });
    } else if !attached {
      // No condition slots at all — add a new one
      conditions.push(Some(extra_cond.clone()));
      params.push(String::new());
      defaults.push(None);
      heads.push(None);
    }
  }

  // Store in UPVALUES for introspection and cleanup.
  // If an upvalue with the same original LHS already exists, replace it.
  // Use original_lhs (with Condition wrapper) for display purposes.
  let lhs_str = crate::syntax::expr_to_string(original_lhs);
  crate::UPVALUES.with(|m| {
    let mut defs = m.borrow_mut();
    let entry = defs.entry(tag_name).or_insert_with(Vec::new);
    if let Some(pos) =
      entry.iter().position(|(_, _, _, _, _, _, orig_lhs, _)| {
        crate::syntax::expr_to_string(orig_lhs) == lhs_str
      })
    {
      entry[pos] = (
        outer_func.clone(),
        params.clone(),
        conditions.clone(),
        defaults.clone(),
        heads.clone(),
        final_body.clone(),
        original_lhs.clone(),
        body.clone(),
      );
    } else {
      entry.push((
        outer_func.clone(),
        params.clone(),
        conditions.clone(),
        defaults.clone(),
        heads.clone(),
        final_body.clone(),
        original_lhs.clone(),
        body.clone(),
      ));
    }
  });

  // Store in FUNC_DEFS under the outer function name.
  // Remove any existing upvalue definition with the same params/heads/conditions
  // before inserting the new one (to avoid duplicates on redefinition).
  // Only remove if conditions also match (different conditions = different rules).
  let blank_types = vec![1u8; params.len()];
  let cond_strs: Vec<String> = conditions
    .iter()
    .map(|c| {
      c.as_ref()
        .map_or(String::new(), crate::syntax::expr_to_string)
    })
    .collect();
  crate::FUNC_DEFS.with(|m| {
    let mut defs = m.borrow_mut();
    let entry = defs.entry(outer_func).or_insert_with(Vec::new);
    entry.retain(|(p, c, _, h, bt, _)| {
      if p == &params && h == &heads && bt == &blank_types {
        // Only remove if conditions also match
        let existing_cond_strs: Vec<String> = c
          .iter()
          .map(|cond| {
            cond
              .as_ref()
              .map_or(String::new(), crate::syntax::expr_to_string)
          })
          .collect();
        existing_cond_strs != cond_strs
      } else {
        true
      }
    });
    // Insert after existing upvalue entries (params starting with _up)
    // to maintain definition order, but before DownValues.
    let upvalue_count = entry
      .iter()
      .take_while(|(p, _, _, _, _, _)| {
        p.iter().any(|name| name.starts_with("_up"))
      })
      .count();
    entry.insert(
      upvalue_count,
      (params, conditions, defaults, heads, blank_types, final_body),
    );
  });

  if evaluate_rhs {
    Ok(body)
  } else {
    Ok(Expr::Identifier("Null".to_string()))
  }
}

/// TagUnset[tag, lhs] — removes an upvalue definition for tag.
/// g /: f[g[x_]] =. removes the upvalue stored for g under f.
pub fn tag_unset_ast(tag: &Expr, lhs: &Expr) -> Result<Expr, InterpreterError> {
  let tag_name = match tag {
    Expr::Identifier(s) => s.clone(),
    _ => {
      return Err(InterpreterError::EvaluationError(
        "TagUnset: first argument must be a symbol".into(),
      ));
    }
  };

  let outer_func = match lhs {
    Expr::FunctionCall { name, .. } => name.clone(),
    Expr::CurriedCall { func, .. } => {
      if let Expr::FunctionCall { name, .. } = func.as_ref() {
        name.clone()
      } else {
        return Ok(Expr::Identifier("Null".to_string()));
      }
    }
    _ => return Ok(Expr::Identifier("Null".to_string())),
  };

  let lhs_str = crate::syntax::expr_to_string(lhs);

  // Remove matching entries from UPVALUES
  let removed = crate::UPVALUES.with(|m| {
    let mut defs = m.borrow_mut();
    let mut removed_entries = Vec::new();
    if let Some(entry) = defs.get_mut(&tag_name) {
      entry.retain(
        |(
          _of,
          _params,
          _conds,
          _defaults,
          _heads,
          _body,
          orig_lhs,
          _orig_body,
        )| {
          let orig_lhs_str = crate::syntax::expr_to_string(orig_lhs);
          if orig_lhs_str == lhs_str {
            removed_entries
              .push((_params.clone(), crate::syntax::expr_to_string(_body)));
            false
          } else {
            true
          }
        },
      );
      if entry.is_empty() {
        defs.remove(&tag_name);
      }
    }
    removed_entries
  });

  // Also remove from FUNC_DEFS
  if !removed.is_empty() {
    crate::FUNC_DEFS.with(|m| {
      if let Some(entry) = m.borrow_mut().get_mut(&outer_func) {
        for (params, body_str) in &removed {
          entry.retain(|(p, _, _, _, _, b)| {
            !(p == params && crate::syntax::expr_to_string(b) == *body_str)
          });
        }
      }
    });
  }

  Ok(Expr::Identifier("Null".to_string()))
}

/// Convert parser-level BinaryOp/UnaryOp/Comparison nodes that may appear on
/// an UpSet-like LHS (e.g. `a + b ^= rhs`) into the equivalent FunctionCall,
/// so the UpSet/UpSetDelayed handlers can extract head + args uniformly.
fn normalize_lhs_for_upset(lhs: &Expr) -> Expr {
  match lhs {
    Expr::BinaryOp { op, left, right } => {
      let head = match op {
        BinaryOperator::Plus | BinaryOperator::Minus => "Plus",
        BinaryOperator::Times | BinaryOperator::Divide => "Times",
        BinaryOperator::Power => "Power",
        _ => return lhs.clone(),
      };
      let right_expr = match op {
        BinaryOperator::Minus => Expr::FunctionCall {
          name: "Times".to_string(),
          args: vec![Expr::Integer(-1), (**right).clone()].into(),
        },
        BinaryOperator::Divide => Expr::FunctionCall {
          name: "Power".to_string(),
          args: vec![(**right).clone(), Expr::Integer(-1)].into(),
        },
        _ => (**right).clone(),
      };
      Expr::FunctionCall {
        name: head.to_string(),
        args: vec![(**left).clone(), right_expr].into(),
      }
    }
    _ => lhs.clone(),
  }
}

/// UpSet[lhs, rhs] — automatically assigns upvalues to all symbols in the arguments of lhs.
/// f[g] ^= 5 stores an upvalue for g such that f[g] evaluates to 5.
pub fn upset_ast(lhs: &Expr, rhs: &Expr) -> Result<Expr, InterpreterError> {
  // Normalize parser-level BinaryOp/UnaryOp into FunctionCall so expressions
  // like `a + b ^= 2` (parsed as Plus[a, b] via BinaryOp) can serve as LHS.
  let normalized_lhs = normalize_lhs_for_upset(lhs);
  let (lhs_head, lhs_args) = match &normalized_lhs {
    Expr::FunctionCall { name, args } => (name.clone(), args.clone()),
    _ => {
      // Atomic LHS (e.g. `a ^= 3`): wolframscript emits UpSet::normal and
      // leaves the call unevaluated. Match that behaviour rather than
      // aborting with an InterpreterError.
      crate::emit_message(&format!(
        "UpSet::normal: Nonatomic expression expected at position 1 in {} ^= {}.",
        crate::syntax::expr_to_string(lhs),
        crate::syntax::expr_to_string(rhs)
      ));
      return Ok(Expr::FunctionCall {
        name: "UpSet".to_string(),
        args: vec![lhs.clone(), rhs.clone()].into(),
      });
    }
  };
  let lhs = &normalized_lhs;

  // Evaluate the RHS
  let eval_rhs = evaluate_expr_to_expr(rhs)?;

  // `Format[sym] ^= value` goes to FormatValues, not UpValues, in
  // Wolfram. Woxi doesn't implement FormatValues yet, but we still
  // skip UPVALUES storage so UpValues[sym] stays empty.
  if lhs_head == "Format" {
    return Ok(eval_rhs);
  }

  // `N[sym, …] ^= value` goes to NValues (under sym), not UpValues
  // — Wolfram redirects these the same way `N[sym, …] = value`
  // does. Mirror the `set_ast` N-handler so `UpValues[sym]` stays
  // empty and `NValues[sym]` reflects the rule.
  if lhs_head == "N"
    && (lhs_args.len() == 1 || lhs_args.len() == 2)
    && let Expr::Identifier(sym_name) = &lhs_args[0]
  {
    let prec_part = if lhs_args.len() == 2 {
      let prec_eval = evaluate_expr_to_expr(&lhs_args[1])?;
      let as_real = match &prec_eval {
        Expr::Integer(n) => Some(Expr::Real(*n as f64)),
        Expr::Real(_) => Some(prec_eval.clone()),
        _ => None,
      };
      if let Some(prec_real) = as_real {
        Expr::List(
          vec![prec_real, Expr::Identifier("Infinity".to_string())].into(),
        )
      } else {
        prec_eval
      }
    } else {
      Expr::List(
        vec![
          Expr::Identifier("MachinePrecision".to_string()),
          Expr::Identifier("MachinePrecision".to_string()),
        ]
        .into(),
      )
    };
    let canonical_lhs = Expr::FunctionCall {
      name: "N".to_string(),
      args: vec![Expr::Identifier(sym_name.clone()), prec_part].into(),
    };
    N_VALUES.with(|m| {
      let mut map = m.borrow_mut();
      let entries = map.entry(sym_name.clone()).or_default();
      entries.retain(|(lhs_p, _)| {
        !crate::evaluator::pattern_matching::expr_equal(lhs_p, &canonical_lhs)
      });
      entries.push((canonical_lhs, eval_rhs.clone()));
    });
    return Ok(eval_rhs);
  }

  // Find all tag symbols in the arguments. Patterns like `_Q` carry the
  // tag in their `head` field; `Blank[Q]` (FunctionCall) is the same shape
  // when written via FullForm.
  let mut tags = Vec::new();
  for arg in &lhs_args {
    if let Some(h) = pattern_head_tag(arg) {
      tags.push(h);
      continue;
    }
    match arg {
      Expr::Identifier(s) => tags.push(s.clone()),
      Expr::FunctionCall { name, .. } => tags.push(name.clone()),
      _ => {} // Skip non-symbol arguments (integers, etc.)
    }
  }

  if tags.is_empty() {
    return Err(InterpreterError::EvaluationError(format!(
      "UpSet::nosym: {} does not contain a symbol to attach a rule to.",
      crate::syntax::expr_to_string(lhs)
    )));
  }

  // Store upvalue for each tag
  for tag in &tags {
    tag_set_delayed_ast(&Expr::Identifier(tag.clone()), lhs, &eval_rhs, true)?;
  }

  // UpSet returns the evaluated RHS
  Ok(eval_rhs)
}

/// UpSetDelayed[lhs, rhs] — like UpSet but with delayed evaluation (RHS not evaluated).
/// f[g] ^:= body stores a delayed upvalue for g such that f[g] evaluates body each time.
pub fn upset_delayed_ast(
  lhs: &Expr,
  rhs: &Expr,
) -> Result<Expr, InterpreterError> {
  let normalized_lhs = normalize_lhs_for_upset(lhs);
  let (_, lhs_args) = match &normalized_lhs {
    Expr::FunctionCall { name, args } => (name.clone(), args.clone()),
    _ => {
      return Err(InterpreterError::EvaluationError(format!(
        "UpSetDelayed::normal: Nonatomic expression expected at position 1 in {} ^:= {}",
        crate::syntax::expr_to_string(lhs),
        crate::syntax::expr_to_string(rhs)
      )));
    }
  };
  let lhs = &normalized_lhs;

  // Find all tag symbols in the arguments. Patterns like `_Q` carry the
  // tag in their `head` field; `Blank[Q]` (FunctionCall) is the same shape
  // when written via FullForm.
  let mut tags = Vec::new();
  for arg in &lhs_args {
    if let Some(h) = pattern_head_tag(arg) {
      tags.push(h);
      continue;
    }
    match arg {
      Expr::Identifier(s) => tags.push(s.clone()),
      Expr::FunctionCall { name, .. } => tags.push(name.clone()),
      _ => {} // Skip non-symbol arguments
    }
  }

  if tags.is_empty() {
    return Err(InterpreterError::EvaluationError(format!(
      "UpSetDelayed::nosym: {} does not contain a symbol to attach a rule to.",
      crate::syntax::expr_to_string(lhs)
    )));
  }

  // Store delayed upvalue for each tag (evaluate_rhs=false for delayed)
  for tag in &tags {
    tag_set_delayed_ast(&Expr::Identifier(tag.clone()), lhs, rhs, false)?;
  }

  // UpSetDelayed returns Null
  Ok(Expr::Identifier("Null".to_string()))
}
