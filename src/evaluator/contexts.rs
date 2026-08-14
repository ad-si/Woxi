//! Wolfram Language contexts: read-time symbol resolution.
//!
//! In the Wolfram Language a symbol's name is resolved *when it is read*,
//! not when it is evaluated. A bare `foo` is looked up in each context on
//! `$ContextPath` in turn; the first context that already has a symbol of
//! that name wins, and when none does the symbol is created in `$Context`.
//! That single rule is what makes the package idiom work:
//!
//! ```wolfram
//! BeginPackage["P`"]        (* $Context = P`, path = {P`, System`} *)
//! pub::usage = "…";         (* creates P`pub — nothing else has "pub" *)
//! Begin["`Private`"]        (* $Context = P`Private`, path unchanged *)
//! priv[] := 1               (* creates P`Private`priv — private *)
//! pub[] := priv[] + 1       (* "pub" is found on the path → P`pub *)
//! End[]
//! EndPackage[]              (* P` joins $ContextPath → pub is visible *)
//! ```
//!
//! Woxi keeps one flat symbol store, so a resolved symbol is represented by
//! its full name: `P`Private`priv`. Symbols in `Global`` keep their short
//! name as their key — that is what the store has always held, so nothing
//! changes for code that never opens a context. Resolution is therefore a
//! no-op until a `Begin`/`BeginPackage` moves `$Context` or `$ContextPath`
//! away from its default, which [`contexts_active`] reports.
//!
//! Symbols that ship with the language stay unqualified: they are `System``
//! symbols, and Woxi implements them in one namespace.

use std::cell::RefCell;
use std::collections::HashSet;

use crate::syntax::Expr;

thread_local! {
  /// Every symbol created so far, by full name (`Global`` symbols by their
  /// short name). Merely mentioning a symbol creates it, which is what
  /// lets a later mention in another context find it.
  static SYMBOL_TABLE: RefCell<HashSet<String>> = RefCell::new(HashSet::new());
}

/// Forget every symbol created in a context — used when the session resets.
pub fn clear_symbol_table() {
  SYMBOL_TABLE.with(|t| t.borrow_mut().clear());
}

/// Drop a symbol from the table — `Remove` makes a symbol stop existing.
pub fn forget_symbol(full_name: &str) {
  SYMBOL_TABLE.with(|t| t.borrow_mut().remove(full_name));
}

/// The full names of all symbols created in a context so far.
pub fn created_symbols() -> Vec<String> {
  SYMBOL_TABLE.with(|t| t.borrow().iter().cloned().collect())
}

/// Whether any context construct is in play. While `$Context` is `Global``
/// and `$ContextPath` is its default, every symbol resolves to itself and
/// the whole mechanism stays out of the way.
pub fn contexts_active() -> bool {
  crate::current_context() != "Global`"
    || crate::current_context_path()
      != vec!["System`".to_string(), "Global`".to_string()]
    || crate::has_context_aliases()
}

/// Expand a `$ContextAliases` prefix: with `mp`` aliased to `MyPkg``, the
/// name `mp`Foo` is read as `MyPkg`Foo`. Only the first segment is an alias,
/// and the rest of the name rides along — `mp`Sub`Foo` becomes
/// `MyPkg`Sub`Foo`. A name that no alias claims comes back unchanged.
pub fn expand_alias(name: &str) -> String {
  if !crate::has_context_aliases() {
    return name.to_string();
  }
  let Some(first) = name.find('`') else {
    return name.to_string();
  };
  // A name that starts with a backtick is relative to `$Context`; there is no
  // leading segment for an alias to match.
  if first == 0 {
    return name.to_string();
  }
  let prefix = &name[..=first];
  crate::context_aliases()
    .into_iter()
    .find(|(alias, _)| alias == prefix)
    .map_or_else(
      || name.to_string(),
      |(_, target)| format!("{target}{}", &name[first + 1..]),
    )
}

thread_local! {
  /// The contexts an input unit is being *read* against, when that differs
  /// from the ones open right now — everything on one line resolves against
  /// the state at the start of the line, even the statements that a
  /// `BeginPackage` earlier on the same line has since changed.
  static READ_CONTEXT: RefCell<Vec<(String, Vec<String>)>> =
    const { RefCell::new(Vec::new()) };
}

/// The contexts to resolve against: the ones the current input unit is being
/// read with, or simply the ones open now.
pub fn read_context() -> (String, Vec<String>) {
  READ_CONTEXT
    .with(|r| r.borrow().last().cloned())
    .unwrap_or_else(|| {
      (crate::current_context(), crate::current_context_path())
    })
}

/// Resolves symbols against a saved `$Context`/`$ContextPath` pair for as
/// long as it is alive.
pub struct ReadContext;

impl ReadContext {
  pub fn install(saved: &(String, Vec<String>)) -> Self {
    READ_CONTEXT.with(|r| r.borrow_mut().push(saved.clone()));
    Self
  }
}

impl Drop for ReadContext {
  fn drop(&mut self) {
    READ_CONTEXT.with(|r| {
      r.borrow_mut().pop();
    });
  }
}

/// Report every `$ContextAliases` entry that cannot do its job.
///
/// An alias claims a context name for itself, so a context that already holds
/// symbols of its own becomes unreachable once its name is an alias, and one
/// that is on `$ContextPath` would be searched under a name that no longer
/// means it. The Wolfram Language warns about both, and re-checks the whole
/// mapping every time it changes — so a warning repeats until the entry that
/// caused it is taken back out.
pub fn validate_aliases() {
  let path = crate::current_context_path();
  let symbols = known_symbols();
  for (alias, target) in crate::context_aliases() {
    if path.contains(&alias) {
      crate::emit_message_to_stdout(&format!(
        "$ContextAliases::cxconflict: Warning: the alias {alias} -> {target} \
         conflicts with the value of $ContextPath."
      ));
    }
    if symbols.iter().any(|(context, _)| *context == alias) {
      crate::emit_message_to_stdout(&format!(
        "$ContextAliases::cxinuse: Warning: Symbols already exist in the \
         context {alias}. These symbols will not be able to be accessed \
         while {alias} is in $ContextAliases."
      ));
    }
  }
}

/// Record `alias` as standing for `target` in `$ContextAliases`.
pub fn set_alias(alias: &str, target: &str) {
  crate::set_context_alias(alias, Some(target));
  validate_aliases();
}

/// The store key for `name` in `context`. `Global`` symbols are keyed by
/// their short name, the way Woxi has always stored them.
fn key_for(context: &str, name: &str) -> String {
  if context == "Global`" {
    name.to_string()
  } else {
    format!("{context}{name}")
  }
}

/// The context a resolved symbol belongs to — the part up to and including
/// its last backtick, or `Global`` for an unqualified user symbol.
pub fn context_of(full_name: &str) -> String {
  match full_name.rfind('`') {
    Some(last) => full_name[..=last].to_string(),
    None => "Global`".to_string(),
  }
}

/// The short name of a resolved symbol: everything after its last backtick.
pub fn short_name(full_name: &str) -> &str {
  match full_name.rfind('`') {
    Some(last) => &full_name[last + 1..],
    None => full_name,
  }
}

/// Whether `name` is an ordinary user symbol — one that resolution applies
/// to. Symbols that ship with the language are `System`` symbols and keep
/// their short name, and so do the `$…` system variables.
fn is_user_symbol(name: &str) -> bool {
  if name.is_empty() || name.starts_with('$') {
    return false;
  }
  if !name.starts_with(|c: char| c.is_ascii_alphabetic() || c == '`') {
    return false;
  }
  // A pattern variable arrives with its blank suffix stripped, but a stray
  // `_` (or a slot) is not a symbol name.
  if name.contains(['_', '#', '%']) {
    return false;
  }
  let bare = short_name(name);
  !crate::evaluator::is_builtin_symbol(bare)
    && crate::evaluator::get_builtin_attributes(bare).is_empty()
}

/// Whether a symbol with this full name exists — either created in a
/// context or already carrying a definition from before contexts came into
/// play (Woxi's stores are keyed the same way).
fn symbol_exists(full_name: &str) -> bool {
  if SYMBOL_TABLE.with(|t| t.borrow().contains(full_name)) {
    return true;
  }
  // A store that is mid-update is borrowed mutably by the evaluator; a
  // lookup that lands there (rendering a message while a value is being
  // assigned, say) reads it as "not here" rather than panicking. Every such
  // symbol is in the table anyway once it has been read.
  crate::ENV.with(|e| e.try_borrow().is_ok_and(|m| m.contains_key(full_name)))
    || crate::FUNC_DEFS
      .with(|m| m.try_borrow().is_ok_and(|m| m.contains_key(full_name)))
    || crate::FUNC_ATTRS
      .with(|m| m.try_borrow().is_ok_and(|m| m.contains_key(full_name)))
    || crate::MEMO_VALUES
      .with(|m| m.try_borrow().is_ok_and(|m| m.contains_key(full_name)))
    || crate::UPVALUES
      .with(|m| m.try_borrow().is_ok_and(|m| m.contains_key(full_name)))
}

/// Record `full_name` as created, reporting `::shdw` when the same short
/// name now lives in more than one context — the symbol just created
/// shadows (or is shadowed by) the others.
///
/// Only contexts that can actually shadow are counted: those on
/// `$ContextPath`, the current context, and `Global``, which every session
/// returns to. A package's private context is on none of them, so defining
/// `Global`priv` alongside `P`Private`priv` is not a clash.
fn create_symbol(full_name: &str) {
  let is_new =
    SYMBOL_TABLE.with(|t| t.borrow_mut().insert(full_name.to_string()));
  if !is_new {
    return;
  }
  let bare = short_name(full_name);
  let own_context = context_of(full_name);
  let mut visible = crate::current_context_path();
  visible.push(crate::current_context());
  visible.push("Global`".to_string());
  let mut others: Vec<String> = created_symbols()
    .iter()
    .filter(|other| short_name(other) == bare && **other != full_name)
    .map(|other| context_of(other))
    .collect();
  // Symbols that predate any context live in Global`.
  if own_context != "Global`"
    && crate::get_defined_names().iter().any(|n| n == bare)
  {
    others.push("Global`".to_string());
  }
  others.retain(|ctx| visible.contains(ctx));
  others.sort();
  others.dedup();
  if others.is_empty() || !visible.contains(&own_context) {
    return;
  }
  let contexts = std::iter::once(own_context.clone())
    .chain(others)
    .collect::<Vec<_>>()
    .join(", ");
  crate::emit_message_to_stdout(&format!(
    "{bare}::shdw: Symbol {bare} appears in multiple contexts {{{contexts}}}; \
     definitions in context {own_context} may shadow or be shadowed by other \
     definitions."
  ));
}

/// Resolve a name as the Wolfram Language does when it reads it: the first
/// context on `$ContextPath` that already has the symbol wins, otherwise it
/// is created in `$Context`.
pub fn resolve(name: &str) -> String {
  if !contexts_active() || !is_user_symbol(name) {
    return name.to_string();
  }
  // A name that starts with a backtick is relative to the current context;
  // one that already carries a context is absolute.
  if let Some(relative) = name.strip_prefix('`') {
    let full = format!("{}{relative}", read_context().0);
    create_symbol(&full);
    return full;
  }
  if name.contains('`') {
    let full = expand_alias(name);
    create_symbol(&full);
    return full;
  }
  let (current, path) = read_context();
  for context in path {
    let key = key_for(&context, name);
    if symbol_exists(&key) {
      return key;
    }
  }
  let key = key_for(&current, name);
  create_symbol(&key);
  key
}

/// The symbol a name refers to right now, without creating one. Renderers
/// print a symbol under its visible short name, so anything that looks state
/// up from a rendered name has to map it back first.
pub fn resolve_existing(name: &str) -> String {
  if !contexts_active() || !is_user_symbol(name) {
    return name.to_string();
  }
  if name.contains('`') {
    return expand_alias(name);
  }
  crate::current_context_path()
    .into_iter()
    .map(|context| key_for(&context, name))
    .find(|key| symbol_exists(key))
    .unwrap_or_else(|| name.to_string())
}

/// How a resolved symbol is written: its short name where that reads back
/// as the same symbol, its full name otherwise. `P`pub` prints as `pub`
/// while `P`` is on `$ContextPath`, and as `P`pub` once it is not.
pub fn display_name(full_name: &str) -> String {
  if !full_name.contains('`') {
    return full_name.to_string();
  }
  let bare = short_name(full_name);
  if context_of(full_name) == "System`" {
    return bare.to_string();
  }
  // The short name is enough wherever reading it back names this very
  // symbol: the first context on `$ContextPath` that has it, or — when none
  // does — the current context, where reading the name would create it.
  let read_back = crate::current_context_path()
    .into_iter()
    .map(|ctx| key_for(&ctx, bare))
    .find(|key| symbol_exists(key))
    .unwrap_or_else(|| key_for(&crate::current_context(), bare));
  if read_back == full_name {
    bare.to_string()
  } else {
    full_name.to_string()
  }
}

/// Every symbol `Names`/`Contexts` can see, as `(context, short name)`.
/// Built-ins are `System`` symbols; symbols created in a context carry it
/// in their name; everything else in the stores is a `Global`` symbol.
pub fn known_symbols() -> Vec<(String, String)> {
  let mut symbols: Vec<(String, String)> = Vec::new();
  for builtin in crate::evaluator::all_builtin_symbol_names() {
    symbols.push(("System`".to_string(), builtin.to_string()));
  }
  for name in created_symbols()
    .into_iter()
    .chain(crate::get_defined_names())
    .chain(message_symbols())
  {
    // `$…` variables and the stores Woxi keeps under a built-in's name
    // (message texts live under `MessageName`) are `System`` symbols, not
    // user symbols in `Global``.
    if !is_user_symbol(&name) {
      continue;
    }
    symbols.push((context_of(&name), short_name(&name).to_string()));
  }
  symbols.sort();
  symbols.dedup();
  symbols
}

/// The order the Wolfram Language lists symbol names in: case-insensitive,
/// with `$` sorting after the letters (`zz1`, `zzA`, `zzb`, `zz$`).
pub fn name_sort_key(name: &str) -> String {
  name
    .chars()
    .map(|c| match c {
      '$' => '\u{7f}',
      other => other.to_ascii_lowercase(),
    })
    .collect()
}

/// The symbols that carry a message (`f::usage = "…"`). Messages are filed
/// as `MessageName` down-values rather than under the symbol itself, so a
/// symbol that only declares a usage — the way a package exports one — has
/// to be read back out of them.
fn message_symbols() -> Vec<String> {
  let defs = crate::FUNC_DEFS
    .with(|m| m.try_borrow().map(|d| d.get("MessageName").cloned()));
  let Ok(Some(defs)) = defs else {
    return Vec::new();
  };
  let mut symbols = Vec::new();
  for (params, conds, ..) in &defs {
    for cond in conds {
      if let Some(Expr::Comparison {
        operands,
        operators,
      }) = cond
        && operators.len() == 1
        && matches!(operators[0], crate::syntax::ComparisonOp::SameQ)
        && operands.len() == 2
        && matches!(&operands[0], Expr::Identifier(p) if params.first() == Some(p))
        && let Expr::Identifier(symbol) = &operands[1]
      {
        symbols.push(symbol.clone());
      }
    }
  }
  symbols
}

/// The store key a `(context, short name)` pair names.
pub fn full_name(context: &str, name: &str) -> String {
  key_for(context, name)
}

/// Resolve every symbol in `expr`, as reading the expression would.
///
/// Definitions store the rewritten body, so a package's private helpers stay
/// reachable from the package's own functions after the context is left —
/// exactly as they do in the Wolfram Language, where the names in a stored
/// definition were resolved when the definition was read.
pub fn rewrite(expr: &Expr) -> Expr {
  if !contexts_active() {
    return expr.clone();
  }
  map_symbols(expr, &resolve)
}

/// Rewrite `expr` the way it should be *written*: every symbol under the
/// short name that reads back as it, and its full name otherwise.
pub fn to_display(expr: &Expr) -> Expr {
  map_symbols(expr, &display_name)
}

/// Apply `f` to every symbol name in `expr` — heads, bare symbols, pattern
/// variables and named-function parameters alike.
fn map_symbols(expr: &Expr, f: &dyn Fn(&str) -> String) -> Expr {
  let resolve = |name: &String| f(name);
  let sub = |e: &Expr| map_symbols(e, f);
  let boxed = |e: &Expr| Box::new(map_symbols(e, f));
  match expr {
    Expr::Identifier(name) => Expr::Identifier(resolve(name)),
    Expr::Constant(name) => Expr::Constant(name.clone()),
    Expr::FunctionCall { name, args } => Expr::FunctionCall {
      name: resolve(name),
      args: args.iter().map(sub).collect(),
    },
    Expr::List(items) => Expr::List(items.iter().map(sub).collect()),
    Expr::BinaryOp { op, left, right } => Expr::BinaryOp {
      op: *op,
      left: boxed(left),
      right: boxed(right),
    },
    Expr::UnaryOp { op, operand } => Expr::UnaryOp {
      op: *op,
      operand: boxed(operand),
    },
    Expr::Comparison {
      operands,
      operators,
    } => Expr::Comparison {
      operands: operands.iter().map(sub).collect(),
      operators: operators.clone(),
    },
    Expr::CompoundExpr(items) => {
      Expr::CompoundExpr(items.iter().map(sub).collect())
    }
    Expr::Association(pairs) => {
      Expr::Association(pairs.iter().map(|(k, v)| (sub(k), sub(v))).collect())
    }
    Expr::Rule {
      pattern,
      replacement,
    } => Expr::Rule {
      pattern: boxed(pattern),
      replacement: boxed(replacement),
    },
    Expr::RuleDelayed {
      pattern,
      replacement,
    } => Expr::RuleDelayed {
      pattern: boxed(pattern),
      replacement: boxed(replacement),
    },
    Expr::ReplaceAll { expr, rules } => Expr::ReplaceAll {
      expr: boxed(expr),
      rules: boxed(rules),
    },
    Expr::ReplaceRepeated { expr, rules } => Expr::ReplaceRepeated {
      expr: boxed(expr),
      rules: boxed(rules),
    },
    Expr::Map { func, list } => Expr::Map {
      func: boxed(func),
      list: boxed(list),
    },
    Expr::Apply { func, list } => Expr::Apply {
      func: boxed(func),
      list: boxed(list),
    },
    Expr::MapApply { func, list } => Expr::MapApply {
      func: boxed(func),
      list: boxed(list),
    },
    Expr::PrefixApply { func, arg } => Expr::PrefixApply {
      func: boxed(func),
      arg: boxed(arg),
    },
    Expr::Postfix { expr, func } => Expr::Postfix {
      expr: boxed(expr),
      func: boxed(func),
    },
    Expr::Part { expr, index } => Expr::Part {
      expr: boxed(expr),
      index: boxed(index),
    },
    Expr::CurriedCall { func, args } => Expr::CurriedCall {
      func: boxed(func),
      args: args.iter().map(sub).collect(),
    },
    Expr::Function { body } => Expr::Function { body: boxed(body) },
    Expr::NamedFunction {
      params,
      body,
      bracketed,
    } => Expr::NamedFunction {
      params: params.iter().map(&resolve).collect(),
      body: boxed(body),
      bracketed: *bracketed,
    },
    // A pattern's name is a symbol of its own — `f[x_]` inside a package
    // creates that package's `x`. Its head is a type test (`_Integer`) and
    // resolves like any other symbol.
    Expr::Pattern {
      name,
      head,
      blank_type,
    } => Expr::Pattern {
      name: resolve(name),
      head: head.as_ref().map(&resolve),
      blank_type: *blank_type,
    },
    Expr::PatternOptional {
      name,
      head,
      default,
    } => Expr::PatternOptional {
      name: resolve(name),
      head: head.as_ref().map(&resolve),
      default: default.as_ref().map(|d| boxed(d)),
    },
    Expr::PatternTest {
      name,
      head,
      blank_type,
      test,
    } => Expr::PatternTest {
      name: resolve(name),
      head: head.as_ref().map(resolve),
      blank_type: *blank_type,
      test: boxed(test),
    },
    other => other.clone(),
  }
}
