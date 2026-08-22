//! Taking a symbol's definitions away and (optionally) putting them back.
//!
//! Wolfram's `Block[{s}, …]` localizes *everything* `s` stands for, not just
//! its own value: down-, sub-, up-, n- and format values, options, messages
//! and attributes all disappear for the duration of the body and come back
//! when it exits. `Block[{Hold}, Hold[1 + 1]]` is `Hold[2]` because even
//! `Hold`'s `HoldAll` is gone while the block runs.
//!
//! `ClearAll[s]` needs the same removal without the putting-back, so both go
//! through [`take_symbol_values`]; only `Block` keeps the snapshot and calls
//! [`restore_symbol_values`].

use super::*;
use crate::syntax::Expr;

/// One `FUNC_DEFS` entry: `(params, conditions, defaults, head_constraints,
/// blank_types, body)`.
type FuncDef = (
  Vec<String>,
  Vec<Option<Expr>>,
  Vec<Option<Expr>>,
  Vec<Option<String>>,
  Vec<u8>,
  Expr,
);

/// One `UPVALUES` entry: `(outer_func, params, conditions, defaults, heads,
/// body, original_lhs, original_body)`.
type UpValue = (
  String,
  Vec<String>,
  Vec<Option<Expr>>,
  Vec<Option<Expr>>,
  Vec<Option<String>>,
  Expr,
  Expr,
  Expr,
);

/// Heads that hold other symbols' values: `f::msg` is a `MessageName`
/// DownValue and `Default[f, …]` a `Default` one, but both belong to `f`.
const BORROWED_HEADS: [&str; 2] = ["MessageName", "Default"];

/// Whether a DownValue stored under `head` is really one of `sym`'s values.
fn belongs_to(head: &str, sym: &str, def: &FuncDef) -> bool {
  match head {
    "MessageName" => {
      crate::evaluator::assignment::is_message_of(sym, &def.0, &def.1)
    }
    _ => def.0.first().is_some_and(|p| p == sym),
  }
}

/// Every definition a symbol had, as taken out of the global stores.
///
/// Restoring writes each field back over whatever the body left behind, so a
/// definition made inside a `Block` is dropped exactly as Wolfram drops it.
#[derive(Default)]
pub(crate) struct SymbolValues {
  name: String,
  own: Option<crate::StoredValue>,
  down: Option<Vec<FuncDef>>,
  memo: Option<std::collections::HashMap<String, (Vec<Expr>, Expr)>>,
  attrs: Option<Vec<String>>,
  attrs_removed: Option<Vec<String>>,
  options: Option<Vec<Expr>>,
  options_delayed: bool,
  opts_inline: Option<Vec<Option<Vec<Expr>>>>,
  format: Option<Vec<(String, Expr, Expr)>>,
  sub: Option<Vec<(Expr, Expr)>>,
  n: Option<Vec<(Expr, Expr)>>,
  up: Option<Vec<UpValue>>,
  /// The mirrored copies an UpValue installs in the outer head's DownValues,
  /// grouped by that head.
  up_mirrors: Vec<(String, Vec<FuncDef>)>,
  /// `MessageName` / `Default` DownValues that name this symbol.
  borrowed: Vec<(String, Vec<FuncDef>)>,
}

/// Remove everything `sym` is defined to be and hand it back.
///
/// The symbol is left as bare as a never-mentioned name: `Attributes[sym]`
/// reports `{}` afterwards, which for a builtin means masking its default
/// attributes through `FUNC_ATTRS_REMOVED`.
pub(crate) fn take_symbol_values(sym: &str) -> SymbolValues {
  let mut saved = SymbolValues {
    name: sym.to_string(),
    ..Default::default()
  };

  saved.own = ENV.with(|e| e.borrow_mut().remove(sym));
  saved.down = crate::FUNC_DEFS.with(|m| m.borrow_mut().remove(sym));
  saved.memo = crate::MEMO_VALUES.with(|m| m.borrow_mut().remove(sym));
  saved.attrs = crate::FUNC_ATTRS.with(|m| m.borrow_mut().remove(sym));
  saved.options = crate::FUNC_OPTIONS.with(|m| m.borrow_mut().remove(sym));
  saved.options_delayed =
    crate::FUNC_OPTIONS_DELAYED.with(|m| m.borrow_mut().remove(sym));
  saved.opts_inline =
    crate::FUNC_OPTS_INLINE.with(|m| m.borrow_mut().remove(sym));
  saved.format = crate::evaluator::assignment::FORMAT_VALUES
    .with(|m| m.borrow_mut().remove(sym));
  saved.sub = crate::evaluator::assignment::SUB_VALUES
    .with(|m| m.borrow_mut().remove(sym));
  saved.n =
    crate::evaluator::assignment::N_VALUES.with(|m| m.borrow_mut().remove(sym));

  // `f::msg` and `Default[f, …]` hang off another head's DownValues but are
  // f's values, so they go too.
  for head in BORROWED_HEADS {
    let mine = crate::FUNC_DEFS.with(|m| {
      let mut map = m.borrow_mut();
      let Some(entries) = map.get_mut(head) else {
        return Vec::new();
      };
      if !entries.iter().any(|d| belongs_to(head, sym, d)) {
        return Vec::new();
      }
      let (mine, rest): (Vec<FuncDef>, Vec<FuncDef>) = std::mem::take(entries)
        .into_iter()
        .partition(|d| belongs_to(head, sym, d));
      *entries = rest;
      mine
    });
    if !mine.is_empty() {
      saved.borrowed.push((head.to_string(), mine));
    }
  }

  // An UpValue is stored twice: under its tag symbol, and mirrored into the
  // outer head's DownValues so dispatch finds it. Take both halves.
  saved.up = crate::UPVALUES.with(|m| m.borrow_mut().remove(sym));
  if let Some(up_defs) = &saved.up {
    for (outer_func, params, _, _, _, body, _, _) in up_defs {
      let body_str = crate::syntax::expr_to_string(body);
      let pulled = crate::FUNC_DEFS.with(|m| {
        let mut map = m.borrow_mut();
        let Some(entries) = map.get_mut(outer_func) else {
          return Vec::new();
        };
        let matches = |d: &FuncDef| {
          &d.0 == params && crate::syntax::expr_to_string(&d.5) == body_str
        };
        if !entries.iter().any(matches) {
          return Vec::new();
        }
        let (mine, rest): (Vec<FuncDef>, Vec<FuncDef>) =
          std::mem::take(entries).into_iter().partition(matches);
        *entries = rest;
        mine
      });
      if !pulled.is_empty() {
        saved.up_mirrors.push((outer_func.clone(), pulled));
      }
    }
  }

  // Attributes last: a builtin has no FUNC_ATTRS entry to remove, so its
  // defaults have to be masked explicitly for `Attributes[sym]` to be `{}`.
  saved.attrs_removed =
    crate::FUNC_ATTRS_REMOVED.with(|m| m.borrow_mut().remove(sym));
  let builtin = get_builtin_attributes(sym);
  if !builtin.is_empty() {
    crate::FUNC_ATTRS_REMOVED.with(|m| {
      m.borrow_mut().insert(
        sym.to_string(),
        builtin
          .iter()
          .map(std::string::ToString::to_string)
          .collect(),
      )
    });
  }

  saved
}

/// Put back what [`take_symbol_values`] took, discarding whatever was defined
/// in the meantime.
pub(crate) fn restore_symbol_values(saved: SymbolValues) {
  let sym = saved.name.as_str();

  restore_map(&crate::ENV, sym, saved.own);
  restore_map(&crate::FUNC_DEFS, sym, saved.down);
  restore_map(&crate::MEMO_VALUES, sym, saved.memo);
  restore_map(&crate::FUNC_ATTRS, sym, saved.attrs);
  restore_map(&crate::FUNC_ATTRS_REMOVED, sym, saved.attrs_removed);
  restore_map(&crate::FUNC_OPTIONS, sym, saved.options);
  restore_map(&crate::FUNC_OPTS_INLINE, sym, saved.opts_inline);
  restore_map(
    &crate::evaluator::assignment::FORMAT_VALUES,
    sym,
    saved.format,
  );
  restore_map(&crate::evaluator::assignment::SUB_VALUES, sym, saved.sub);
  restore_map(&crate::evaluator::assignment::N_VALUES, sym, saved.n);
  restore_map(&crate::UPVALUES, sym, saved.up);

  crate::FUNC_OPTIONS_DELAYED.with(|m| {
    let mut set = m.borrow_mut();
    if saved.options_delayed {
      set.insert(sym.to_string());
    } else {
      set.remove(sym);
    }
  });

  for (head, entries) in saved.borrowed {
    crate::FUNC_DEFS.with(|m| {
      let mut map = m.borrow_mut();
      let slot = map.entry(head.clone()).or_default();
      slot.retain(|d| !belongs_to(&head, sym, d));
      slot.extend(entries);
    });
  }
  for (head, entries) in saved.up_mirrors {
    crate::FUNC_DEFS
      .with(|m| m.borrow_mut().entry(head).or_default().extend(entries));
  }
}

/// Reinstate (or clear) a symbol's entry in one of the definition maps.
fn restore_map<V: 'static>(
  store: &'static std::thread::LocalKey<
    std::cell::RefCell<std::collections::HashMap<String, V>>,
  >,
  sym: &str,
  value: Option<V>,
) {
  store.with(|m| {
    let mut map = m.borrow_mut();
    match value {
      Some(v) => {
        map.insert(sym.to_string(), v);
      }
      None => {
        map.remove(sym);
      }
    }
  });
}
