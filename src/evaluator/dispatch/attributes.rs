#[allow(unused_imports)]
use super::*;

pub fn dispatch_attributes(
  name: &str,
  args: &[Expr],
) -> Option<Result<Expr, InterpreterError>> {
  match name {
    "SetAttributes" if args.len() == 2 => {
      let func_names: Vec<String> = match &args[0] {
        Expr::Identifier(name) => vec![name.clone()],
        Expr::List(items) => items
          .iter()
          .filter_map(|item| {
            if let Expr::Identifier(n) = item {
              Some(n.clone())
            } else {
              None
            }
          })
          .collect(),
        _ => vec![],
      };
      let attr: Vec<String> = match &args[1] {
        Expr::Identifier(a) => vec![a.clone()],
        Expr::List(items) => items
          .iter()
          .filter_map(|item| {
            if let Expr::Identifier(a) = item {
              Some(a.clone())
            } else {
              None
            }
          })
          .collect(),
        _ => vec![],
      };
      if !func_names.is_empty() {
        let mut locked = false;
        crate::FUNC_ATTRS.with(|m| {
          let mut attrs = m.borrow_mut();
          for func_name in &func_names {
            if let Some(existing) = attrs.get(func_name)
              && existing.contains(&"Locked".to_string())
            {
              crate::emit_message(&format!(
                "Attributes::locked: Symbol {} is locked.",
                func_name
              ));
              locked = true;
              continue;
            }
            let entry = attrs.entry(func_name.clone()).or_insert_with(Vec::new);
            for a in &attr {
              if !entry.contains(a) {
                entry.push(a.clone());
              }
            }
          }
        });
        if locked {
          return Some(Ok(Expr::Identifier("Null".to_string())));
        }
        return Some(Ok(Expr::Identifier("Null".to_string())));
      }
    }
    "ClearAttributes" if args.len() == 2 => {
      let func_names: Vec<String> = match &args[0] {
        Expr::Identifier(name) => vec![name.clone()],
        Expr::List(items) => items
          .iter()
          .filter_map(|item| {
            if let Expr::Identifier(n) = item {
              Some(n.clone())
            } else {
              None
            }
          })
          .collect(),
        _ => vec![],
      };
      let to_remove: Vec<String> = match &args[1] {
        Expr::Identifier(a) => vec![a.clone()],
        Expr::List(items) => items
          .iter()
          .filter_map(|item| {
            if let Expr::Identifier(a) = item {
              Some(a.clone())
            } else {
              None
            }
          })
          .collect(),
        _ => vec![],
      };
      if !func_names.is_empty() {
        crate::FUNC_ATTRS.with(|m| {
          let mut attrs = m.borrow_mut();
          for func_name in &func_names {
            if let Some(existing) = attrs.get(func_name)
              && existing.contains(&"Locked".to_string())
            {
              crate::emit_message(&format!(
                "Attributes::locked: Symbol {} is locked.",
                func_name
              ));
              continue;
            }
            if let Some(entry) = attrs.get_mut(func_name) {
              entry.retain(|a| !to_remove.contains(a));
            }
          }
        });
        return Some(Ok(Expr::Identifier("Null".to_string())));
      }
    }
    "Protect" => {
      let mut protected_syms = Vec::new();
      for arg in args {
        if let Expr::Identifier(sym) = arg {
          crate::FUNC_ATTRS.with(|m| {
            let mut attrs = m.borrow_mut();
            let entry = attrs.entry(sym.clone()).or_insert_with(Vec::new);
            if !entry.contains(&"Protected".to_string()) {
              entry.push("Protected".to_string());
            }
          });
          protected_syms.push(Expr::String(sym.clone()));
        }
      }
      return Some(Ok(Expr::List(protected_syms)));
    }
    "Unprotect" => {
      let mut unprotected_syms = Vec::new();
      for arg in args {
        if let Expr::Identifier(sym) = arg {
          let is_locked = {
            let builtin = get_builtin_attributes(sym);
            if builtin.contains(&"Locked") {
              true
            } else {
              crate::FUNC_ATTRS.with(|m| {
                m.borrow()
                  .get(sym.as_str())
                  .is_some_and(|attrs| attrs.contains(&"Locked".to_string()))
              })
            }
          };
          if is_locked {
            crate::emit_message(&format!(
              "Protect::locked: Symbol {} is locked.",
              sym
            ));
            continue;
          }
          // A symbol counts as Protected if either its builtin default
          // attributes or its user-stored attributes contain "Protected".
          let was_user_protected = crate::FUNC_ATTRS.with(|m| {
            let mut attrs = m.borrow_mut();
            if let Some(entry) = attrs.get_mut(sym) {
              let before_len = entry.len();
              entry.retain(|a| a != "Protected");
              before_len != entry.len()
            } else {
              false
            }
          });
          let was_builtin_protected =
            get_builtin_attributes(sym).contains(&"Protected");
          if was_user_protected || was_builtin_protected {
            unprotected_syms.push(Expr::String(sym.clone()));
          }
        }
      }
      return Some(Ok(Expr::List(unprotected_syms)));
    }
    "Clear" => {
      for arg in args {
        match arg {
          Expr::Identifier(sym) => {
            ENV.with(|e| e.borrow_mut().remove(sym));
            crate::FUNC_DEFS.with(|m| m.borrow_mut().remove(sym));
          }
          Expr::String(pattern) => {
            for sym in matching_user_symbols(pattern) {
              ENV.with(|e| e.borrow_mut().remove(&sym));
              crate::FUNC_DEFS.with(|m| m.borrow_mut().remove(&sym));
            }
          }
          _ => {}
        }
      }
      return Some(Ok(Expr::Identifier("Null".to_string())));
    }
    "ClearAll" => {
      let clear_one = |sym: &str| {
        ENV.with(|e| e.borrow_mut().remove(sym));
        crate::FUNC_DEFS.with(|m| m.borrow_mut().remove(sym));
        crate::FUNC_ATTRS.with(|m| m.borrow_mut().remove(sym));
        let up_defs = crate::UPVALUES.with(|m| m.borrow_mut().remove(sym));
        if let Some(up_defs) = up_defs {
          for (
            outer_func,
            params,
            _conds,
            _defaults,
            _heads,
            body,
            _orig_lhs,
            _orig_body,
          ) in &up_defs
          {
            let body_str = expr_to_string(body);
            crate::FUNC_DEFS.with(|m| {
              if let Some(entry) = m.borrow_mut().get_mut(outer_func) {
                entry.retain(|(p, _, _, _, _, b)| {
                  !(p == params && expr_to_string(b) == body_str)
                });
              }
            });
          }
        }
      };
      for arg in args {
        match arg {
          Expr::Identifier(sym) => clear_one(sym),
          Expr::String(pattern) => {
            for sym in matching_user_symbols(pattern) {
              clear_one(&sym);
            }
          }
          _ => {}
        }
      }
      return Some(Ok(Expr::Identifier("Null".to_string())));
    }
    _ => {}
  }
  None
}

/// Resolve a Wolfram-style symbol pattern (e.g. `"Global`*"`, `"x*"`,
/// `"Global`x"`) to the matching user-defined symbols tracked by Woxi.
/// Woxi stores user symbols without a context prefix, so `Global`x` and
/// `x` refer to the same symbol here.
fn matching_user_symbols(pattern: &str) -> Vec<String> {
  let simple_pattern = pattern.strip_prefix("Global`").unwrap_or(pattern);
  // Pre-compute the user-defined symbol list once so we don't borrow
  // ENV/FUNC_DEFS while they are being mutated by the caller.
  let names = crate::get_defined_names();
  if !simple_pattern.contains('*') && !simple_pattern.contains('@') {
    return if names.iter().any(|n| n == simple_pattern) {
      vec![simple_pattern.to_string()]
    } else {
      Vec::new()
    };
  }
  let regex_pattern = format!(
    "^{}$",
    simple_pattern
      .replace('.', "\\.")
      .replace('*', ".*")
      .replace('@', "[a-z]+")
  );
  match regex::Regex::new(&regex_pattern) {
    Ok(re) => names.into_iter().filter(|n| re.is_match(n)).collect(),
    Err(_) => Vec::new(),
  }
}
