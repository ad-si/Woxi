//! `Enclose` and the `Confirm*` family — Wolfram's structured error handling.
//!
//! A `Confirm*` that is not satisfied throws its `Failure[…]` object to the
//! nearest enclosing `Enclose`, which returns it (or hands it to a handler).
//! The throw rides on the ordinary `Throw`/`Catch` machinery under a reserved
//! tag, so an unrelated user `Catch` cannot intercept it.

#[allow(unused_imports)]
use super::*;
use std::cell::Cell;

thread_local! {
  /// How many `Enclose` calls are currently evaluating on this thread. A
  /// `Confirm*` with none around it has nowhere to throw to and reports
  /// `::confirmnotag` instead.
  static ENCLOSE_DEPTH: Cell<u32> = const { Cell::new(0) };
}

/// The reserved tag a confirmation failure is thrown with. The backtick-free
/// name cannot be typed as a symbol, so no user `Catch` can name it.
const CONFIRMATION_TAG: &str = "System`Private`ConfirmationFailure";

pub fn inside_enclose() -> bool {
  ENCLOSE_DEPTH.with(|d| d.get()) > 0
}

/// Runs `body` with the `Enclose` depth raised, restoring it even when the
/// body throws.
pub fn with_enclose_scope<T>(body: impl FnOnce() -> T) -> T {
  ENCLOSE_DEPTH.with(|d| d.set(d.get() + 1));
  let result = body();
  ENCLOSE_DEPTH.with(|d| d.set(d.get().saturating_sub(1)));
  result
}

fn string(s: &str) -> Expr {
  Expr::String(s.to_string())
}

fn symbol(s: &str) -> Expr {
  Expr::Identifier(s.to_string())
}

fn call(name: &str, args: Vec<Expr>) -> Expr {
  Expr::FunctionCall {
    name: name.to_string(),
    args: args.into(),
  }
}

fn failure(tag: &str, pairs: Vec<(&str, Expr)>) -> Expr {
  let assoc =
    Expr::Association(pairs.into_iter().map(|(k, v)| (string(k), v)).collect());
  call("Failure", vec![string(tag), assoc])
}

/// The `Failure[…]` a `Confirm*` throws, carrying the same keys wolframscript
/// puts there so `Enclose[…, "prop"]` and `f["prop"]` see the same data.
fn confirmation_failure(
  kind: &str,
  template: &str,
  parameters: Vec<Expr>,
  expression: Expr,
  information: Expr,
  extra: Vec<(&str, Expr)>,
) -> Expr {
  let mut pairs = vec![
    ("MessageTemplate", string(template)),
    ("MessageParameters", Expr::List(parameters.into())),
    ("ConfirmationType", string(kind)),
    ("Expression", expression),
    ("Information", information),
  ];
  pairs.extend(extra);
  failure("ConfirmationFailed", pairs)
}

/// The failure a `Confirm*` returns when it is evaluated outside any
/// `Enclose`: wolframscript reports `::confirmnotag` and hands back a failure
/// describing the call rather than throwing.
pub fn no_enclose_failure(name: &str, held_call: Expr) -> Expr {
  crate::emit_message(&format!(
    "{name}::confirmnotag: {name} has no tag or surrounding Enclose."
  ));
  failure(
    "confirmnotag",
    vec![
      (
        "MessageTemplate",
        call("MessageName", vec![symbol(name), string("confirmnotag")]),
      ),
      ("MessageParameters", Expr::List(vec![symbol(name)].into())),
      ("HeldInput", call("Hold", vec![held_call])),
    ],
  )
}

/// Values `Confirm` treats as failures.
pub fn is_failure_value(e: &Expr) -> bool {
  matches!(e, Expr::Identifier(s) if s == "$Failed" || s == "$Aborted")
    || matches!(e, Expr::FunctionCall { name, .. }
      if name == "Failure" || name == "Missing")
}

/// Throwing form: the value on success, or an `Err` carrying the failure to
/// the enclosing `Enclose`.
pub fn throw_failure(f: Expr) -> InterpreterError {
  InterpreterError::ThrowValue(
    Box::new(f),
    Some(Box::new(symbol(CONFIRMATION_TAG))),
  )
}

/// True when a caught `Throw` is one of ours.
pub fn is_confirmation_throw(tag: &Option<Box<Expr>>) -> bool {
  matches!(tag.as_deref(), Some(Expr::Identifier(s)) if s == CONFIRMATION_TAG)
}

/// `Confirm[expr]` / `Confirm[expr, info]`.
pub fn confirm_failure(value: &Expr, information: Expr) -> Expr {
  let has_info = !matches!(&information, Expr::Identifier(s) if s == "Null");
  let (template, parameters) = if has_info {
    (crate::syntax::expr_to_output(&information), Vec::new())
  } else {
    ("`` encountered.".to_string(), vec![value.clone()])
  };
  confirmation_failure(
    "Confirm",
    &template,
    parameters,
    value.clone(),
    information,
    Vec::new(),
  )
}

/// `ConfirmBy[expr, f]`.
pub fn confirm_by_failure(value: &Expr, f: &Expr, information: Expr) -> Expr {
  confirmation_failure(
    "ConfirmBy",
    "``[``] did not return True.",
    vec![f.clone(), value.clone()],
    value.clone(),
    information,
    vec![("Function", f.clone())],
  )
}

/// `ConfirmMatch[expr, patt]`.
pub fn confirm_match_failure(
  value: &Expr,
  pattern: &Expr,
  information: Expr,
) -> Expr {
  confirmation_failure(
    "ConfirmMatch",
    "`` does not match ``.",
    vec![value.clone(), pattern.clone()],
    value.clone(),
    information,
    vec![("Pattern", pattern.clone())],
  )
}

/// `ConfirmAssert[test]` — the held test rides along so the failure can show
/// what was asserted.
pub fn confirm_assert_failure(
  held: &Expr,
  evaluated: &Expr,
  information: Expr,
) -> Expr {
  confirmation_failure(
    "ConfirmAssert",
    "Assertion `` failed.",
    vec![call("HoldForm", vec![held.clone()])],
    evaluated.clone(),
    information,
    vec![
      ("Test", evaluated.clone()),
      ("HeldTest", call("Hold", vec![held.clone()])),
    ],
  )
}

/// `Failure[tag, assoc]["prop"]` — the tag, the formatted message, the
/// standard property list, or any key stored in the association.
pub fn failure_property(args: &[Expr], property: &str) -> Option<Expr> {
  let [tag, Expr::Association(pairs)] = args else {
    return None;
  };
  let lookup = |key: &str| -> Option<Expr> {
    pairs
      .iter()
      .find(|(k, _)| matches!(k, Expr::String(s) if s == key))
      .map(|(_, v)| v.clone())
  };
  match property {
    "Tag" => Some(tag.clone()),
    // Everything the failure can answer: the properties every failure has,
    // plus whatever keys its own association carries, sorted.
    "Properties" => {
      let mut names: Vec<String> = [
        "HeldMessageTemplate",
        "Message",
        "MessageName",
        "MessageTemplate",
        "StyledMessage",
        "Tag",
      ]
      .iter()
      .map(|s| s.to_string())
      .collect();
      for (k, _) in pairs.iter() {
        if let Expr::String(key) = k
          && !names.contains(key)
        {
          names.push(key.clone());
        }
      }
      names.sort();
      Some(Expr::List(
        names.iter().map(|p| string(p)).collect::<Vec<_>>().into(),
      ))
    }
    "Message" => {
      let template = match lookup("MessageTemplate") {
        Some(Expr::String(ref s)) => s.clone(),
        _ => {
          return Some(call(
            "Missing",
            vec![string("NotAvailable"), string(property)],
          ));
        }
      };
      let parameters = match lookup("MessageParameters") {
        Some(Expr::List(ref items)) => items.to_vec(),
        _ => Vec::new(),
      };
      Some(Expr::String(fill_template(&template, &parameters)))
    }
    _ => Some(lookup(property).unwrap_or_else(|| {
      call("Missing", vec![string("NotAvailable"), string(property)])
    })),
  }
}

/// Substitute the ``` `` ``` slots of a message template with the parameters,
/// in order. A template with more slots than parameters leaves the extras.
fn fill_template(template: &str, parameters: &[Expr]) -> String {
  let mut out = String::with_capacity(template.len());
  let mut rest = template;
  let mut next = parameters.iter();
  while let Some(at) = rest.find("``") {
    out.push_str(&rest[..at]);
    match next.next() {
      Some(p) => out.push_str(&crate::syntax::expr_to_output(p)),
      None => out.push_str("``"),
    }
    rest = &rest[at + 2..];
  }
  out.push_str(rest);
  out
}

// ---------------------------------------------------------------------------
// Success, Exception
// ---------------------------------------------------------------------------

/// `Success[tag, assoc]["prop"]` — a plain lookup in the association. Unlike
/// `Failure`, `Success` has no computed properties at all: `"Tag"` and
/// `"Message"` are absent keys unless the association happens to hold them,
/// and a missing key reports `KeyAbsent` rather than `NotAvailable`.
pub fn success_property(args: &[Expr], property: &str) -> Option<Expr> {
  let [_tag, Expr::Association(pairs)] = args else {
    return None;
  };
  // "Properties" lists the association's own keys, in the order written.
  if property == "Properties" {
    return Some(Expr::List(
      pairs
        .iter()
        .filter_map(|(k, _)| match k {
          Expr::String(s) => Some(string(s)),
          _ => None,
        })
        .collect::<Vec<_>>()
        .into(),
    ));
  }
  Some(
    pairs
      .iter()
      .find(|(k, _)| matches!(k, Expr::String(s) if s == property))
      .map(|(_, v)| v.clone())
      .unwrap_or_else(|| {
        call("Missing", vec![string("KeyAbsent"), string(property)])
      }),
  )
}

/// `Exception[tags, assoc]["prop"]` — like `Success` a lookup, but an absent
/// key reports `NotAvailable`, as `Failure` does.
pub fn exception_property(args: &[Expr], property: &str) -> Option<Expr> {
  let [_tags, Expr::Association(pairs)] = args else {
    return None;
  };
  Some(
    pairs
      .iter()
      .find(|(k, _)| matches!(k, Expr::String(s) if s == property))
      .map(|(_, v)| v.clone())
      .unwrap_or_else(|| {
        call("Missing", vec![string("NotAvailable"), string(property)])
      }),
  )
}

/// A usable exception tag: a string or a symbol. Anything else has to go
/// through the untagged-exception path.
fn is_exception_tag(e: &Expr) -> bool {
  matches!(e, Expr::String(_) | Expr::Identifier(_))
}

/// True for an already-canonical `Exception[{tags…}, <|…|>]`.
fn is_canonical_exception(e: &Expr) -> bool {
  matches!(e, Expr::FunctionCall { name, args }
    if name == "Exception"
      && args.len() == 2
      && matches!(&args[0], Expr::List(_))
      && matches!(&args[1], Expr::Association(_)))
}

fn exception_object(tags: Vec<Expr>, payload: Option<Expr>) -> Expr {
  let mut pairs: Vec<(Expr, Expr)> = Vec::new();
  if let Some(p) = payload {
    pairs.push((string("ExceptionPayload"), p));
  }
  pairs.push((string("ExceptionValidated"), symbol("True")));
  pairs.push((string("ExceptionSystemVersion"), string("1")));
  Expr::FunctionCall {
    name: "Exception".to_string(),
    args: vec![Expr::List(tags.into()), Expr::Association(pairs)].into(),
  }
}

/// The exception wolframscript builds when the specification is not a tag:
/// a fully-formed `ErrorHandlingException` describing the refusal.
fn untagged_exception(spec: &Expr) -> Expr {
  crate::emit_message(&format!(
    "Exception::untagged: The construction of the untagged exception from \
     general expression {} is not supported. Please provide some exception \
     tag.",
    crate::syntax::expr_to_output(spec)
  ));
  let pairs: Vec<(Expr, Expr)> = vec![
    (string("ErrorType"), string("UnttaggedExceptionPayload")),
    (string("ExceptionFailureTag"), string("ErrorHandlingError")),
    (string("FailingFunction"), symbol("Exception")),
    (string("FailingPayload"), spec.clone()),
    // A delayed entry: the association convention is to store the value as
    // a RuleDelayed keyed by itself, which is what renders as `:>`.
    (
      string("MessageTemplate"),
      Expr::RuleDelayed {
        pattern: Box::new(string("MessageTemplate")),
        replacement: Box::new(call(
          "MessageName",
          vec![symbol("Exception"), string("untagged")],
        )),
      },
    ),
    (
      string("MessageParameters"),
      Expr::List(vec![spec.clone()].into()),
    ),
    (string("ExceptionValidated"), symbol("True")),
    (string("ExceptionSystemVersion"), string("1")),
  ];
  Expr::FunctionCall {
    name: "Exception".to_string(),
    args: vec![
      Expr::List(vec![string("ErrorHandlingException")].into()),
      Expr::Association(pairs),
    ]
    .into(),
  }
}

/// `Exception[spec]` / `Exception[spec, payload]` — canonicalize to
/// `Exception[{tags…}, <|…|>]`. Re-wrapping an exception is a no-op.
pub fn exception_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  // wolframscript echoes a bare `Exception[]` without complaining.
  if args.is_empty() {
    return Ok(unevaluated("Exception", args));
  }
  if args.len() == 1 && is_canonical_exception(&args[0]) {
    return Ok(args[0].clone());
  }
  let spec = &args[0];
  let tags: Option<Vec<Expr>> = match spec {
    Expr::List(items) if items.iter().all(is_exception_tag) => {
      Some(items.to_vec())
    }
    e if is_exception_tag(e) => Some(vec![e.clone()]),
    _ => None,
  };
  let Some(tags) = tags else {
    return Ok(untagged_exception(spec));
  };
  Ok(exception_object(tags, args.get(1).cloned()))
}

/// `ExceptionQ[expr]` — True for a canonical exception. `ExceptionQ[expr, t]`
/// additionally requires `t` to be one of its tags.
pub fn exception_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let mut ok = is_canonical_exception(&args[0]);
  if ok && args.len() == 2 {
    let Expr::FunctionCall { args: eargs, .. } = &args[0] else {
      unreachable!("checked by is_canonical_exception")
    };
    let Expr::List(tags) = &eargs[0] else {
      unreachable!("checked by is_canonical_exception")
    };
    ok = tags.iter().any(|t| {
      crate::syntax::expr_to_string(t)
        == crate::syntax::expr_to_string(&args[1])
    });
  }
  Ok(Expr::Identifier(
    if ok { "True" } else { "False" }.to_string(),
  ))
}

/// `ExceptionTypes[]` — the registry of exception types, which starts empty
/// and has no registration interface yet, so it is always empty.
pub fn exception_types_ast(_args: &[Expr]) -> Result<Expr, InterpreterError> {
  Ok(Expr::List(Vec::new().into()))
}

/// `ExceptionTypeRegisteredQ[sym]` — nothing is registered, so always False.
pub fn exception_type_registered_q_ast(
  _args: &[Expr],
) -> Result<Expr, InterpreterError> {
  Ok(Expr::Identifier("False".to_string()))
}
