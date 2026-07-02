//! Symbolic `HTTPRequest` objects.
//!
//! `HTTPRequest` builds a symbolic request object; no network access is
//! performed (sending a request is `URLRead`'s job, which Woxi does not
//! implement). The object canonicalizes to `HTTPRequest[url, assoc]` and
//! supports property extraction like `req["Method"]` or `req["Domain"]`.

use crate::InterpreterError;
use crate::syntax::Expr;

/// The components of a request URL, either parsed from a URL string or
/// supplied explicitly through the request association.
#[derive(Default)]
struct UrlParts {
  scheme: Option<String>,
  user: Option<String>,
  password: Option<String>,
  domain: Option<String>,
  port: Option<i128>,
  path: String,
  query: Vec<(String, String)>,
  fragment: Option<String>,
}

/// `HTTPRequest[url]`, `HTTPRequest[url, assoc]`, `HTTPRequest[assoc]` —
/// construct the symbolic request object.
///
/// Canonicalization mirrors wolframscript: the one-argument URL form gains
/// an empty association (`HTTPRequest["url"]` → `HTTPRequest["url", <||>]`)
/// and a `URL["…"]` wrapper is unwrapped to the plain URL string. The
/// association forms are already canonical and stay as given.
pub fn http_request_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let rebuild = |new_args: Vec<Expr>| Expr::FunctionCall {
    name: "HTTPRequest".to_string(),
    args: new_args.into(),
  };
  match args.len() {
    1 => match unwrap_url(&args[0]) {
      url @ Expr::String(_) => {
        Ok(rebuild(vec![url, Expr::Association(vec![])]))
      }
      other => Ok(rebuild(vec![other])),
    },
    2 => Ok(rebuild(vec![unwrap_url(&args[0]), args[1].clone()])),
    _ => Ok(rebuild(args.to_vec())),
  }
}

/// `HTTPRequest[…]["property"]` — extract a property of the request object.
/// Returns `None` for properties Woxi does not resolve, so the caller can
/// keep the curried form unevaluated.
pub fn http_request_property(func_args: &[Expr], prop: &str) -> Option<Expr> {
  let (url, assoc): (Option<&str>, Option<&[(Expr, Expr)]>) = match func_args {
    [Expr::String(u)] => (Some(u.as_str()), None),
    [Expr::Association(pairs)] => (None, Some(pairs)),
    [Expr::String(u), Expr::Association(pairs)] => {
      (Some(u.as_str()), Some(pairs))
    }
    _ => return None,
  };
  // The association may carry the URL itself (`<|"URL" -> "…"|>`).
  let base_url = url.or_else(|| match assoc.and_then(|a| assoc_get(a, "URL")) {
    Some(Expr::String(u)) => Some(u.as_str()),
    _ => None,
  });
  let mut parts = base_url.map(parse_url).unwrap_or_default();
  if let Some(pairs) = assoc {
    apply_assoc_overrides(&mut parts, pairs);
  }

  let opt_str = |v: &Option<String>| match v {
    Some(s) => Expr::String(s.clone()),
    None => Expr::Identifier("None".to_string()),
  };
  match prop {
    "URL" => Some(Expr::String(build_url(&parts))),
    "Scheme" => Some(opt_str(&parts.scheme)),
    "UserName" => Some(opt_str(&parts.user)),
    "Password" => Some(opt_str(&parts.password)),
    "Domain" => Some(opt_str(&parts.domain)),
    "Port" => Some(match parts.port {
      Some(p) => Expr::Integer(p),
      None => Expr::Identifier("None".to_string()),
    }),
    "Path" => Some(Expr::String(parts.path.clone())),
    "Query" => Some(query_to_expr(&parts.query)),
    "Fragment" => Some(opt_str(&parts.fragment)),
    "Method" => Some(match assoc.and_then(|a| assoc_get(a, "Method")) {
      Some(Expr::String(m)) => Expr::String(m.clone()),
      _ => Expr::String("GET".to_string()),
    }),
    "Headers" => Some(match assoc.and_then(|a| assoc_get(a, "Headers")) {
      Some(Expr::Association(pairs)) => Expr::List(
        pairs
          .iter()
          .map(|(k, v)| Expr::Rule {
            pattern: Box::new(k.clone()),
            replacement: Box::new(v.clone()),
          })
          .collect(),
      ),
      Some(list @ Expr::List(_)) => list.clone(),
      Some(rule @ Expr::Rule { .. }) => Expr::List(vec![rule.clone()].into()),
      _ => Expr::List(vec![].into()),
    }),
    "Body" => Some(match assoc.and_then(|a| assoc_get(a, "Body")) {
      Some(body) => body.clone(),
      None => Expr::String(String::new()),
    }),
    _ => None,
  }
}

/// Unwrap a `URL["…"]` wrapper to its plain URL string; anything else
/// passes through unchanged.
fn unwrap_url(expr: &Expr) -> Expr {
  if let Expr::FunctionCall { name, args } = expr
    && name == "URL"
    && args.len() == 1
    && matches!(&args[0], Expr::String(_))
  {
    return args[0].clone();
  }
  expr.clone()
}

/// Look up a string key in association pairs.
fn assoc_get<'a>(pairs: &'a [(Expr, Expr)], key: &str) -> Option<&'a Expr> {
  pairs.iter().find_map(|(k, v)| match k {
    Expr::String(s) if s == key => Some(v),
    _ => None,
  })
}

/// Parse a URL string into its components. The authority part (userinfo,
/// domain, port) is only recognized when a `scheme://` prefix is present;
/// otherwise the whole remainder is a path, matching `URLParse`.
fn parse_url(url: &str) -> UrlParts {
  let mut parts = UrlParts::default();
  let mut rest = url;
  let has_authority = if let Some(idx) = rest.find("://") {
    parts.scheme = Some(rest[..idx].to_string());
    rest = &rest[idx + 3..];
    true
  } else {
    false
  };
  if let Some(idx) = rest.find('#') {
    parts.fragment = Some(rest[idx + 1..].to_string());
    rest = &rest[..idx];
  }
  if let Some(idx) = rest.find('?') {
    parts.query = parse_query(&rest[idx + 1..]);
    rest = &rest[..idx];
  }
  if !has_authority {
    parts.path = rest.to_string();
    return parts;
  }
  let (authority, path) = match rest.find('/') {
    Some(idx) => (&rest[..idx], &rest[idx..]),
    None => (rest, ""),
  };
  parts.path = path.to_string();
  let mut auth = authority;
  if let Some(idx) = auth.rfind('@') {
    let userinfo = &auth[..idx];
    auth = &auth[idx + 1..];
    match userinfo.split_once(':') {
      Some((u, p)) => {
        parts.user = Some(u.to_string());
        parts.password = Some(p.to_string());
      }
      None => parts.user = Some(userinfo.to_string()),
    }
  }
  match auth.rsplit_once(':') {
    Some((host, port))
      if !port.is_empty() && port.bytes().all(|b| b.is_ascii_digit()) =>
    {
      parts.domain = non_empty(host);
      parts.port = port.parse().ok();
    }
    _ => parts.domain = non_empty(auth),
  }
  parts
}

fn non_empty(s: &str) -> Option<String> {
  if s.is_empty() {
    None
  } else {
    Some(s.to_string())
  }
}

/// Parse `a=1&b=2` into key/value pairs. A key without `=` maps to "".
fn parse_query(query: &str) -> Vec<(String, String)> {
  query
    .split('&')
    .filter(|kv| !kv.is_empty())
    .map(|kv| match kv.split_once('=') {
      Some((k, v)) => (k.to_string(), v.to_string()),
      None => (kv.to_string(), String::new()),
    })
    .collect()
}

/// Fold explicit URL components from the request association into `parts`.
fn apply_assoc_overrides(parts: &mut UrlParts, pairs: &[(Expr, Expr)]) {
  let as_opt_string = |v: &Expr| -> Option<Option<String>> {
    match v {
      Expr::String(s) => Some(Some(s.clone())),
      Expr::Identifier(n) if n == "None" => Some(None),
      _ => None,
    }
  };
  for (key, value) in pairs {
    let Expr::String(key) = key else { continue };
    match key.as_str() {
      "Scheme" => {
        if let Some(v) = as_opt_string(value) {
          parts.scheme = v;
        }
      }
      "UserName" => {
        if let Some(v) = as_opt_string(value) {
          parts.user = v;
        }
      }
      "Password" => {
        if let Some(v) = as_opt_string(value) {
          parts.password = v;
        }
      }
      "Domain" => {
        if let Some(v) = as_opt_string(value) {
          parts.domain = v;
        }
      }
      "Port" => match value {
        Expr::Integer(p) => parts.port = Some(*p),
        Expr::Identifier(n) if n == "None" => parts.port = None,
        _ => {}
      },
      "Path" => {
        if let Expr::String(p) = value {
          parts.path = p.clone();
        }
      }
      "Fragment" => {
        if let Some(v) = as_opt_string(value) {
          parts.fragment = v;
        }
      }
      "Query" => match value {
        Expr::String(q) => parts.query = parse_query(q),
        Expr::List(items) => {
          parts.query = items
            .iter()
            .filter_map(|item| match item {
              Expr::Rule {
                pattern,
                replacement,
              }
              | Expr::RuleDelayed {
                pattern,
                replacement,
              } => Some((
                rule_side_to_string(pattern),
                rule_side_to_string(replacement),
              )),
              _ => None,
            })
            .collect();
        }
        Expr::Association(qpairs) => {
          parts.query = qpairs
            .iter()
            .map(|(k, v)| (rule_side_to_string(k), rule_side_to_string(v)))
            .collect();
        }
        _ => {}
      },
      _ => {}
    }
  }
}

/// Render a query key or value: strings verbatim, anything else in
/// canonical form (so `"n" -> 1` yields `n=1`).
fn rule_side_to_string(expr: &Expr) -> String {
  match expr {
    Expr::String(s) => s.clone(),
    other => crate::syntax::expr_to_string(other),
  }
}

/// Reassemble the URL string from its components.
fn build_url(parts: &UrlParts) -> String {
  let mut url = String::new();
  if let Some(scheme) = &parts.scheme {
    url.push_str(scheme);
    url.push_str("://");
  }
  if parts.domain.is_some() {
    if let Some(user) = &parts.user {
      url.push_str(user);
      if let Some(password) = &parts.password {
        url.push(':');
        url.push_str(password);
      }
      url.push('@');
    }
  }
  if let Some(domain) = &parts.domain {
    url.push_str(domain);
  }
  if let Some(port) = parts.port {
    url.push(':');
    url.push_str(&port.to_string());
  }
  url.push_str(&parts.path);
  if !parts.query.is_empty() {
    url.push('?');
    for (i, (k, v)) in parts.query.iter().enumerate() {
      if i > 0 {
        url.push('&');
      }
      url.push_str(k);
      url.push('=');
      url.push_str(v);
    }
  }
  if let Some(fragment) = &parts.fragment {
    url.push('#');
    url.push_str(fragment);
  }
  url
}

/// Render query pairs as the `{"a" -> "1", …}` list of rules.
fn query_to_expr(query: &[(String, String)]) -> Expr {
  Expr::List(
    query
      .iter()
      .map(|(k, v)| Expr::Rule {
        pattern: Box::new(Expr::String(k.clone())),
        replacement: Box::new(Expr::String(v.clone())),
      })
      .collect(),
  )
}
