//! Symbolic `HTTPRequest` objects and `URLRead`.
//!
//! `HTTPRequest` builds a symbolic request object; no network access is
//! performed until the request is sent with `URLRead`. The object
//! canonicalizes to `HTTPRequest[url, assoc]` and supports property
//! extraction like `req["Method"]`, `req[Method]`, or
//! `req[{"Scheme", "Domain", Method}]`.

use crate::InterpreterError;
use crate::syntax::Expr;

/// The user agent wolframscript sends and reports for HTTP requests.
const USER_AGENT: &str = "Wolfram HTTPClient 15.";

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

/// The properties wolframscript reports for `req["Properties"]`, in its
/// order. All entries are strings except the trailing `Method` symbol.
const PROPERTY_NAMES: [&str; 22] = [
  "AbsoluteDomain",
  "AbsolutePath",
  "Body",
  "BodyByteArray",
  "BodyBytes",
  "ContentType",
  "Cookies",
  "Domain",
  "FormRules",
  "Fragment",
  "Headers",
  "Password",
  "Path",
  "PathString",
  "Port",
  "Query",
  "QueryString",
  "Scheme",
  "URL",
  "User",
  "UserAgent",
  "Username",
];

/// `HTTPRequest[…][prop]` — extract one or several properties of the
/// request object. `prop` may be a property string, the `Method` symbol, or
/// a list of those (which yields an association keyed by the given specs,
/// with `$Failed` for unknown entries). Unknown properties emit
/// `HTTPRequest::notprop`; a `None` return keeps the curried form
/// unevaluated.
pub fn http_request_extract(func_args: &[Expr], arg: &Expr) -> Option<Expr> {
  if !matches!(
    func_args,
    [Expr::String(_)]
      | [Expr::Association(_)]
      | [Expr::String(_), Expr::Association(_)]
  ) {
    return None;
  }
  let resolve = |spec: &Expr| -> Option<Expr> {
    match spec {
      Expr::String(p) => http_request_property(func_args, p),
      Expr::Identifier(p) if p == "Method" => {
        http_request_property(func_args, "Method")
      }
      _ => None,
    }
  };
  match arg {
    Expr::List(items) => Some(Expr::Association(
      items
        .iter()
        .map(|item| {
          let value = resolve(item).unwrap_or_else(|| {
            emit_notprop(item, func_args.len());
            Expr::Identifier("$Failed".to_string())
          });
          (item.clone(), value)
        })
        .collect(),
    )),
    Expr::String(_) | Expr::Identifier(_) => match resolve(arg) {
      Some(value) => Some(value),
      None => {
        emit_notprop(arg, func_args.len());
        None
      }
    },
    _ => None,
  }
}

/// Emit the `HTTPRequest::notprop` message for an unknown property spec.
fn emit_notprop(spec: &Expr, arg_count: usize) {
  let name = match spec {
    Expr::String(s) => s.clone(),
    other => crate::syntax::expr_to_string(other),
  };
  crate::emit_message_to_stdout(&format!(
    "HTTPRequest::notprop: {name} is not a known property for \
     HTTPRequest[«{arg_count}»]. Use HTTPRequest[«{arg_count}»][\"Properties\"] \
     for a list of properties."
  ));
}

/// `HTTPRequest[…]["property"]` — extract a property of the request object.
/// Returns `None` for unknown properties.
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
  let base_url =
    url.or_else(|| match assoc.and_then(|a| assoc_get(a, "URL")) {
      Some(Expr::String(u)) => Some(u.as_str()),
      _ => None,
    });
  let mut parts = base_url.map(parse_url).unwrap_or_default();
  if let Some(pairs) = assoc {
    apply_assoc_overrides(&mut parts, pairs);
  }
  // `Body` distinguishes "key present with a string value" (even an empty
  // one — it still sets the default ContentType) from "absent".
  let body: Option<&str> = match assoc.and_then(|a| assoc_get(a, "Body")) {
    Some(Expr::String(b)) => Some(b.as_str()),
    _ => None,
  };
  let headers = request_headers(assoc);

  let opt_str = |v: &Option<String>| match v {
    Some(s) => Expr::String(s.clone()),
    None => Expr::Identifier("None".to_string()),
  };
  let none = || Expr::Identifier("None".to_string());
  match prop {
    "URL" => Some(Expr::String(build_url(&parts))),
    "Scheme" => Some(opt_str(&parts.scheme)),
    "User" => Some(match (&parts.user, &parts.password) {
      (Some(u), Some(p)) => Expr::String(format!("{u}:{p}")),
      (Some(u), None) => Expr::String(u.clone()),
      (None, _) => none(),
    }),
    "Username" => Some(opt_str(&parts.user)),
    "Password" => Some(opt_str(&parts.password)),
    "Domain" => Some(opt_str(&parts.domain)),
    "AbsoluteDomain" => Some(Expr::String(absolute_domain(&parts))),
    "AbsolutePath" => Some(Expr::String(format!(
      "{}{}",
      absolute_domain(&parts),
      parts.path
    ))),
    "Port" => Some(match parts.port {
      Some(p) => Expr::Integer(p),
      None => none(),
    }),
    // `Path` is the URLParse-style segment list; the raw string is
    // `PathString` (an empty path has no segments at all).
    "Path" => Some(Expr::List(if parts.path.is_empty() {
      vec![].into()
    } else {
      parts
        .path
        .split('/')
        .map(|s| Expr::String(s.to_string()))
        .collect()
    })),
    "PathString" => Some(Expr::String(parts.path.clone())),
    "Query" => Some(query_to_expr(&parts.query)),
    "QueryString" => Some(if parts.query.is_empty() {
      none()
    } else {
      Expr::String(
        parts
          .query
          .iter()
          .map(|(k, v)| format!("{k}={v}"))
          .collect::<Vec<_>>()
          .join("&"),
      )
    }),
    "Fragment" => Some(opt_str(&parts.fragment)),
    "Method" => Some(match assoc.and_then(|a| assoc_get(a, "Method")) {
      Some(Expr::String(m)) => Expr::String(m.clone()),
      _ => Expr::String("GET".to_string()),
    }),
    "Headers" => Some(Expr::List(
      headers
        .iter()
        .map(|(name, value)| Expr::Rule {
          pattern: Box::new(Expr::String(name.clone())),
          replacement: Box::new(value.clone()),
        })
        .collect(),
    )),
    "UserAgent" => headers
      .iter()
      .find(|(name, _)| name == "user-agent")
      .map(|(_, value)| value.clone()),
    "ContentType" => Some(
      match headers.iter().find(|(name, _)| name == "content-type") {
        Some((_, value)) => value.clone(),
        None if body.is_some() => {
          Expr::String("text/plain;charset=utf-8".to_string())
        }
        None => none(),
      },
    ),
    "Body" => Some(match assoc.and_then(|a| assoc_get(a, "Body")) {
      Some(body) => body.clone(),
      None => Expr::String(String::new()),
    }),
    "BodyBytes" => Some(Expr::List(
      body
        .unwrap_or_default()
        .bytes()
        .map(|b| Expr::Integer(b as i128))
        .collect(),
    )),
    "BodyByteArray" => {
      use base64::Engine as _;
      Some(Expr::FunctionCall {
        name: "ByteArray".to_string(),
        args: vec![Expr::String(
          base64::engine::general_purpose::STANDARD
            .encode(body.unwrap_or_default()),
        )]
        .into(),
      })
    }
    "Cookies" => Some(Expr::Identifier("Automatic".to_string())),
    "FormRules" => Some(none()),
    "Properties" => Some(Expr::List(
      PROPERTY_NAMES
        .iter()
        .map(|name| Expr::String(name.to_string()))
        .chain(std::iter::once(Expr::Identifier("Method".to_string())))
        .collect(),
    )),
    _ => None,
  }
}

/// The request headers as lowercase-name/value pairs: the explicitly given
/// headers in order, with wolframscript's default user agent appended when
/// none is given.
fn request_headers(assoc: Option<&[(Expr, Expr)]>) -> Vec<(String, Expr)> {
  let name_of = |k: &Expr| -> Option<String> {
    match k {
      Expr::String(s) => Some(s.to_lowercase()),
      Expr::Identifier(s) => Some(s.to_lowercase()),
      _ => None,
    }
  };
  let mut headers: Vec<(String, Expr)> = Vec::new();
  match assoc.and_then(|a| assoc_get(a, "Headers")) {
    Some(Expr::Association(pairs)) => {
      for (k, v) in pairs {
        if let Some(name) = name_of(k) {
          headers.push((name, v.clone()));
        }
      }
    }
    Some(Expr::List(items)) => {
      for item in items.iter() {
        if let Expr::Rule {
          pattern,
          replacement,
        } = item
          && let Some(name) = name_of(pattern)
        {
          headers.push((name, (**replacement).clone()));
        }
      }
    }
    Some(Expr::Rule {
      pattern,
      replacement,
    }) => {
      if let Some(name) = name_of(pattern) {
        headers.push((name, (**replacement).clone()));
      }
    }
    _ => {}
  }
  if !headers.iter().any(|(name, _)| name == "user-agent") {
    headers.push((
      "user-agent".to_string(),
      Expr::String(USER_AGENT.to_string()),
    ));
  }
  headers
}

/// `scheme://[user[:password]@]domain[:port]` — empty when there is no
/// scheme (matching wolframscript's `AbsoluteDomain` for schemeless URLs).
fn absolute_domain(parts: &UrlParts) -> String {
  if parts.scheme.is_none() {
    return String::new();
  }
  let no_suffix = UrlParts {
    scheme: parts.scheme.clone(),
    user: parts.user.clone(),
    password: parts.password.clone(),
    domain: parts.domain.clone(),
    port: parts.port,
    ..Default::default()
  };
  build_url(&no_suffix)
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

/// Look up a key in association pairs. Both string keys (`"Method"`) and
/// symbol keys (`Method`, as in `<|Method -> "POST"|>`) are accepted.
fn assoc_get<'a>(pairs: &'a [(Expr, Expr)], key: &str) -> Option<&'a Expr> {
  pairs.iter().find_map(|(k, v)| match k {
    Expr::String(s) | Expr::Identifier(s) if s == key => Some(v),
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
  if parts.domain.is_some()
    && let Some(user) = &parts.user
  {
    url.push_str(user);
    if let Some(password) = &parts.password {
      url.push(':');
      url.push_str(password);
    }
    url.push('@');
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

/// `URLRead[req]` — send the request and return the `HTTPResponse` object.
///
/// The request is sent through `curl` (the same HTTP engine wolframscript
/// uses via libcurl), following redirects and transparently decompressing
/// the body while keeping the response headers as received. Connection
/// failures emit `URLRead::invhttp` and return the same
/// `Failure["ConnectionFailure", …]` object wolframscript produces.
#[cfg(not(target_arch = "wasm32"))]
pub fn url_read_ast(arg: &Expr) -> Result<Expr, InterpreterError> {
  use base64::Engine as _;
  let unevaluated = || {
    Ok(Expr::FunctionCall {
      name: "URLRead".to_string(),
      args: vec![arg.clone()].into(),
    })
  };
  let func_args: Vec<Expr> = match &unwrap_url(arg) {
    Expr::String(u) => vec![Expr::String(u.clone())],
    Expr::FunctionCall { name, args } if name == "HTTPRequest" => args.to_vec(),
    _ => return unevaluated(),
  };
  let url = match http_request_property(&func_args, "URL").as_ref() {
    Some(Expr::String(u)) => u.clone(),
    _ => return unevaluated(),
  };
  let method = match http_request_property(&func_args, "Method").as_ref() {
    Some(Expr::String(m)) => m.clone(),
    _ => "GET".to_string(),
  };
  let assoc = func_args.iter().find_map(|a| match a {
    Expr::Association(pairs) => Some(pairs.as_slice()),
    _ => None,
  });
  let body = match assoc.and_then(|a| assoc_get(a, "Body")) {
    Some(Expr::String(b)) => b.clone(),
    _ => String::new(),
  };

  let mut cmd = std::process::Command::new("curl");
  cmd.args(["-s", "-S", "-i", "-L", "--compressed", "--max-time", "60"]);
  if method != "GET" {
    cmd.args(["-X", &method]);
  }
  for (name, value) in request_headers(assoc) {
    let value = match &value {
      Expr::String(s) => s.clone(),
      other => crate::syntax::expr_to_string(other),
    };
    cmd.arg("-H").arg(format!("{name}: {value}"));
  }
  if !body.is_empty() {
    cmd.arg("--data-binary").arg(&body);
  }
  cmd.arg(&url);
  let output = cmd.output().map_err(|e| {
    InterpreterError::EvaluationError(format!(
      "URLRead: failed to run curl: {e}"
    ))
  })?;

  if !output.status.success() {
    let stderr = String::from_utf8_lossy(&output.stderr);
    crate::emit_message_to_stdout(&format!(
      "URLRead::invhttp: {}",
      curl_error_text(&stderr)
    ));
    return Ok(connection_failure(&url, &func_args));
  }

  let Some((headers, status, response_body)) = parse_response(&output.stdout)
  else {
    return unevaluated();
  };
  Ok(Expr::FunctionCall {
    name: "HTTPResponse".to_string(),
    args: vec![
      Expr::FunctionCall {
        name: "ByteArray".to_string(),
        args: vec![Expr::String(
          base64::engine::general_purpose::STANDARD.encode(response_body),
        )]
        .into(),
      },
      Expr::Association(vec![
        (
          Expr::String("Headers".to_string()),
          Expr::List(
            headers
              .into_iter()
              .map(|(name, value)| {
                Expr::List(vec![Expr::String(name), Expr::String(value)].into())
              })
              .collect(),
          ),
        ),
        (
          Expr::String("StatusCode".to_string()),
          Expr::Integer(status),
        ),
        (
          Expr::String("Cookies".to_string()),
          Expr::List(vec![].into()),
        ),
      ]),
      Expr::Rule {
        pattern: Box::new(Expr::Identifier("CharacterEncoding".to_string())),
        replacement: Box::new(Expr::Identifier("Automatic".to_string())),
      },
    ]
    .into(),
  })
}

/// The human-readable part of a curl error line: `curl: (6) Could not
/// resolve host: x` becomes `Could not resolve host: x.` (wolframscript
/// reports libcurl's message with a trailing period).
#[cfg(not(target_arch = "wasm32"))]
fn curl_error_text(stderr: &str) -> String {
  let line = stderr.lines().find(|l| !l.trim().is_empty()).unwrap_or("");
  let mut text = line.trim();
  if let Some(rest) = text.strip_prefix("curl: (")
    && let Some(idx) = rest.find(") ")
  {
    text = &rest[idx + 2..];
  }
  let mut text = text.trim().to_string();
  if text.is_empty() {
    text = "Connection failure".to_string();
  }
  if !text.ends_with('.') {
    text.push('.');
  }
  text
}

/// The `Failure["ConnectionFailure", …]` object URLRead returns when the
/// request cannot be sent, mirroring wolframscript's structure.
#[cfg(not(target_arch = "wasm32"))]
fn connection_failure(url: &str, func_args: &[Expr]) -> Expr {
  let request = http_request_ast(func_args).unwrap_or(Expr::FunctionCall {
    name: "HTTPRequest".to_string(),
    args: func_args.to_vec().into(),
  });
  let template_key = Expr::String("MessageTemplate".to_string());
  Expr::FunctionCall {
    name: "Failure".to_string(),
    args: vec![
      Expr::String("ConnectionFailure".to_string()),
      Expr::Association(vec![
        (
          template_key.clone(),
          // `:>` entry: the association convention stores the delayed
          // value as the full RuleDelayed.
          Expr::RuleDelayed {
            pattern: Box::new(template_key),
            replacement: Box::new(Expr::FunctionCall {
              name: "MessageName".to_string(),
              args: vec![
                Expr::Identifier("URLRead".to_string()),
                Expr::String("iurl".to_string()),
              ]
              .into(),
            }),
          },
        ),
        (
          Expr::String("MessageParameters".to_string()),
          Expr::List(vec![Expr::String(url.to_string())].into()),
        ),
        (
          Expr::String("URL".to_string()),
          Expr::String(url.to_string()),
        ),
        (Expr::String("HTTPRequest".to_string()), request),
        (Expr::String("Counter".to_string()), Expr::Integer(1)),
      ]),
    ]
    .into(),
  }
}

/// Split a raw `curl -i` capture into the final response's headers, status
/// code, and body. Informational (1xx) and redirect blocks followed by
/// another `HTTP/` block are skipped, so only the final response counts.
#[cfg(not(target_arch = "wasm32"))]
fn parse_response(data: &[u8]) -> Option<(Vec<(String, String)>, i128, &[u8])> {
  let mut rest = data;
  loop {
    if !rest.starts_with(b"HTTP/") {
      return None;
    }
    let sep = rest.windows(4).position(|w| w == b"\r\n\r\n")?;
    let block = String::from_utf8_lossy(&rest[..sep]).into_owned();
    let after = &rest[sep + 4..];
    let mut lines = block.lines();
    let status: i128 = lines.next()?.split_whitespace().nth(1)?.parse().ok()?;
    let informational = (100..200).contains(&status);
    let redirected =
      (300..400).contains(&status) && after.starts_with(b"HTTP/");
    if informational || redirected {
      rest = after;
      continue;
    }
    let headers = lines
      .filter_map(|line| {
        line.split_once(':').map(|(name, value)| {
          (name.to_string(), value.trim_start().to_string())
        })
      })
      .collect();
    return Some((headers, status, after));
  }
}

/// The URL components of one parsed `URLParse` input. Unlike `UrlParts`
/// (whose query is pre-split), the path and query keep their raw text so
/// `PathString`/`QueryString` can report them exactly as given while
/// `Path`/`Query` percent-decode.
#[derive(Default)]
struct ParsedUrl {
  scheme: Option<String>,
  user: Option<String>,
  password: Option<String>,
  domain: Option<String>,
  port: Option<i128>,
  raw_path: String,
  raw_query: Option<String>,
  fragment: Option<String>,
}

/// Parse a URL string for `URLParse`. Handles opaque scheme forms like
/// `mailto:user@example.com` and protocol-relative `//host/path` in addition
/// to `scheme://host` URLs; the scheme and domain are lowercased, everything
/// else keeps its case. Returns `Err(port_text)` for a non-numeric port,
/// which `URLParse` reports as `::nvldval` with `$Failed`.
fn parse_url_components(url: &str) -> Result<ParsedUrl, String> {
  let mut out = ParsedUrl::default();
  let mut rest = url;
  if let Some(idx) = rest.find('#') {
    out.fragment = Some(rest[idx + 1..].to_string());
    rest = &rest[..idx];
  }
  if let Some(idx) = rest.find('?') {
    out.raw_query = Some(rest[idx + 1..].to_string());
    rest = &rest[..idx];
  }
  if let Some(idx) = rest.find(':') {
    let candidate = &rest[..idx];
    let valid_scheme = !candidate.is_empty()
      && candidate
        .chars()
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic())
      && candidate
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '-' | '.'));
    if valid_scheme {
      out.scheme = Some(candidate.to_ascii_lowercase());
      rest = &rest[idx + 1..];
    }
  }
  if let Some(after) = rest.strip_prefix("//") {
    let (authority, path) = match after.find('/') {
      Some(idx) => (&after[..idx], &after[idx..]),
      None => (after, ""),
    };
    out.raw_path = path.to_string();
    let mut auth = authority;
    if let Some(idx) = auth.rfind('@') {
      let userinfo = &auth[..idx];
      auth = &auth[idx + 1..];
      match userinfo.split_once(':') {
        Some((u, p)) => {
          out.user = Some(u.to_string());
          out.password = Some(p.to_string());
        }
        None => out.user = Some(userinfo.to_string()),
      }
    }
    match auth.rsplit_once(':') {
      Some((host, port)) => {
        if port.is_empty() || !port.bytes().all(|b| b.is_ascii_digit()) {
          return Err(port.to_string());
        }
        out.domain = non_empty(&host.to_ascii_lowercase());
        out.port = port.parse().ok();
      }
      None => out.domain = non_empty(&auth.to_ascii_lowercase()),
    }
  } else {
    out.raw_path = rest.to_string();
  }
  Ok(out)
}

/// Decode one `key` or `value` of a query string: `+` means space, and
/// percent-escapes decode to bytes (`%2B` is how a literal `+` survives).
fn decode_query_component(s: &str) -> String {
  crate::functions::string_ast::percent_decode(&s.replace('+', " "))
}

/// URLParse[url] — parse a URL into an association of its components.
/// URLParse[url, "component"] / URLParse[url, {"c1", "c2"}] / URLParse[url,
/// All] return the given component(s) only. `Path` segments and `Query`
/// keys/values are percent-decoded; `Fragment`, `PathString`, and
/// `QueryString` stay raw, matching wolframscript.
pub fn url_parse_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || {
    Ok(Expr::FunctionCall {
      name: "URLParse".to_string(),
      args: args.to_vec().into(),
    })
  };
  if args.is_empty() || args.len() > 2 {
    return unevaluated();
  }

  let unwrapped = unwrap_url(&args[0]);
  let url = match &unwrapped {
    Expr::String(s) => s.clone(),
    other => {
      crate::emit_message(&format!(
        "URLParse::invuri: The URI {} is invalid.",
        crate::syntax::expr_to_output(other)
      ));
      return unevaluated();
    }
  };
  let parsed = match parse_url_components(&url) {
    Ok(p) => p,
    Err(port) => {
      crate::emit_message(&format!(
        "URLParse::nvldval: {} is not a valid value",
        port
      ));
      return Ok(Expr::Identifier("$Failed".to_string()));
    }
  };

  let none = || Expr::Identifier("None".to_string());
  let opt_str = |v: &Option<String>| match v {
    Some(s) => Expr::String(s.clone()),
    None => none(),
  };
  let component_value = |name: &str| -> Option<Expr> {
    match name {
      "Scheme" => Some(opt_str(&parsed.scheme)),
      "User" => Some(match (&parsed.user, &parsed.password) {
        (Some(u), Some(p)) => Expr::String(format!("{u}:{p}")),
        (Some(u), None) => Expr::String(u.clone()),
        (None, _) => none(),
      }),
      "Domain" => Some(opt_str(&parsed.domain)),
      "Port" => Some(match parsed.port {
        Some(p) => Expr::Integer(p),
        None => none(),
      }),
      "Path" => Some(Expr::List(if parsed.raw_path.is_empty() {
        vec![].into()
      } else {
        parsed
          .raw_path
          .split('/')
          .map(|s| {
            Expr::String(crate::functions::string_ast::percent_decode(s))
          })
          .collect()
      })),
      "Query" => Some(Expr::List(
        parsed
          .raw_query
          .as_deref()
          .unwrap_or("")
          .split('&')
          .filter(|kv| !kv.is_empty())
          .map(|kv| {
            let (k, v) = kv.split_once('=').unwrap_or((kv, ""));
            Expr::Rule {
              pattern: Box::new(Expr::String(decode_query_component(k))),
              replacement: Box::new(Expr::String(decode_query_component(v))),
            }
          })
          .collect(),
      )),
      "Fragment" => Some(opt_str(&parsed.fragment)),
      "PathString" => Some(Expr::String(parsed.raw_path.clone())),
      "QueryString" => Some(match &parsed.raw_query {
        Some(q) => Expr::String(q.clone()),
        None => none(),
      }),
      "Username" => Some(opt_str(&parsed.user)),
      "Password" => Some(opt_str(&parsed.password)),
      "AbsolutePath" | "AbsoluteDomain" => {
        let domain_parts = UrlParts {
          scheme: parsed.scheme.clone(),
          user: parsed.user.clone(),
          password: parsed.password.clone(),
          domain: parsed.domain.clone(),
          port: parsed.port,
          ..Default::default()
        };
        let mut s = absolute_domain(&domain_parts);
        if name == "AbsolutePath" {
          s.push_str(&parsed.raw_path);
        }
        Some(Expr::String(s))
      }
      _ => None,
    }
  };
  let assoc_of = |names: &[&str]| {
    Expr::Association(
      names
        .iter()
        .map(|n| (Expr::String(n.to_string()), component_value(n).unwrap()))
        .collect(),
    )
  };
  const DEFAULT_COMPONENTS: [&str; 7] = [
    "Scheme", "User", "Domain", "Port", "Path", "Query", "Fragment",
  ];
  const ALL_COMPONENTS: [&str; 13] = [
    "Scheme",
    "User",
    "Domain",
    "Port",
    "Path",
    "Query",
    "Fragment",
    "PathString",
    "QueryString",
    "Username",
    "Password",
    "AbsolutePath",
    "AbsoluteDomain",
  ];

  let Some(spec) = args.get(1) else {
    return Ok(assoc_of(&DEFAULT_COMPONENTS));
  };
  let invcomp = || {
    crate::emit_message(&format!(
      "URLParse::invcomp: The component specification {} is invalid.",
      crate::syntax::expr_to_output(spec)
    ));
  };
  match spec {
    Expr::Identifier(s) if s == "All" => Ok(assoc_of(&ALL_COMPONENTS)),
    Expr::String(s) => match component_value(s) {
      Some(v) => Ok(v),
      None => {
        invcomp();
        unevaluated()
      }
    },
    Expr::List(items) => {
      let values: Option<Vec<Expr>> = items
        .iter()
        .map(|item| match item {
          Expr::String(s) => component_value(s),
          _ => None,
        })
        .collect();
      match values {
        Some(v) => Ok(Expr::List(v.into())),
        None => {
          invcomp();
          unevaluated()
        }
      }
    }
    _ => {
      invcomp();
      unevaluated()
    }
  }
}
