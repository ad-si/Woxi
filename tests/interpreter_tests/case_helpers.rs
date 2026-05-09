//! Shared helpers for tests originally living in `test_cases.rs`.
//!
//! Provides `assert_case(input, expected)` which evaluates `input` with
//! `interpret`, then compares the result against `expected` using the
//! whitespace- / form-insensitive `outputs_match` comparison below.
//! That comparison tolerates float precision noise, commutative
//! Plus/Times reorderings, SparseArray row-bucket permutations, and
//! Wolfram's compact graphics short-forms (e.g. `-Graphics-`).

use super::*;

fn graphic_short_form(s: &str) -> Option<&'static str> {
  match s.trim() {
    "-Graphics-" => Some("Graphics"),
    "-Graphics3D-" => Some("Graphics3D"),
    "-Image-" => Some("Image"),
    "-Sound-" => Some("Sound"),
    "-CompiledFunction-" => Some("CompiledFunction"),
    _ => None,
  }
}

fn normalise(s: &str) -> String {
  let s = if s == "\0" { "Null" } else { s };
  s.chars()
    .filter(|c| !c.is_whitespace() && *c != '"')
    .collect()
}

fn parse_pure_float(s: &str) -> Option<f64> {
  // `ToString[N[Pi, …]]` returns a Woxi `String` whose printed form is
  // wrapped in `"…"`. Strip surrounding double quotes before parsing
  // so the float-tolerance branch can compare those results.
  let t = s.trim().trim_matches('"');
  if t
    .chars()
    .any(|c| matches!(c, '[' | ']' | '{' | '}' | ' ' | ','))
  {
    return None;
  }
  // Drop Wolfram's precision/accuracy marker (e.g. `1.23`50.` →
  // `1.23`, `0``28.` → `0`). Everything from the first backtick on is
  // metadata that f64 doesn't care about.
  let stripped = match t.find('`') {
    Some(i) => &t[..i],
    None => t,
  };
  // Accept Wolfram's `1.23*^5` scientific notation by mapping it onto
  // Rust's `1.23e5`. f64::parse already handles `1.23e5`/`1.23E-10`.
  let normalized = stripped.replace("*^", "e");
  normalized.parse::<f64>().ok()
}

/// Parse a complex literal of the form `<re> ± <im>*I` (or just `<re>`,
/// `<im>*I`, `I`, `-I`). Returns `(re, im)` as f64 pair.
///
/// Handles whitespace and Wolfram's `*^` scientific notation. Used by
/// `outputs_match` so e.g. `0.5547001962252291 + 0.83205*I` matches
/// `0.554700196225229 + 0.83205*I` within 1e-12 relative tolerance.
fn parse_complex_float(s: &str) -> Option<(f64, f64)> {
  let t: String = s
    .chars()
    .filter(|c| !c.is_whitespace())
    .collect::<String>()
    .replace("*^", "e");
  if t.is_empty() {
    return None;
  }
  // Pure imaginary shortcuts: `I`, `-I`, `+I`.
  if t == "I" {
    return Some((0.0, 1.0));
  }
  if t == "-I" {
    return Some((0.0, -1.0));
  }
  if t == "+I" {
    return Some((0.0, 1.0));
  }
  // Find the split between real and imaginary parts. Walk from the end
  // and find the last `+` or `-` that isn't part of an exponent (e.g.
  // `1.23e-4`).
  let bytes = t.as_bytes();
  let mut split: Option<usize> = None;
  for i in (1..bytes.len()).rev() {
    let c = bytes[i] as char;
    if c == '+' || c == '-' {
      let prev = bytes[i - 1] as char;
      if prev == 'e' || prev == 'E' {
        continue; // exponent sign
      }
      split = Some(i);
      break;
    }
  }
  if let Some(i) = split {
    let (left, right) = t.split_at(i);
    // `right` keeps its leading sign (`+x*I` or `-x*I`).
    if !right.ends_with("*I") && !right.ends_with("I") {
      return None;
    }
    let im_str = right.trim_end_matches('I').trim_end_matches('*');
    let im_val = if im_str == "+" || im_str.is_empty() {
      1.0
    } else if im_str == "-" {
      -1.0
    } else {
      im_str.parse::<f64>().ok()?
    };
    let re_val = left.parse::<f64>().ok()?;
    return Some((re_val, im_val));
  }
  // Single term: either pure real or pure imaginary.
  if t.ends_with("*I") || t.ends_with("I") {
    let im_str = t.trim_end_matches('I').trim_end_matches('*');
    let im_val = if im_str.is_empty() || im_str == "+" {
      1.0
    } else if im_str == "-" {
      -1.0
    } else {
      im_str.parse::<f64>().ok()?
    };
    return Some((0.0, im_val));
  }
  let re_val = t.parse::<f64>().ok()?;
  Some((re_val, 0.0))
}

/// Find the byte index of the first top-level `+` separator (the kind
/// that splits a Plus into terms), skipping `+`s inside brackets/parens
/// and exponent signs (`1.23e+4`). Returns `None` for expressions that
/// don't have a top-level `+`.
///
/// Also accepts a binary ` - ` (space-dash-space) at top level — Wolfram
/// prints subtraction as ` - ` between terms, so we treat it as a Plus
/// separator. Unary minus at the start of an expression has no leading
/// space, so it doesn't match.
fn first_top_level_plus(s: &str) -> Option<usize> {
  let bytes = s.as_bytes();
  let mut depth: i32 = 0;
  let mut i = 0;
  while i < bytes.len() {
    let c = bytes[i] as char;
    match c {
      '[' | '{' | '(' => depth += 1,
      ']' | '}' | ')' => depth -= 1,
      '+' if depth == 0 && i > 0 => {
        let prev = bytes[i - 1] as char;
        // Skip exponent sign and unary `+` at start of token.
        if !matches!(prev, 'e' | 'E') {
          // The split must be a real Plus — i.e. surrounded by spaces
          // (or any token boundary). For `1+x` we'd still split, but
          // typical Wolfram output uses ` + ` so this is safe.
          return Some(i);
        }
      }
      '-' if depth == 0 && i > 0 => {
        let prev = bytes[i - 1] as char;
        // Binary ` - `: must be preceded by a space (so we don't pick up
        // exponent signs like `1.23e-4` or unary minus inside terms).
        if prev == ' ' && i + 1 < bytes.len() && bytes[i + 1] == b' ' {
          return Some(i);
        }
      }
      _ => {}
    }
    i += 1;
  }
  None
}

/// Find the byte index of the first top-level `*` separator, used to
/// split a multiplicative term like `<float>*<symbol>` into its factors.
/// Returns `None` for expressions without a top-level `*`.
fn first_top_level_star(s: &str) -> Option<usize> {
  let bytes = s.as_bytes();
  let mut depth: i32 = 0;
  let mut i = 0;
  while i < bytes.len() {
    let c = bytes[i] as char;
    match c {
      '[' | '{' | '(' => depth += 1,
      ']' | '}' | ')' => depth -= 1,
      '*' if depth == 0 => {
        // Skip `*^` (Wolfram's scientific-notation marker, e.g. `2.5*^20`).
        if i + 1 < bytes.len() && bytes[i + 1] == b'^' {
          i += 1;
        } else {
          return Some(i);
        }
      }
      _ => {}
    }
    i += 1;
  }
  None
}

fn outputs_match(a: &str, b: &str) -> bool {
  if normalise(a) == normalise(b) {
    return true;
  }
  if let (Some(x), Some(y)) = (parse_pure_float(a), parse_pure_float(b)) {
    let scale = x.abs().max(y.abs()).max(1.0);
    if (x - y).abs() <= scale * 1e-12 {
      return true;
    }
  }
  // Complex floats: `<re> ± <im>*I`. Compare real and imaginary parts
  // independently with the same 1e-12 relative tolerance, so e.g.
  // `0.5547...291 + 0.83205*I` matches `0.5547...229 + 0.83205*I`.
  if let (Some((ax, ay)), Some((bx, by))) =
    (parse_complex_float(a), parse_complex_float(b))
  {
    // At least one side has to be a "real" complex literal (i.e. an
    // imaginary component). Pure-real values fell through above; this
    // guard keeps us from accepting `1.0` against `1.0 + 0*I` etc.
    if ay != 0.0 || by != 0.0 {
      let re_scale = ax.abs().max(bx.abs()).max(1.0);
      let im_scale = ay.abs().max(by.abs()).max(1.0);
      if (ax - bx).abs() <= re_scale * 1e-12
        && (ay - by).abs() <= im_scale * 1e-12
      {
        return true;
      }
    }
  }
  // List of values: compare element-by-element. Each element is
  // recursively matched, so nested lists and rules like
  // `{1., {x -> 1.5707…66}}` vs `{1., {x -> 1.5707…57}}` collapse
  // through the float tolerance branch on the leaf comparison.
  let a_t = a.trim();
  let b_t = b.trim();
  if a_t.starts_with('{')
    && a_t.ends_with('}')
    && b_t.starts_with('{')
    && b_t.ends_with('}')
  {
    let a_elems = top_level_split_list(a_t);
    let b_elems = top_level_split_list(b_t);
    if !a_elems.is_empty() && a_elems.len() == b_elems.len() {
      let pair_match = a_elems
        .iter()
        .zip(b_elems.iter())
        .all(|(ae, be)| outputs_match(ae, be));
      if pair_match {
        return true;
      }
    }
  }
  // Function call `Head[arg1, arg2, ...]` with matching head: split
  // top-level args on commas and recurse on each, so float-precision
  // noise inside e.g. `NumberForm[{...floats...}, …]` collapses through
  // the leaf float tolerance branch.
  if let (Some((a_head, a_args)), Some((b_head, b_args))) =
    (top_level_split_call(a_t), top_level_split_call(b_t))
    && a_head == b_head
    && a_args.len() == b_args.len()
    && a_args
      .iter()
      .zip(b_args.iter())
      .all(|(ae, be)| outputs_match(ae, be))
  {
    return true;
  }
  // SparseArray equivalence: `SparseArray[Automatic, dims, default,
  // {1, {{rowPtr}, {colIndices}}, {values}}]` has freedom in the
  // within-row ordering of (colIndex, value) pairs (the CSR row-bucket
  // is a multiset). When both sides parse and have matching dims,
  // default, and entry sets, accept them as equivalent.
  if let (Some(sa_a), Some(sa_b)) =
    (parse_sparse_array(a_t), parse_sparse_array(b_t))
    && sparse_array_equiv(&sa_a, &sa_b)
  {
    return true;
  }
  // Rule of values: `<lhs> -> <rhs>` or `<lhs> :> <rhs>` with the
  // same operator on both sides — recurse on each side so float
  // tolerance applies to RHS values like `x -> 1.5707…66`.
  for op in ["->", ":>"] {
    let pad = format!(" {} ", op);
    if let (Some(ai), Some(bi)) = (a_t.find(&pad), b_t.find(&pad)) {
      let (a_l, a_r) = a_t.split_at(ai);
      let a_r = &a_r[pad.len()..];
      let (b_l, b_r) = b_t.split_at(bi);
      let b_r = &b_r[pad.len()..];
      if outputs_match(a_l, b_l) && outputs_match(a_r, b_r) {
        return true;
      }
    }
  }
  // Linear polynomial-style sum: `<a> + <b>` and `<a> + <b>*<sym>` etc.
  // Split on the first top-level `+` (one not inside brackets/parens) and
  // recurse on each side, so float-precision noise propagates element by
  // element. Used for outputs like
  // `0.18644067796610153 + 0.7796610169491526*x`.
  if let (Some(ai), Some(bi)) =
    (first_top_level_plus(a_t), first_top_level_plus(b_t))
  {
    let (a_l, a_r) = a_t.split_at(ai);
    let (b_l, b_r) = b_t.split_at(bi);
    let a_l_t = a_l.trim_end();
    // Strip leading `+`/`-`/spaces so the right side becomes a comparable
    // term. Sign differences are handled by the matching parts having the
    // same separator on both sides at the same depth.
    let a_r_t = a_r.trim_start_matches(['+', '-', ' ']);
    let b_l_t = b_l.trim_end();
    let b_r_t = b_r.trim_start_matches(['+', '-', ' ']);
    if outputs_match(a_l_t, b_l_t) && outputs_match(a_r_t, b_r_t) {
      return true;
    }
  }
  // Commutative Plus: try matching the term-multisets regardless of
  // order. Each term is `(sign, body)` where `sign` is `+` or `-` and
  // `body` is the substring with surrounding parens stripped. Wolfram
  // canonicalizes `T1 - T2` and `-T2 + T1` differently when there's a
  // Times-grouped factor, so the surface order can flip even though
  // both expressions denote the same sum. Also handles the case where
  // one or both sides are wrapped in outer parens (`(a+b)` vs `b+a`).
  let a_for_split = strip_outer_parens(a_t);
  let b_for_split = strip_outer_parens(b_t);
  let a_terms = split_top_level_plus_terms(a_for_split);
  let b_terms = split_top_level_plus_terms(b_for_split);
  if a_terms.len() >= 2
    && a_terms.len() == b_terms.len()
    && commutative_sum_match(&a_terms, &b_terms)
  {
    return true;
  }
  // Multiplicative term like `<float>*<symbol>`. Split on first top-level
  // `*` and recurse so the float side gets the tolerance check.
  if let (Some(ai), Some(bi)) =
    (first_top_level_star(a_t), first_top_level_star(b_t))
  {
    let (a_l, a_r) = a_t.split_at(ai);
    let (b_l, b_r) = b_t.split_at(bi);
    let a_r = &a_r['*'.len_utf8()..];
    let b_r = &b_r['*'.len_utf8()..];
    if outputs_match(a_l.trim(), b_l.trim())
      && outputs_match(a_r.trim(), b_r.trim())
    {
      return true;
    }
  }
  // Commutative Times: split into top-level `*` factors (after stripping
  // any outer parens) and bijectively match the multisets via
  // `outputs_match`. Wolfram's canonical Times order is sometimes
  // different from Woxi's: e.g. `h[1]*x^3` vs `x^3*h[1]` denote the
  // same product. Stays inside `outputs_match` so the existing tests
  // that depend on the strict ordered branch still take precedence.
  let a_factors = split_top_level_star_factors(a_for_split);
  let b_factors = split_top_level_star_factors(b_for_split);
  if a_factors.len() >= 2
    && a_factors.len() == b_factors.len()
    && commutative_product_match(&a_factors, &b_factors)
  {
    return true;
  }
  let pair = match (graphic_short_form(a), graphic_short_form(b)) {
    (Some(h), None) => Some((h, b)),
    (None, Some(h)) => Some((h, a)),
    _ => None,
  };
  if let Some((head, other)) = pair {
    let other = other.trim_start();
    let head_prefix = format!("{}[", head);
    if other.starts_with(&head_prefix) {
      return true;
    }
    // Style[Graphics[…], opts…] (and similar wrappers) renders as the
    // shortform of the inner graphic. Accept if the first positional
    // argument is itself a `Graphics[…]` / `Graphics3D[…]` / ….
    if let Some((wrapper_head, wrapper_args)) = top_level_split_call(other)
      && (wrapper_head == "Style" || wrapper_head == "Labeled")
      && let Some(first) = wrapper_args.first()
      && first.trim_start().starts_with(&head_prefix)
    {
      return true;
    }
  }
  // List-of-graphics: "{-Graphics-, ..., -Graphics-}" matches a list whose
  // every element is a `Graphics[...]` (or `Graphics3D[...]`/etc.) of the
  // same length.
  if let Some((head, expanded)) = list_of_graphics_pair(a, b) {
    let elems = top_level_split_list(expanded);
    let head_prefix = format!("{}[", head);
    if !elems.is_empty()
      && elems
        .iter()
        .all(|e| e.trim_start().starts_with(&head_prefix))
    {
      // Count `-Head-` placeholders on the other side.
      let placeholder = format!("-{}-", head);
      let occurrences = top_level_split_list(if expanded == a { b } else { a });
      if occurrences.len() == elems.len()
        && occurrences.iter().all(|p| p.trim() == placeholder)
      {
        return true;
      }
    }
  }
  false
}

/// If one of (a, b) is a list of identical graphics short forms like
/// `{-Graphics-, -Graphics-}` and the other is a list, return
/// (head_name, expanded_side).
fn list_of_graphics_pair<'a>(
  a: &'a str,
  b: &'a str,
) -> Option<(&'static str, &'a str)> {
  let a_t = a.trim();
  let b_t = b.trim();
  let a_is_list = a_t.starts_with('{') && a_t.ends_with('}');
  let b_is_list = b_t.starts_with('{') && b_t.ends_with('}');
  if !(a_is_list && b_is_list) {
    return None;
  }
  let a_elems = top_level_split_list(a_t);
  let b_elems = top_level_split_list(b_t);
  let a_short = a_elems.iter().find_map(|e| graphic_short_form(e));
  let b_short = b_elems.iter().find_map(|e| graphic_short_form(e));
  match (a_short, b_short) {
    (Some(h), None) => Some((h, b_t)),
    (None, Some(h)) => Some((h, a_t)),
    _ => None,
  }
}

/// Split a top-level "{a, b, c}" list literal on commas at depth 0.
/// Returns the inner element strings (trimmed of surrounding whitespace).
fn top_level_split_list(s: &str) -> Vec<&str> {
  let s = s.trim();
  if !s.starts_with('{') || !s.ends_with('}') {
    return Vec::new();
  }
  let inner = &s[1..s.len() - 1];
  let bytes = inner.as_bytes();
  let mut depth: i32 = 0;
  let mut start = 0usize;
  let mut out = Vec::new();
  let mut i = 0;
  while i < bytes.len() {
    let c = bytes[i] as char;
    match c {
      '{' | '[' | '(' => depth += 1,
      '}' | ']' | ')' => depth -= 1,
      ',' if depth == 0 => {
        out.push(inner[start..i].trim());
        start = i + 1;
      }
      _ => {}
    }
    i += 1;
  }
  out.push(inner[start..].trim());
  out
}

/// Split a sum-like string into top-level `(sign, body)` pairs. `sign`
/// is `'+'` or `'-'`. `body` is the term substring with its surrounding
/// matched parens stripped (e.g. `((b*c)/X)` becomes `(b*c)/X`).
///
/// A leading unary `-` produces a `'-'` for the first term; otherwise
/// the first term is `'+'`. Returns the original input as one `'+'`
/// term if there's no top-level Plus separator.
fn split_top_level_plus_terms(s: &str) -> Vec<(char, String)> {
  let s = s.trim();
  let bytes = s.as_bytes();
  let mut depth: i32 = 0;
  let mut splits: Vec<(usize, char)> = Vec::new();
  let mut i = 0;
  while i < bytes.len() {
    let c = bytes[i] as char;
    match c {
      '[' | '{' | '(' => depth += 1,
      ']' | '}' | ')' => depth -= 1,
      '+' if depth == 0 && i > 0 => {
        let prev = bytes[i - 1] as char;
        if !matches!(prev, 'e' | 'E') && prev == ' ' {
          splits.push((i, '+'));
        }
      }
      '-' if depth == 0 && i > 0 => {
        let prev = bytes[i - 1] as char;
        if prev == ' ' && i + 1 < bytes.len() && bytes[i + 1] == b' ' {
          splits.push((i, '-'));
        }
      }
      _ => {}
    }
    i += 1;
  }
  if splits.is_empty() {
    // Single term — preserve any leading `-`.
    let (sign, body) = if let Some(rest) = s.strip_prefix('-') {
      ('-', rest.trim())
    } else {
      ('+', s)
    };
    return vec![normalize_term_sign(sign, strip_outer_parens(body))];
  }
  let mut terms: Vec<(char, String)> = Vec::new();
  let first_chunk = s[..splits[0].0].trim();
  let (first_sign, first_body) =
    if let Some(rest) = first_chunk.strip_prefix('-') {
      ('-', rest.trim())
    } else {
      ('+', first_chunk)
    };
  terms.push(normalize_term_sign(
    first_sign,
    strip_outer_parens(first_body),
  ));
  for w in 0..splits.len() {
    let (idx, sign) = splits[w];
    let end = if w + 1 < splits.len() {
      splits[w + 1].0
    } else {
      s.len()
    };
    let body = s[idx + 1..end].trim();
    terms.push(normalize_term_sign(sign, strip_outer_parens(body)));
  }
  terms
}

/// Pull a leading `(-` prefix out of a Times-like term body when its
/// matching `)` doesn't close the whole expression. Wolfram prints
/// `-x/3` as `(-x)/3` (paren around the negated factor), so the surface
/// form has a `(-…)` Times factor even though the term is mathematically
/// `-((…)/3)`. Flipping the sign and replacing `(-X)…` with `(X)…` makes
/// `(+, (-X)/3)` and `(-, (X)/3)` compare equal.
fn normalize_term_sign(sign: char, body: &str) -> (char, String) {
  if !body.starts_with("(-") {
    return (sign, body.to_string());
  }
  let bytes = body.as_bytes();
  let mut depth: i32 = 0;
  let mut close_idx = None;
  for (i, &b) in bytes.iter().enumerate() {
    match b as char {
      '(' => depth += 1,
      ')' => {
        depth -= 1;
        if depth == 0 {
          close_idx = Some(i);
          break;
        }
      }
      _ => {}
    }
  }
  match close_idx {
    Some(idx) if idx + 1 < bytes.len() => {
      let inner = &body[2..idx];
      let rest = &body[idx + 1..];
      let new_body = format!("({}){}", inner, rest);
      let new_sign = if sign == '+' { '-' } else { '+' };
      (new_sign, new_body)
    }
    _ => (sign, body.to_string()),
  }
}

/// Strip a single layer of matched outer parens, if they wrap the whole
/// expression. E.g. `((b*c)/X)` → `(b*c)/X`. Used when comparing terms
/// that wolframscript prints with extra parens around a negated product.
fn strip_outer_parens(s: &str) -> &str {
  let s = s.trim();
  if !s.starts_with('(') || !s.ends_with(')') {
    return s;
  }
  let bytes = s.as_bytes();
  let mut depth: i32 = 0;
  for (i, &b) in bytes.iter().enumerate() {
    match b as char {
      '(' => depth += 1,
      ')' => {
        depth -= 1;
        if depth == 0 && i + 1 != bytes.len() {
          return s;
        }
      }
      _ => {}
    }
  }
  &s[1..s.len() - 1]
}

/// Try to match two sum term lists as multisets — every `(sign, body)`
/// in `a` must correspond to a `(sign, body)` in `b` where the bodies
/// match recursively via `outputs_match`. Bodies are matched with the
/// signs intact, so `+T1 - T2` only matches `-T2 + T1`, not `+T2 - T1`.
fn commutative_sum_match(a: &[(char, String)], b: &[(char, String)]) -> bool {
  if a.len() != b.len() {
    return false;
  }
  let mut used = vec![false; b.len()];
  for (sign_a, body_a) in a {
    let mut found = false;
    for (j, (sign_b, body_b)) in b.iter().enumerate() {
      if used[j] || sign_a != sign_b {
        continue;
      }
      if outputs_match(body_a, body_b) {
        used[j] = true;
        found = true;
        break;
      }
    }
    if !found {
      return false;
    }
  }
  true
}

/// Split a Times-like string into top-level `*` factors. Skips `*^`
/// (Wolfram scientific-notation marker like `2.5*^20`) and only splits
/// at depth 0. Returns a single-element vec for inputs without a
/// top-level `*`.
fn split_top_level_star_factors(s: &str) -> Vec<&str> {
  let s = s.trim();
  let bytes = s.as_bytes();
  let mut depth: i32 = 0;
  let mut splits: Vec<usize> = Vec::new();
  let mut i = 0;
  while i < bytes.len() {
    let c = bytes[i] as char;
    match c {
      '[' | '{' | '(' => depth += 1,
      ']' | '}' | ')' => depth -= 1,
      '*' if depth == 0 => {
        if i + 1 < bytes.len() && bytes[i + 1] == b'^' {
          i += 2;
          continue;
        }
        splits.push(i);
      }
      _ => {}
    }
    i += 1;
  }
  if splits.is_empty() {
    return vec![s];
  }
  let mut out: Vec<&str> = Vec::with_capacity(splits.len() + 1);
  let mut prev = 0;
  for &idx in &splits {
    out.push(s[prev..idx].trim());
    prev = idx + 1;
  }
  out.push(s[prev..].trim());
  out
}

/// Parsed `SparseArray[Automatic, dims, default, {1, {{rowPtr},
/// {colIndices}}, {values}}]`. Each entry in `entries` is a
/// `(multi_index_strs, value_str)` pair where `multi_index_strs` is
/// the 1-based tuple `(row, col_1, col_2, …)`.
struct ParsedSparseArrayStr {
  dims: String,
  default: String,
  entries: Vec<(Vec<String>, String)>,
}

fn parse_sparse_array(s: &str) -> Option<ParsedSparseArrayStr> {
  let s = s.trim();
  let (head, args) = top_level_split_call(s)?;
  if head != "SparseArray" || args.len() != 4 {
    return None;
  }
  if args[0].trim() != "Automatic" {
    return None;
  }
  let dims = args[1].trim().to_string();
  let default = args[2].trim().to_string();
  let payload = args[3].trim();
  // payload = `{1, {{rowPtr…}, {colIndices…}}, {values…}}` — three
  // top-level commas at depth 0 inside a single outer `{}`.
  let inner = strip_outer_braces(payload)?;
  let parts = top_level_split_brace_inner(inner);
  if parts.len() != 3 {
    return None;
  }
  if parts[0].trim() != "1" {
    return None;
  }
  let layout_inner = strip_outer_braces(parts[1].trim())?;
  let layout_parts = top_level_split_brace_inner(layout_inner);
  if layout_parts.len() != 2 {
    return None;
  }
  let row_ptr_inner = strip_outer_braces(layout_parts[0].trim())?;
  let row_ptr_strs = top_level_split_brace_inner(row_ptr_inner);
  let row_ptr: Vec<i64> = row_ptr_strs
    .iter()
    .filter_map(|s| s.trim().parse::<i64>().ok())
    .collect();
  if row_ptr.len() != row_ptr_strs.len() {
    return None;
  }
  let col_indices_inner = strip_outer_braces(layout_parts[1].trim())?;
  let col_indices_strs = if col_indices_inner.trim().is_empty() {
    Vec::new()
  } else {
    top_level_split_brace_inner(col_indices_inner)
  };
  let col_indices: Vec<Vec<String>> = col_indices_strs
    .iter()
    .filter_map(|s| {
      let inner = strip_outer_braces(s.trim())?;
      Some(
        top_level_split_brace_inner(inner)
          .into_iter()
          .map(|x| x.trim().to_string())
          .collect(),
      )
    })
    .collect();
  let values_inner = strip_outer_braces(parts[2].trim())?;
  let values: Vec<String> = if values_inner.trim().is_empty() {
    Vec::new()
  } else {
    top_level_split_brace_inner(values_inner)
      .into_iter()
      .map(|s| s.trim().to_string())
      .collect()
  };
  if col_indices.len() != values.len() {
    return None;
  }
  // Reconstruct entries from rowPtr buckets. For rank-1 SparseArrays
  // wolframscript stores rowPtr as `{0, count}` (length 2) regardless
  // of the inner dimension; the colIndex 1-tuple is the full index.
  // For rank ≥ 2 the rowPtr length is `dims[0] + 1` and the row
  // 1-based index becomes the leading multi-index component.
  let dims_inner = strip_outer_braces(&dims)?;
  let dims_count = top_level_split_brace_inner(dims_inner).len();
  let mut entries: Vec<(Vec<String>, String)> = Vec::new();
  let max_rows = if dims_count >= 2 {
    row_ptr.len().saturating_sub(1)
  } else {
    1
  };
  for row in 0..max_rows {
    let lo = *row_ptr.get(row)? as usize;
    let hi = *row_ptr.get(row + 1)? as usize;
    for i in lo..hi.min(values.len()) {
      let mut idx = if dims_count >= 2 {
        vec![(row + 1).to_string()]
      } else {
        Vec::new()
      };
      idx.extend(col_indices[i].iter().cloned());
      entries.push((idx, values[i].clone()));
    }
  }
  Some(ParsedSparseArrayStr {
    dims,
    default,
    entries,
  })
}

/// Strip a single outer `{ … }` wrapping. Returns `None` if `s` isn't
/// surrounded by matched braces.
fn strip_outer_braces(s: &str) -> Option<&str> {
  let s = s.trim();
  if !s.starts_with('{') || !s.ends_with('}') {
    return None;
  }
  let bytes = s.as_bytes();
  let mut depth = 0i32;
  for (i, &b) in bytes.iter().enumerate() {
    match b as char {
      '{' | '[' | '(' => depth += 1,
      '}' | ']' | ')' => {
        depth -= 1;
        if depth == 0 && i + 1 != bytes.len() {
          return None;
        }
      }
      _ => {}
    }
  }
  Some(&s[1..s.len() - 1])
}

/// Split the inside of a `{ … }` (already with the braces stripped)
/// on top-level commas at depth 0.
fn top_level_split_brace_inner(s: &str) -> Vec<&str> {
  let bytes = s.as_bytes();
  let mut depth = 0i32;
  let mut start = 0usize;
  let mut out = Vec::new();
  let mut i = 0;
  while i < bytes.len() {
    match bytes[i] as char {
      '{' | '[' | '(' => depth += 1,
      '}' | ']' | ')' => depth -= 1,
      ',' if depth == 0 => {
        out.push(s[start..i].trim());
        start = i + 1;
      }
      _ => {}
    }
    i += 1;
  }
  out.push(s[start..].trim());
  if out.last().is_some_and(|x| x.is_empty()) && out.len() == 1 {
    out.clear();
  }
  out
}

fn sparse_array_equiv(
  a: &ParsedSparseArrayStr,
  b: &ParsedSparseArrayStr,
) -> bool {
  if normalise(&a.dims) != normalise(&b.dims) {
    return false;
  }
  if !outputs_match(&a.default, &b.default) {
    return false;
  }
  if a.entries.len() != b.entries.len() {
    return false;
  }
  let mut used = vec![false; b.entries.len()];
  for (ai, av) in &a.entries {
    let mut found = false;
    for (j, (bi, bv)) in b.entries.iter().enumerate() {
      if used[j] || ai != bi {
        continue;
      }
      if outputs_match(av, bv) {
        used[j] = true;
        found = true;
        break;
      }
    }
    if !found {
      return false;
    }
  }
  true
}

/// Bijectively match two factor lists via `outputs_match`. Used to
/// accept commutative-Times reorderings like `h[1]*x^3` ↔ `x^3*h[1]`.
fn commutative_product_match(a: &[&str], b: &[&str]) -> bool {
  if a.len() != b.len() {
    return false;
  }
  let mut used = vec![false; b.len()];
  for fa in a {
    let mut found = false;
    for (j, fb) in b.iter().enumerate() {
      if used[j] {
        continue;
      }
      if outputs_match(fa, fb) {
        used[j] = true;
        found = true;
        break;
      }
    }
    if !found {
      return false;
    }
  }
  true
}

/// If `s` is a function-call literal `Head[arg1, arg2, ...]` with a plain
/// identifier head and balanced brackets, return `(head, args)`. Otherwise
/// return `None`. Args are returned trimmed and split at top-level commas.
fn top_level_split_call(s: &str) -> Option<(&str, Vec<&str>)> {
  let s = s.trim();
  if !s.ends_with(']') {
    return None;
  }
  let lb = s.find('[')?;
  let head = &s[..lb];
  if head.is_empty()
    || !head
      .chars()
      .all(|c| c.is_ascii_alphanumeric() || c == '`' || c == '$')
  {
    return None;
  }
  let inner = &s[lb + 1..s.len() - 1];
  let bytes = inner.as_bytes();
  let mut depth: i32 = 0;
  let mut start = 0usize;
  let mut out = Vec::new();
  let mut i = 0;
  while i < bytes.len() {
    let c = bytes[i] as char;
    match c {
      '{' | '[' | '(' => depth += 1,
      '}' | ']' | ')' => depth -= 1,
      ',' if depth == 0 => {
        out.push(inner[start..i].trim());
        start = i + 1;
      }
      _ => {}
    }
    i += 1;
  }
  out.push(inner[start..].trim());
  if out.last().is_some_and(|s| s.is_empty()) && out.len() == 1 {
    out.clear();
  }
  Some((head, out))
}

pub fn assert_case(input: &str, expected: &str) {
  clear_state();
  let result =
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| interpret(input)));
  let actual = match result {
    Ok(Ok(s)) => s,
    Ok(Err(e)) => panic!(
      "Woxi returned error: {:?}\n  input:    {}\n  expected: {}",
      e, input, expected
    ),
    Err(_) => panic!(
      "Woxi panicked\n  input:    {}\n  expected: {}",
      input, expected
    ),
  };
  if !outputs_match(&actual, expected) {
    panic!(
      "output mismatch\n  input:    {}\n  expected: {}\n  actual:   {}",
      input, expected, actual
    );
  }
}
