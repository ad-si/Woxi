use super::*;
use crate::syntax::ImageType;

fn expr_to_i64(expr: &Expr) -> Option<i64> {
  match expr {
    Expr::Integer(n) => Some(*n as i64),
    Expr::BigInteger(n) => {
      use num_traits::ToPrimitive;
      n.to_i64()
    }
    Expr::Real(f) => Some(*f as i64),
    _ => None,
  }
}

fn normalize_span_pos(pos: i64, len: i64) -> Option<i64> {
  if pos == 0 {
    return None;
  }
  let normalized = if pos < 0 { len + pos + 1 } else { pos };
  if (1..=len).contains(&normalized) {
    Some(normalized)
  } else {
    None
  }
}

fn part_take_unevaluated(expr: &Expr, index: &Expr) -> Expr {
  Expr::Part {
    expr: Box::new(expr.clone()),
    index: Box::new(index.clone()),
  }
}

/// `a | b | c` (Alternatives) is a flat, associative head in WL but is stored
/// as a nested BinaryOp chain. Rewrite it into a flattened
/// `Alternatives[a, b, c]` FunctionCall so positional operations (Part, Take,
/// …) see all operands as siblings, matching wolframscript. Returns `None`
/// when `expr` is not such a chain.
pub fn flatten_alternatives_binop(expr: &Expr) -> Option<Expr> {
  fn gather(e: &Expr, out: &mut Vec<Expr>) -> bool {
    if let Expr::BinaryOp {
      op: BinaryOperator::Alternatives,
      left,
      right,
    } = e
    {
      gather(left, out);
      gather(right, out);
      true
    } else {
      out.push(e.clone());
      false
    }
  }
  // Anything but an Alternatives node is left alone without copying it.
  if !matches!(
    expr,
    Expr::BinaryOp {
      op: BinaryOperator::Alternatives,
      ..
    }
  ) {
    return None;
  }
  let mut parts = Vec::new();
  gather(expr, &mut parts);
  Some(call("Alternatives", parts))
}

fn part_take_warn(expr: &Expr, start: i64, end: i64) {
  crate::emit_message_to_stdout(&crate::syntax::message_with_expr_2d(
    &format!("Part::take: Cannot take positions {start} through {end} in "),
    expr,
    ".",
  ));
}

/// `Part::partw` about a position `expr` lacks, quoting `expr` in
/// OutputForm as the kernel does.
fn part_missing_warn(position: &str, expr: &Expr) {
  crate::emit_message_to_stdout(&crate::syntax::message_with_expr_2d(
    &format!("Part::partw: Part {position} of "),
    expr,
    " does not exist.",
  ));
}

/// A span endpoint as a position in a part of length `len`: an integer, or
/// `UpTo[n]`, which stands for `Min[n, len]` at either end of the span
/// (`l[[2 ;; UpTo[9]]]` stops at the last element, `{}[[1 ;; UpTo[3]]]` is
/// `{}`).
fn span_endpoint(e: &Expr, len: i64) -> Option<i64> {
  match e {
    Expr::FunctionCall { name, args } if name == "UpTo" && args.len() == 1 => {
      expr_to_i64(&args[0])
        .filter(|n| *n >= 0)
        .map(|n| n.min(len))
    }
    other => expr_to_i64(other),
  }
}

fn extract_span_from_items(
  expr: &Expr,
  index: &Expr,
  items: &[Expr],
  start_expr: &Expr,
  end_expr: &Expr,
  step_expr: Option<&Expr>,
  rebuild: impl FnOnce(Vec<Expr>) -> Expr,
) -> Result<Expr, InterpreterError> {
  let len = items.len() as i64;
  let start_raw = if matches!(start_expr, Expr::Identifier(s) if s == "All") {
    1
  } else if let Some(v) = span_endpoint(start_expr, len) {
    v
  } else {
    return Ok(part_take_unevaluated(expr, index));
  };
  let step = match step_expr {
    Some(e) => match expr_to_i64(e) {
      Some(v) => v,
      None => return Ok(part_take_unevaluated(expr, index)),
    },
    None => 1,
  };
  if step == 0 {
    let end_raw = if matches!(end_expr, Expr::Identifier(s) if s == "All") {
      len
    } else if let Some(v) = span_endpoint(end_expr, len) {
      v
    } else {
      return Ok(part_take_unevaluated(expr, index));
    };
    part_take_warn(expr, start_raw, end_raw);
    return Ok(part_take_unevaluated(expr, index));
  }
  let end_raw = if matches!(end_expr, Expr::Identifier(s) if s == "All") {
    if step < 0 { -1 } else { len }
  } else if let Some(v) = span_endpoint(end_expr, len) {
    v
  } else {
    return Ok(part_take_unevaluated(expr, index));
  };
  // An open end is reported as `-1`, as wolframscript writes it:
  // `{1, 2, 3}[[5 ;;]]` cannot take positions 5 through -1.
  let end_shown = if matches!(end_expr, Expr::Identifier(s) if s == "All") {
    -1
  } else {
    end_raw
  };

  let start = if let Some(v) = normalize_span_pos(start_raw, len) {
    v
  } else if start_raw == len + 1 {
    // A start one past the end is a valid empty span: `{a}[[2;;]]` → `{}`,
    // `{a,b,c}[[4;;]]` → `{}` (matching wolfram). The empty-result branch
    // below (end == start - 1) then yields the empty list.
    len + 1
  } else {
    part_take_warn(expr, start_raw, end_shown);
    return Ok(part_take_unevaluated(expr, index));
  };
  let end = if let Some(v) = normalize_span_pos(end_raw, len) {
    v
  } else {
    // An end one before the start is a valid empty span: `{x}[[;; -2]]` →
    // `{}` (end normalizes to 0). The empty-result branch below handles it.
    let e = if end_raw < 0 {
      len + end_raw + 1
    } else {
      end_raw
    };
    if e == start - 1 {
      e
    } else {
      part_take_warn(expr, start_raw, end_shown);
      return Ok(part_take_unevaluated(expr, index));
    }
  };

  if (step > 0 && start > end) || (step < 0 && start < end) {
    // Reverse-direction span. wolframscript treats the adjacent case
    // (`end == start - step`) as an empty result — so `{a,b,c}[[-1;;-2]]`
    // and `{…}[[3;;2]]` produce `{}`, and `Plus[a,b,c,d][[-1;;-2]]`
    // folds to `0`. Any wider inversion (e.g. `5;;2`) is an error.
    if (step > 0 && end == start - 1) || (step < 0 && end == start + 1) {
      let empty = rebuild(Vec::new());
      return crate::evaluator::evaluate_expr_to_expr(&empty);
    }
    part_take_warn(expr, start_raw, end_shown);
    return Ok(part_take_unevaluated(expr, index));
  }

  let mut out = Vec::new();
  let mut i = start;
  loop {
    if (step > 0 && i > end) || (step < 0 && i < end) {
      break;
    }
    out.push(items[(i - 1) as usize].clone());
    i += step;
  }
  Ok(rebuild(out))
}

/// Whether `taken` is the unevaluated `obj[[index]]` that a failed extraction
/// hands back, as opposed to a piece of `obj` that is a `Part` expression
/// itself (the `x[[2]]` held inside `Dynamic[x[[2]]]`).
pub fn part_failed(obj: &Expr, taken: &Expr, index: &Expr) -> bool {
  // `Expr` has no structural equality; the printed forms stand in for it
  // (only reached once the extraction came back as a `Part`).
  let Expr::Part { expr, index: i } = taken else {
    return false;
  };
  let form =
    |e: &Expr| crate::syntax::format_expr(e, crate::syntax::ExprForm::Output);
  form(i) == form(index) && form(expr) == form(obj)
}

/// Get the innermost base of nested Part expressions
pub fn get_part_base(expr: &Expr) -> &Expr {
  match expr {
    Expr::Part { expr: inner, .. } => get_part_base(inner),
    _ => expr,
  }
}

/// Apply a chain of Part indices to an expression, handling All by mapping over elements
pub fn apply_part_indices(
  expr: &Expr,
  indices: &[Expr],
) -> Result<Expr, InterpreterError> {
  if indices.is_empty() {
    return Ok(expr.clone());
  }
  // A NumericArray indexes the array it stands for (see
  // `extract_part_ast_rest`), with every index — `All` included — applied
  // to that array rather than to the wrapper's arguments.
  if let Expr::FunctionCall { name, args: na } = expr
    && name == "NumericArray"
    && (na.len() == 1 || na.len() == 2)
    && matches!(&na[0], Expr::List(_))
  {
    let part = apply_part_indices(&na[0], indices)?;
    if matches!(part, Expr::List(_)) {
      let mut wrap = vec![part];
      wrap.extend(na.get(1).cloned());
      return crate::evaluator::evaluate_function_call_ast(
        "NumericArray",
        &wrap,
      );
    }
    return Ok(part);
  }
  // A SparseArray, likewise, indexes the dense array it stands for — `All`
  // included — and an array part of it is a SparseArray again.
  if let Expr::FunctionCall { name, args: sa } = expr
    && name == "SparseArray"
    && sa.len() == 4
    && indices.len() > 1
    && let Some(part) = sparse_array_part(sa, indices)
  {
    return part;
  }
  if let Expr::FunctionCall { name, args: sa } = expr
    && name == "SparseArray"
    && sa.len() == 4
    && indices.len() > 1
  {
    let dense = crate::functions::list_helpers_ast::sparse_array_ast(sa)?;
    if matches!(&dense, Expr::List(_)) {
      let part = apply_part_indices(&dense, indices)?;
      if matches!(part, Expr::List(_)) {
        return crate::evaluator::evaluate_function_call_ast(
          "SparseArray",
          &[part],
        );
      }
      return Ok(part);
    }
  }
  let idx = &indices[0];
  let rest = &indices[1..];
  if matches!(idx, Expr::Identifier(s) if s == "All") {
    // All: map remaining indices over each element
    if rest.is_empty() {
      // Part[expr, All] with no more indices — return as-is
      return Ok(expr.clone());
    }
    match expr {
      Expr::List(items) => {
        let mapped: Result<Vec<Expr>, InterpreterError> = items
          .iter()
          .map(|item| apply_part_indices(item, rest))
          .collect();
        Ok(Expr::List(mapped?.into()))
      }
      Expr::FunctionCall { name, args } => {
        let mapped: Result<Vec<Expr>, InterpreterError> = args
          .iter()
          .map(|item| apply_part_indices(item, rest))
          .collect();
        Ok(Expr::FunctionCall {
          name: name.clone(),
          args: mapped?.into(),
        })
      }
      _ => apply_part_indices(expr, rest),
    }
  } else if let Expr::List(sub_indices) = idx {
    // Part[expr, {i1, i2, ...}, rest...] → {Part[expr, i1, rest...], ...}
    // (map remaining indices over each extracted element, like All does)
    if rest.is_empty() {
      let extracted = extract_part_ast(expr, idx)?;
      return Ok(extracted);
    }
    let mut results = Vec::with_capacity(sub_indices.len());
    for sub_idx in sub_indices {
      let extracted = extract_part_ast(expr, sub_idx)?;
      results.push(apply_part_indices(&extracted, rest)?);
    }
    Ok(Expr::List(results.into()))
  } else if let Expr::FunctionCall { name, .. } = idx
    && name == "Span"
    && !rest.is_empty()
  {
    // Part[expr, Span[...], rest...] → map remaining indices
    // over each element of the extracted sub-list, the same way a bare
    // list-of-positions index does.
    let extracted = extract_part_ast(expr, idx)?;
    match &extracted {
      Expr::List(items) => {
        let mapped: Result<Vec<Expr>, InterpreterError> = items
          .iter()
          .map(|item| apply_part_indices(item, rest))
          .collect();
        Ok(Expr::List(mapped?.into()))
      }
      Expr::FunctionCall {
        name: head,
        args: fn_args,
      } => {
        let mapped: Result<Vec<Expr>, InterpreterError> = fn_args
          .iter()
          .map(|item| apply_part_indices(item, rest))
          .collect();
        Ok(Expr::FunctionCall {
          name: head.clone(),
          args: mapped?.into(),
        })
      }
      _ => apply_part_indices(&extracted, rest),
    }
  } else {
    // Normal index: extract, then continue with remaining indices
    let extracted = extract_part_ast(expr, idx)?;
    apply_part_indices(&extracted, rest)
  }
}

thread_local! {
  /// The symbols whose stored value was rebuilt by a Part assignment. Such a
  /// value is stored the way it was assembled, without being evaluated again
  /// — that is what wolframscript does, and reading the symbol is what
  /// evaluates it. So `a = sC f[x]; a[[1]] = 1` leaves `Times[1, f[x]]`
  /// behind, and a later `a[[2]] = …` really does address *that* expression's
  /// second part, while `a` itself reads back as `f[x]`.
  ///
  /// Every other stored value came from evaluating a right-hand side and is
  /// already in evaluated form, so the far more common `bigTable[[i]]` never
  /// pays for a re-evaluation it cannot need.
  static PART_ASSIGNED: crate::epoch::EpochCell<std::collections::HashSet<String>> =
    crate::epoch::EpochCell::new(std::collections::HashSet::new());
}

/// Remember that `var_name`'s value was rebuilt by a Part assignment.
pub fn note_part_assigned(var_name: &str) {
  PART_ASSIGNED.with(|set| {
    set
      .borrow_mut_own_value(var_name)
      .insert(var_name.to_string())
  });
}

/// Forget it again — the symbol has just been given a whole new value, which
/// arrived evaluated.
pub fn forget_part_assigned(var_name: &str) {
  PART_ASSIGNED.with(|set| {
    let mut set = set.borrow_mut_own_value(var_name);
    if !set.is_empty() {
      set.remove(var_name);
    }
  });
}

/// Evaluate what was read out of `var_name` unless it is finished data: a
/// Part assignment can have left it unevaluated, and a symbolic piece
/// evaluates in the current environment (`r = {f[t] -> g[t]}` read as
/// `r[[1]]` inside `Table[…, {t, 1, 2}]` is `f[1] -> g[1]`). A piece
/// taken out of a stored list is otherwise handed over as it stands.
pub fn evaluate_if_part_assigned(
  var_name: &str,
  part: Expr,
) -> Result<Expr, InterpreterError> {
  // An unevaluated `Part` (the stored value had no such part) is the
  // answer itself; evaluating it would only extract it again.
  if was_part_assigned(var_name)
    || !(crate::evaluator::core_eval::is_inert_data(&part)
      || matches!(part, Expr::Part { .. }))
  {
    evaluate_expr_to_expr(&part)
  } else {
    Ok(part)
  }
}

fn was_part_assigned(var_name: &str) -> bool {
  PART_ASSIGNED
    .with(|set| !set.borrow().is_empty() && set.borrow().contains(var_name))
}

/// Evaluate the base expression of a Part, with optimization for identifiers
/// in ENV.
///
/// A stored value is handed back as it stands unless a Part assignment left
/// it unevaluated — see `PART_ASSIGNED` — since re-walking a whole table on
/// every `a[[i]]` is what this shortcut exists to avoid. A list is handed
/// back either way: evaluating one is what makes filling a table quadratic,
/// and its elements were evaluated as they were assigned.
pub fn eval_part_base(e: &Expr) -> Result<Expr, InterpreterError> {
  if let Expr::Identifier(var_name) = e {
    let stored =
      match crate::evaluator::core_eval::stored_value_for_lookup(var_name) {
        Some(StoredValue::ExprVal(stored)) => Some(stored),
        _ => None,
      };
    if let Some(stored) = stored {
      // `x := body` (SetDelayed) stores `body` unevaluated — plain
      // identifier lookup re-evaluates it every access (see
      // `needs_reevaluation` in core_eval), and Part must match that or a
      // symbol whose delayed value is a function call (`getCoor := bar[p]`)
      // hands the raw call to `extract_part_ast` and misreads its argument
      // count as a part depth (`getCoor[[2, 2]]` reporting
      // `Part::partw: Part 2 of bar[p] does not exist`). A stored `List` is
      // exempt: `Set`/`SetDelayed` both evaluate a list literal's elements
      // as they are built (`Table`, list assignment, …), so re-walking it
      // here on every access would only pay an O(n) needs_reevaluation scan
      // for no benefit — the table-filling cost this shortcut exists to
      // avoid (see module docs above).
      let needs_eval = !matches!(stored, Expr::List(_))
        && (was_part_assigned(var_name)
          || crate::evaluator::core_eval::needs_reevaluation(
            &stored, var_name,
          ));
      return if needs_eval {
        evaluate_expr_to_expr(&stored)
      } else {
        Ok(stored)
      };
    }
  }
  evaluate_expr_to_expr(e)
}

/// Number of positionally-indexable parts of `expr`, or `None` for atoms and
/// expressions (like ByteArray) whose part count differs from their argument
/// count. Used to validate a list-of-positions spec atomically.
fn positional_length(expr: &Expr) -> Option<usize> {
  match expr {
    Expr::List(items) => Some(items.len()),
    Expr::Rule { .. } | Expr::RuleDelayed { .. } => Some(2),
    Expr::FunctionCall { name, args } if name != "ByteArray" => {
      Some(args.len())
    }
    _ => None,
  }
}

/// Rewrap the elements selected by a list-of-positions spec with the head of
/// the original expression: `f[a,b,c][[{1,3}]]` is `f[a, c]`, `(x->y)[[{1,2}]]`
/// is `x -> y`, and a List keeps the List head. Only used on expressions that
/// `positional_length` accepts.
fn rewrap_part_list(expr: &Expr, elems: Vec<Expr>) -> Expr {
  match expr {
    Expr::FunctionCall { name, .. } => Expr::FunctionCall {
      name: name.clone(),
      args: elems.into(),
    },
    Expr::Rule { .. } if elems.len() == 2 => {
      rule_expr(elems[0].clone(), elems[1].clone())
    }
    Expr::RuleDelayed { .. } if elems.len() == 2 => {
      rule_delayed_expr(elems[0].clone(), elems[1].clone())
    }
    _ => Expr::List(elems.into()),
  }
}

/// Concrete integer position from an index expression (Integer, integer-valued
/// Real, or BigInteger), or `None` for anything else.
fn simple_position(e: &Expr) -> Option<i64> {
  match e {
    Expr::Integer(n) => Some(*n as i64),
    Expr::BigInteger(n) => {
      use num_traits::ToPrimitive;
      n.to_i64()
    }
    Expr::Real(f) if f.is_finite() && f.fract() == 0.0 => Some(*f as i64),
    _ => None,
  }
}

/// The first index of a Part spec that cannot address a part of anything:
/// a symbol other than `All`, a non-integer number, an arbitrary
/// expression, or a list holding anything but positions and keys. WL
/// rejects such a spec with Part::pkspec1 before it looks at the object, so
/// `x[[1 + y]]` reports the spec rather than the depth. Spans and `UpTo`
/// carry their own messages and are left to `extract_part_ast`.
pub fn invalid_part_spec(indices: &[Expr]) -> Option<&Expr> {
  let position_or_key = |e: &Expr| {
    simple_position(e).is_some()
      || matches!(e, Expr::BigInteger(_) | Expr::String(_))
      || matches!(e, Expr::FunctionCall { name, args }
        if name == "Key" && args.len() == 1)
  };
  indices.iter().find(|idx| {
    let valid = position_or_key(idx)
      || matches!(idx, Expr::Identifier(s) if s == "All")
      || matches!(idx, Expr::FunctionCall { name, .. }
        if name == "Span" || name == "UpTo")
      || matches!(idx, Expr::List(items) if items.iter().all(position_or_key));
    !valid
  })
}

/// Append an evaluated Part index to `indices`. Part has no SequenceHold,
/// so an index that evaluated to a Sequence splices into several:
/// `m[[Sequence @@ {i, j}]]` is `m[[i, j]]`, and `m[[Sequence[]]]` is `m`.
pub fn push_part_index(indices: &mut Vec<Expr>, evaluated: Expr) {
  match &evaluated {
    Expr::FunctionCall { name, args } if name == "Sequence" => {
      indices.extend(args.iter().cloned());
    }
    _ => indices.push(evaluated),
  }
}

/// Emit Part::pkspec1 for `index`.
pub fn emit_pkspec1(index: &Expr) {
  crate::emit_message_to_stdout(&format!(
    "Part::pkspec1: The expression {} cannot be used as a part specification.",
    crate::syntax::format_expr(index, crate::syntax::ExprForm::Output)
  ));
}

/// Extract part from expression on AST (expr[[index]])
pub fn extract_part_ast(
  expr: &Expr,
  index: &Expr,
) -> Result<Expr, InterpreterError> {
  // Normalize a nested Alternatives chain (`a | b | c`) into a flat
  // Alternatives[a, b, c] so positional access matches wolframscript.
  if let Some(flat) = flatten_alternatives_binop(expr) {
    return extract_part_ast(&flat, index);
  }

  // A Dataset indexes the data it wraps, not its `Dataset[data, type,
  // metadata]` structure. A part that is itself a collection stays a Dataset,
  // a scalar part comes out bare.
  if let Some(data) = crate::functions::dataset_ast::dataset_contents(expr) {
    let part = extract_part_ast(&data, index)?;
    return Ok(match &part {
      Expr::List(_) | Expr::Association(_) => {
        crate::functions::dataset_ast::dataset_ast(std::slice::from_ref(&part))
      }
      _ => part,
    });
  }

  if let Some(symbolic) = graphics_symbolic_form(expr) {
    return extract_part_ast(&symbolic, index);
  }

  // A tree is an atom: Part cannot reach inside it, so the call stays
  // unevaluated and the caller reports Part::partd.
  if crate::functions::predicate_ast::is_fully_atomic_object(expr) {
    return Ok(Expr::Part {
      expr: Box::new(expr.clone()),
      index: Box::new(index.clone()),
    });
  }

  // A rational or a complex number is an atom as well, however it happens to
  // be stored: `(1/2)[[1]]` must not reach the `Rational[1, 2]` numerator,
  // and `(3 + 4 I)[[1]]` must not reach into the Plus-Times tree. Only part 0
  // — the head — resolves; anything else stays unevaluated for the caller to
  // report as Part::partd.
  if is_atomic_number_expr(expr) {
    if matches!(index, Expr::Integer(0)) {
      return crate::functions::predicate_ast::head_ast(std::slice::from_ref(
        expr,
      ));
    }
    return Ok(Expr::Part {
      expr: Box::new(expr.clone()),
      index: Box::new(index.clone()),
    });
  }
  extract_part_ast_rest(expr, index)
}

/// Whether `expr` is one of the numbers that Wolfram stores as a compound
/// expression but treats as an atom: a `Rational[n, d]` or a complex number.
/// The cheap shape test comes first so the hot Part paths (lists,
/// associations, …) never pay for the complex-number extraction.
pub(crate) fn is_atomic_number_expr(expr: &Expr) -> bool {
  matches!(
    expr,
    Expr::FunctionCall { .. } | Expr::BinaryOp { .. } | Expr::UnaryOp { .. }
  ) && crate::functions::predicate_ast::is_atomic_number(expr)
}

/// The symbolic `Graphics[…]` form of a rendered graphic, so the sequence
/// functions (`Part`, `First`, `Last`) can reach its primitives the way
/// Wolfram's do — `First[Plot[…]]` is the plot's line, not an error.
pub fn graphics_symbolic_form(expr: &Expr) -> Option<Expr> {
  // An image is `Image[data, type, ColorSpace -> …, Interleaving -> …]`, so
  // `img[[1]]` is its pixel array in the image's native type.
  if let Expr::Image {
    color_space,
    channels,
    image_type,
    ..
  } = expr
  {
    let type_name = match image_type {
      ImageType::Bit => "Bit",
      ImageType::Byte => "Byte",
      ImageType::Bit16 => "Bit16",
      ImageType::Real32 => "Real32",
      ImageType::Real64 => "Real64",
    };
    let pixels = crate::functions::image_ast::image_data_ast(&[
      expr.clone(),
      Expr::String(type_name.to_string()),
    ])
    .ok()?;
    let color_space = match color_space {
      Some(cs) => Expr::String(cs.to_string()),
      None => Expr::Identifier("Automatic".to_string()),
    };
    let interleaving = Expr::Identifier(
      if *channels == 1 { "None" } else { "True" }.to_string(),
    );
    let rule = |name: &str, value: Expr| Expr::Rule {
      pattern: Box::new(Expr::Identifier(name.to_string())),
      replacement: Box::new(value),
    };
    return Some(Expr::FunctionCall {
      name: "Image".to_string(),
      args: vec![
        pixels,
        Expr::String(type_name.to_string()),
        rule("ColorSpace", color_space),
        rule("Interleaving", interleaving),
      ]
      .into(),
    });
  }

  // A graphic that remembers its symbolic form (e.g. a plot's
  // `Graphics3D[GraphicsComplex[…], opts]`) uses it directly.
  if let Expr::Graphics {
    structure: Some(symbolic),
    ..
  } = expr
  {
    return Some(symbolic.as_ref().clone());
  }

  // A rendered 2D plot that kept its sampled curves (`PlotSource`)
  // reconstitutes them as primitives, so `ParametricPlot[…][[1]]` yields
  // something a surrounding `Graphics[{…}]` can draw — matching Wolfram
  // where part 1 of a plot is its primitives.
  if let Expr::Graphics {
    structure: None,
    source: Some(source),
    is_3d: false,
    ..
  } = expr
  {
    let series: Vec<Expr> = source
      .series
      .iter()
      .flat_map(|s| {
        let points = Expr::List(
          s.points
            .iter()
            .filter(|(x, y)| x.is_finite() && y.is_finite())
            .map(|(x, y)| {
              Expr::List(vec![Expr::Real(*x), Expr::Real(*y)].into())
            })
            .collect(),
        );
        let rgb_expr = |(r, g, b): (u8, u8, u8)| {
          call(
            "RGBColor",
            vec![
              Expr::Real(f64::from(r) / 255.0),
              Expr::Real(f64::from(g) / 255.0),
              Expr::Real(f64::from(b) / 255.0),
            ],
          )
        };
        let draw = call1(if s.is_scatter { "Point" } else { "Line" }, points);
        let mut prims = vec![Expr::List(vec![rgb_expr(s.color), draw].into())];
        // `Filling` draws a shaded region under the curve — bake it in as
        // an explicit `Polygon` primitive alongside the `Line`, the way
        // Wolfram does. Without this, a Demonstration's `region =
        // First[Plot[…, Filling -> …]]` (extracting the drawing to
        // transform and redraw elsewhere, e.g. with
        // `GeometricTransformation`) silently loses the shading.
        if let Some(polygon_points) = series_fill_polygon(s, &source.series) {
          let fill_color = s.fill_color.unwrap_or(s.color);
          let opacity = s.fill_opacity.unwrap_or(0.2);
          let styled_fill =
            call("Opacity", vec![Expr::Real(opacity), rgb_expr(fill_color)]);
          let polygon = call1(
            "Polygon",
            Expr::List(
              polygon_points
                .into_iter()
                .map(|(x, y)| {
                  Expr::List(vec![Expr::Real(x), Expr::Real(y)].into())
                })
                .collect(),
            ),
          );
          prims.push(Expr::List(vec![styled_fill, polygon].into()));
        }
        prims
      })
      .collect();
    // The options a plot shares with `Graphics` (an `Epilog`, a
    // `Background`, a `PlotRange`, …) stay with the drawing, so editing it —
    // `Append[plot, PlotLabel -> …]` — and redrawing keeps them. Options that
    // only steer sampling or styling (`MaxRecursion`, `PlotStyle`, …) were
    // already applied to the curves.
    // They form one list, as in Wolfram's `Graphics[prims, {opts}]`: part 2
    // of a plot is its option list.
    let options: Vec<Expr> = source
      .options
      .iter()
      .filter(|opt| {
        crate::functions::graphics::option_name_value(opt)
          .is_some_and(|(name, _)| GRAPHICS_PLOT_OPTIONS.contains(&name))
      })
      .cloned()
      .collect();
    return Some(call(
      "Graphics",
      vec![Expr::List(series.into()), Expr::List(options.into())],
    ));
  }
  None
}

/// Plot options that mean the same thing on the `Graphics` a plot is
/// redrawn as.
const GRAPHICS_PLOT_OPTIONS: &[&str] = &[
  "AspectRatio",
  "Axes",
  "AxesLabel",
  "AxesOrigin",
  "AxesStyle",
  "Background",
  "Epilog",
  "Frame",
  "FrameLabel",
  "FrameStyle",
  "FrameTicks",
  "GridLines",
  "ImageSize",
  "LabelStyle",
  "PlotLabel",
  "PlotRange",
  "PlotRangePadding",
  "Prolog",
  "Ticks",
  "TicksStyle",
];

/// The polygon points for a `PlotSeriesData`'s `Filling` level: the curve's
/// points followed by the same x-values back along the reference level, so
/// the closed loop matches what `AreaSeries` draws when rendering the SVG.
/// `None` for unfilled or scatter series (a scatter plot's `Filling` draws
/// per-point stems, not a shaded area, so it has no single polygon).
fn series_fill_polygon(
  s: &crate::syntax::PlotSeriesData,
  all_series: &[crate::syntax::PlotSeriesData],
) -> Option<Vec<(f64, f64)>> {
  if s.is_scatter {
    return None;
  }
  let finite: Vec<(f64, f64)> = s
    .points
    .iter()
    .copied()
    .filter(|(x, y)| x.is_finite() && y.is_finite())
    .collect();
  if finite.len() < 2 {
    return None;
  }
  let (y_min, y_max) = finite
    .iter()
    .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &(_, y)| {
      (lo.min(y), hi.max(y))
    });
  let ref_y = match s.filling {
    crate::syntax::SeriesFilling::None => return None,
    crate::syntax::SeriesFilling::Axis => 0.0,
    crate::syntax::SeriesFilling::Bottom => y_min,
    crate::syntax::SeriesFilling::Top => y_max,
    crate::syntax::SeriesFilling::Value(v) => v,
    crate::syntax::SeriesFilling::Series(j) => {
      let target = all_series.get(j)?;
      return crate::functions::plot::fill_between_polygon(
        &s.points,
        &target.points,
      );
    }
  };
  let mut polygon = finite.clone();
  polygon.extend(finite.iter().rev().map(|&(x, _)| (x, ref_y)));
  Some(polygon)
}

/// A part of `SparseArray[Automatic, dims, background, data]` taken from its
/// stored entries, without building the dense array: each index is an
/// integer (from the end when negative), `All`, a `Span` of integers or a
/// list of integers. `None` for anything else or a position out of range,
/// which the dense route reports.
fn sparse_array_part(
  sa: &[Expr],
  indices: &[Expr],
) -> Option<Result<Expr, InterpreterError>> {
  let [Expr::Identifier(auto), Expr::List(dims), background, data] = sa else {
    return None;
  };
  if auto != "Automatic" || indices.len() > dims.len() {
    return None;
  }
  let dims: Vec<usize> = dims
    .iter()
    .map(|d| match d {
      Expr::Integer(k) if *k >= 0 => Some(*k as usize),
      _ => None,
    })
    .collect::<Option<_>>()?;
  // 0-based positions an index selects; `None` inside for a scalar index.
  let resolve = |k: i128, len: usize| -> Option<usize> {
    let len = len as i128;
    let k = if k < 0 { len + k + 1 } else { k };
    (1..=len).contains(&k).then(|| (k - 1) as usize)
  };
  // (kept in the result, selected indices); a whole dimension is kept as
  // is without listing its indices (`full[d]`).
  let mut selections: Vec<(bool, Vec<usize>)> = Vec::with_capacity(dims.len());
  let mut full: Vec<bool> = Vec::with_capacity(dims.len());
  for (d, &len) in dims.iter().enumerate() {
    let whole = match indices.get(d) {
      None => true,
      Some(Expr::Identifier(a)) => a == "All",
      _ => false,
    };
    full.push(whole);
    if whole {
      selections.push((true, Vec::new()));
      continue;
    }
    let spec = &indices[d];
    let sel = match spec {
      Expr::Integer(k) => (false, vec![resolve(*k, len)?]),
      Expr::List(ks) => (
        true,
        ks.iter()
          .map(|k| match k {
            Expr::Integer(k) => resolve(*k, len),
            _ => None,
          })
          .collect::<Option<_>>()?,
      ),
      Expr::FunctionCall { name, args }
        if name == "Span" && (2..=3).contains(&args.len()) =>
      {
        let bound = |e: &Expr, default: usize| -> Option<usize> {
          match e {
            Expr::Identifier(a) if a == "All" => Some(default),
            Expr::Integer(k) => resolve(*k, len),
            _ => None,
          }
        };
        let start = bound(&args[0], 0)?;
        let end = bound(&args[1], len.checked_sub(1)?)?;
        let step = match args.get(2) {
          None => 1,
          Some(Expr::Integer(st)) if *st > 0 => *st as usize,
          _ => return None,
        };
        if end + 1 < start {
          return None;
        }
        (true, (start..=end).step_by(step).collect())
      }
      _ => return None,
    };
    selections.push(sel);
  }
  // A single element is read from its row of the compressed storage
  // (`{1, {rowPointers, columnIndices}, values}`) without listing the rest.
  if selections.iter().all(|(kept, _)| !kept)
    && let Expr::List(storage) = data
    && let [Expr::Integer(1), Expr::List(structure), Expr::List(values)] =
      storage.as_slice()
    && let [Expr::List(row_ptr), Expr::List(columns)] = structure.as_slice()
  {
    let target: Vec<usize> = selections.iter().map(|(_, sel)| sel[0]).collect();
    let (row, rest) = if dims.len() == 1 {
      (0, &target[..])
    } else {
      (target[0], &target[1..])
    };
    let bound = |i: usize| match row_ptr.get(i) {
      Some(Expr::Integer(k)) if *k >= 0 => Some(*k as usize),
      _ => None,
    };
    if let (Some(lo), Some(hi)) = (bound(row), bound(row + 1)) {
      for e in lo..hi.min(columns.len()).min(values.len()) {
        let Expr::List(cols) = &columns[e] else { break };
        if cols.len() == rest.len()
          && cols
            .iter()
            .zip(rest)
            .all(|(c, &t)| matches!(c, Expr::Integer(k) if *k == t as i128 + 1))
        {
          return Some(Ok(values[e].clone()));
        }
      }
      return Some(Ok(background.clone()));
    }
  }
  // Every selected original index → its places in the result.
  let places: Vec<std::collections::HashMap<usize, Vec<usize>>> = selections
    .iter()
    .zip(&full)
    .map(|((_, sel), &whole)| {
      let mut m: std::collections::HashMap<usize, Vec<usize>> =
        std::collections::HashMap::new();
      if !whole {
        for (out, &i) in sel.iter().enumerate() {
          m.entry(i).or_default().push(out);
        }
      }
      m
    })
    .collect();
  let new_dims: Vec<usize> = selections
    .iter()
    .zip(&full)
    .zip(&dims)
    .filter(|(((kept, _), _), _)| *kept)
    .map(|(((_, sel), &whole), &len)| if whole { len } else { sel.len() })
    .collect();
  // With the first index fixed, only that row's entries are read.
  let row_entries = || -> Option<Vec<(Vec<i128>, Expr)>> {
    if dims.len() < 2 || selections[0].0 {
      return None;
    }
    let Expr::List(storage) = data else {
      return None;
    };
    let [Expr::Integer(1), Expr::List(structure), Expr::List(values)] =
      storage.as_slice()
    else {
      return None;
    };
    let [Expr::List(row_ptr), Expr::List(columns)] = structure.as_slice()
    else {
      return None;
    };
    let row = selections[0].1[0];
    let bound = |i: usize| match row_ptr.get(i) {
      Some(Expr::Integer(k)) if *k >= 0 => Some(*k as usize),
      _ => None,
    };
    let (lo, hi) = (bound(row)?, bound(row + 1)?);
    (lo..hi)
      .map(|e| {
        let Expr::List(cols) = columns.get(e)? else {
          return None;
        };
        let mut pos = vec![row as i128 + 1];
        for c in cols {
          match c {
            Expr::Integer(k) => pos.push(*k),
            _ => return None,
          }
        }
        Some((pos, values.get(e)?.clone()))
      })
      .collect()
  };
  let entries = row_entries().unwrap_or_else(|| {
    crate::functions::list_helpers_ast::sparse_array_extract_rules(&dims, data)
  });
  let mut rules: Vec<Expr> = Vec::new();
  for (pos, value) in entries {
    let mut outs: Vec<Vec<usize>> = vec![Vec::new()];
    let mut hit = true;
    for (d, &p) in pos.iter().enumerate() {
      let own = [(p - 1) as usize];
      let targets: &[usize] = if full[d] {
        &own
      } else if let Some(t) = places[d].get(&((p - 1) as usize)) {
        t
      } else {
        hit = false;
        break;
      };
      if selections[d].0 {
        outs = outs
          .into_iter()
          .flat_map(|o| {
            targets.iter().map(move |t| {
              let mut o = o.clone();
              o.push(*t);
              o
            })
          })
          .collect();
      }
    }
    if !hit {
      continue;
    }
    if new_dims.is_empty() {
      return Some(Ok(value));
    }
    for o in outs {
      rules.push(Expr::Rule {
        pattern: Box::new(Expr::List(
          o.iter().map(|i| Expr::Integer(*i as i128 + 1)).collect(),
        )),
        replacement: Box::new(value.clone()),
      });
    }
  }
  if new_dims.is_empty() {
    return Some(Ok(background.clone()));
  }
  Some(crate::evaluator::evaluate_function_call_ast(
    "SparseArray",
    &[
      Expr::List(rules.into()),
      Expr::List(new_dims.iter().map(|d| Expr::Integer(*d as i128)).collect()),
      background.clone(),
    ],
  ))
}

/// The rest of `extract_part_ast`, after the graphics and Tree atoms have
/// been handled.
fn extract_part_ast_rest(
  expr: &Expr,
  index: &Expr,
) -> Result<Expr, InterpreterError> {
  // A SparseArray indexed with Part is densified, then the part is taken from
  // the dense nested list. This covers every rank and index form (a single
  // integer, a multi-index chain like [[1, 2, 3]], spans, All, ...); without it
  // a higher-rank SparseArray would be treated as an ordinary expression and
  // Part would wrongly index into its stored {Automatic, dims, default, data}
  // arguments (e.g. [[2, 2]] returning the second dimension).
  if let Expr::FunctionCall { name, args: sa } = expr
    && name == "SparseArray"
    && sa.len() == 4
    && let Some(part) = sparse_array_part(sa, std::slice::from_ref(index))
  {
    return part;
  }
  if let Expr::FunctionCall { name, args: sa } = expr
    && name == "SparseArray"
    && sa.len() == 4
  {
    let dense = crate::functions::list_helpers_ast::sparse_array_ast(sa)?;
    if matches!(&dense, Expr::List(_)) {
      let part = extract_part_ast(&dense, index)?;
      if matches!(part, Expr::List(_)) {
        return crate::evaluator::evaluate_function_call_ast(
          "SparseArray",
          &[part],
        );
      }
      return Ok(part);
    }
  }

  // A NumericArray stands for its array: a part that is itself an array
  // stays a NumericArray of the same type, a single element is a number.
  if let Expr::FunctionCall { name, args: na } = expr
    && name == "NumericArray"
    && (na.len() == 1 || na.len() == 2)
    && matches!(&na[0], Expr::List(_))
  {
    let part = extract_part_ast(&na[0], index)?;
    if matches!(part, Expr::List(_)) {
      let mut wrap = vec![part];
      wrap.extend(na.get(1).cloned());
      return crate::evaluator::evaluate_function_call_ast(
        "NumericArray",
        &wrap,
      );
    }
    return Ok(part);
  }

  // A structured matrix (CauchyMatrix[…], BlockDiagonalMatrix[…], the
  // LUDecomposition wrappers, …) is likewise densified first, so Part indexes
  // the matrix it represents rather than its stored StructuredData payload.
  if let Some(dense) =
    crate::functions::linear_algebra_ast::structured_matrix_to_dense(expr)
  {
    return extract_part_ast(&dense, index);
  }

  // `UpTo[n]` is a *Take* specification, not a part specification: Take,
  // Drop and Partition accept it, `[[…]]` does not. wolframscript rejects
  // every `expr[[UpTo[n]]]`, however well-formed, so Woxi does too.
  if let Expr::FunctionCall { name, .. } = index
    && name == "UpTo"
  {
    crate::emit_message_to_stdout(&format!(
      "Part::pkspec1: The expression {} cannot be used as a part specification.",
      crate::syntax::format_expr(index, crate::syntax::ExprForm::Output)
    ));
    return Ok(part_take_unevaluated(expr, index));
  }

  // For associations, handle key-based lookup and integer position indexing
  if let Expr::Association(items) = expr {
    // Positional lookup shared by the single-integer and list-index forms.
    // Returns the (key, value) entry, or None when out of bounds (the caller
    // emits Part::partw).
    let entry_at = |i: i128| -> Option<&(Expr, Expr)> {
      if i == 0 {
        return None;
      }
      let idx = if i > 0 {
        (i as usize) - 1
      } else {
        let pos = items.len() as i128 + i;
        if pos < 0 {
          return None;
        }
        pos as usize
      };
      items.get(idx)
    };
    let missing_for =
      |key: &Expr| call("Missing", vec![str_expr("KeyAbsent"), key.clone()]);
    // Key-based lookup shared by the string, Key[...], and list-index forms.
    // String indices match only string keys; Key[k] matches any key exactly.
    let lookup = |key: &Expr| -> Option<Expr> {
      items
        .iter()
        .find(|(k, _)| crate::evaluator::pattern_matching::expr_equal(k, key))
        .map(|(k, v)| {
          crate::functions::association_ast::assoc_entry_value(k, v)
        })
    };
    match index {
      // Integer index: access by position (return value, not rule)
      Expr::Integer(i) => {
        if *i == 0 {
          return Ok(id_expr("Association"));
        }
        if let Some((k, v)) = entry_at(*i) {
          return Ok(crate::functions::association_ast::assoc_entry_value(
            k, v,
          ));
        }
        part_missing_warn(&i.to_string(), expr);
        return Ok(part_take_unevaluated(expr, index));
      }
      // String index: lookup among string keys; absent -> Missing[KeyAbsent, key]
      Expr::String(_) => {
        if let Some(v) = lookup(index) {
          return Ok(v.clone());
        }
        return Ok(missing_for(index));
      }
      // Key[k] index: exact key-expression lookup (works for non-string keys)
      Expr::FunctionCall { name, args } if name == "Key" && args.len() == 1 => {
        if let Some(v) = lookup(&args[0]) {
          return Ok(v.clone());
        }
        return Ok(missing_for(index));
      }
      // Span: positional slice preserving the key -> value entries
      Expr::FunctionCall { name, args }
        if name == "Span" && (args.len() == 2 || args.len() == 3) =>
      {
        let rule_items: Vec<Expr> = items
          .iter()
          .map(|(k, v)| rule_expr(k.clone(), v.clone()))
          .collect();
        return extract_span_from_items(
          expr,
          index,
          &rule_items,
          &args[0],
          &args[1],
          args.get(2),
          |selected| {
            Expr::Association(
              selected
                .into_iter()
                .map(|r| match &r {
                  Expr::Rule {
                    pattern,
                    replacement,
                  } => ((**pattern).clone(), (**replacement).clone()),
                  _ => (r.clone(), null_expr()),
                })
                .collect(),
            )
          },
        );
      }
      // List index: sub-association of the selected entries
      Expr::List(idxs) => {
        let mut selected: Vec<(Expr, Expr)> = Vec::with_capacity(idxs.len());
        for ix in idxs {
          match ix {
            Expr::Integer(i) => {
              if let Some((k, v)) = entry_at(*i) {
                selected.push((k.clone(), v.clone()));
              } else {
                part_missing_warn(&i.to_string(), expr);
                return Ok(part_take_unevaluated(expr, index));
              }
            }
            Expr::String(_) => match lookup(ix) {
              Some(v) => selected.push((ix.clone(), v.clone())),
              None => selected.push((ix.clone(), missing_for(ix))),
            },
            Expr::FunctionCall { name, args }
              if name == "Key" && args.len() == 1 =>
            {
              match lookup(&args[0]) {
                Some(v) => selected.push((args[0].clone(), v.clone())),
                None => selected.push((args[0].clone(), missing_for(ix))),
              }
            }
            _ => {
              crate::emit_message_to_stdout(&format!(
                "Part::pkspec1: The expression {} cannot be used as a part specification.",
                crate::syntax::format_expr(ix, crate::syntax::ExprForm::Output)
              ));
              return Ok(part_take_unevaluated(expr, index));
            }
          }
        }
        return Ok(Expr::Association(selected));
      }
      // Anything else is not a valid association part specification
      _ => {
        crate::emit_message_to_stdout(&format!(
          "Part::pkspec1: The expression {} cannot be used as a part specification.",
          crate::syntax::format_expr(index, crate::syntax::ExprForm::Output)
        ));
        return Ok(part_take_unevaluated(expr, index));
      }
    }
  }

  // Span[start, end] or Span[start, end, step]
  if let Expr::FunctionCall { name, args } = index
    && name == "Span"
    && (args.len() == 2 || args.len() == 3)
  {
    match expr {
      Expr::List(items) => {
        return extract_span_from_items(
          expr,
          index,
          items,
          &args[0],
          &args[1],
          args.get(2),
          |v: Vec<Expr>| Expr::List(v.into()),
        );
      }
      Expr::FunctionCall {
        name,
        args: fn_args,
      } => {
        return extract_span_from_items(
          expr,
          index,
          fn_args,
          &args[0],
          &args[1],
          args.get(2),
          |selected| Expr::FunctionCall {
            name: name.clone(),
            args: selected.into(),
          },
        );
      }
      _ => return Ok(part_take_unevaluated(expr, index)),
    }
  }

  let idx = match index {
    Expr::Integer(n) => *n as i64,
    Expr::BigInteger(n) => {
      use num_traits::ToPrimitive;
      match n.to_i64() {
        Some(v) => v,
        None => return Ok(part_take_unevaluated(expr, index)),
      }
    }
    // Only integer-valued Reals are valid part specs (e.g. 2.0 -> 2).
    // A non-integer Real (1.5) or a Rational (3/2) errors Part::pkspec1
    // (matching wolframscript), rather than silently truncating 1.5 -> 1.
    Expr::Real(f) if f.is_finite() && f.fract() == 0.0 => *f as i64,
    Expr::Real(_) => {
      crate::emit_message_to_stdout(&format!(
        "Part::pkspec1: The expression {} cannot be used as a part specification.",
        crate::syntax::format_expr(index, crate::syntax::ExprForm::Output)
      ));
      return Ok(part_take_unevaluated(expr, index));
    }
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      crate::emit_message_to_stdout(&format!(
        "Part::pkspec1: The expression {} cannot be used as a part specification.",
        crate::syntax::format_expr(index, crate::syntax::ExprForm::Output)
      ));
      return Ok(part_take_unevaluated(expr, index));
    }
    Expr::List(indices) => {
      // Part[expr, {i1, i2, ...}] collects the elements at each position. When
      // every position is a concrete integer and the expression has a known
      // length, validate the bounds up front: a single out-of-range position
      // fails the WHOLE spec with one Part::partw naming the full index list
      // (matching wolframscript), instead of returning a partially-resolved
      // list with a per-index message.
      //
      // A key only addresses an association (handled above), so a string
      // among the positions fails the whole spec here.
      if indices.iter().any(|i| matches!(i, Expr::String(_))) {
        crate::emit_message_to_stdout(&format!(
          "Part::pspec1: Part specification {} is not applicable.",
          crate::syntax::format_expr(index, crate::syntax::ExprForm::Output)
        ));
        return Ok(part_take_unevaluated(expr, index));
      }
      if let Some(len) = positional_length(expr) {
        let positions: Option<Vec<i64>> =
          indices.iter().map(simple_position).collect();
        if let Some(positions) = positions {
          let len = len as i64;
          let in_range = |i: i64| {
            i == 0 || (1..=len).contains(&i) || (-len..=-1).contains(&i)
          };
          if !positions.iter().all(|&i| in_range(i)) {
            part_missing_warn(
              &crate::syntax::format_expr(
                index,
                crate::syntax::ExprForm::Output,
              ),
              expr,
            );
            return Ok(part_take_unevaluated(expr, index));
          }
          // All positions valid: extract each part and keep the head of the
          // original expression (f[a,b,c][[{1,3}]] -> f[a, c]).
          let mut results = Vec::with_capacity(indices.len());
          for idx_expr in indices {
            results.push(extract_part_ast(expr, idx_expr)?);
          }
          return Ok(rewrap_part_list(expr, results));
        }
      }
      // Fallback: non-positional expression (e.g. ByteArray) or non-integer
      // index entries — extract each part under a List head (legacy behavior).
      let mut results = Vec::new();
      for idx_expr in indices {
        results.push(extract_part_ast(expr, idx_expr)?);
      }
      return Ok(Expr::List(results.into()));
    }
    _ => return Ok(part_take_unevaluated(expr, index)),
  };

  match expr {
    Expr::List(items) => {
      if idx == 0 {
        // Part[{...}, 0] returns the head, which is List
        return Ok(id_expr("List"));
      }
      let len = items.len() as i64;
      let actual_idx = if idx < 0 { len + idx } else { idx - 1 };
      if actual_idx >= 0 && actual_idx < len {
        Ok(items[actual_idx as usize].clone())
      } else {
        // Print warning to stderr and return unevaluated Part expression
        part_missing_warn(&idx.to_string(), expr);
        Ok(part_take_unevaluated(expr, index))
      }
    }
    Expr::FunctionCall { name, args } => {
      if idx == 0 {
        // Part[f[...], 0] returns the head
        return Ok(Expr::Identifier(name.clone()));
      }
      // ByteArray["base64"] indexes into the decoded byte sequence,
      // matching wolframscript: ByteArray[{1, 25, 3}][[2]] -> 25.
      if name == "ByteArray"
        && args.len() == 1
        && let Expr::String(b64) = &args[0]
      {
        use base64::Engine;
        let engine = base64::engine::general_purpose::STANDARD;
        if let Ok(bytes) = engine.decode(b64) {
          let len = bytes.len() as i64;
          let actual_idx = if idx < 0 { len + idx } else { idx - 1 };
          if actual_idx >= 0 && actual_idx < len {
            return Ok(Expr::Integer(bytes[actual_idx as usize] as i128));
          }
          part_missing_warn(&idx.to_string(), expr);
          return Ok(part_take_unevaluated(expr, index));
        }
      }
      let len = args.len() as i64;
      let actual_idx = if idx < 0 { len + idx } else { idx - 1 };
      if actual_idx >= 0 && actual_idx < len {
        Ok(args[actual_idx as usize].clone())
      } else {
        part_missing_warn(&idx.to_string(), expr);
        Ok(part_take_unevaluated(expr, index))
      }
    }
    // `f[a][b, c]` has head `f[a]` and parts `b`, `c`.
    Expr::CurriedCall { func, args } => {
      if idx == 0 {
        return Ok(func.as_ref().clone());
      }
      let len = args.len() as i64;
      let actual_idx = if idx < 0 { len + idx } else { idx - 1 };
      if actual_idx >= 0 && actual_idx < len {
        Ok(args[actual_idx as usize].clone())
      } else {
        part_missing_warn(&idx.to_string(), expr);
        Ok(part_take_unevaluated(expr, index))
      }
    }
    Expr::Rule {
      pattern,
      replacement,
    }
    | Expr::RuleDelayed {
      pattern,
      replacement,
    } => {
      // Rule[a, b] has Part 0 = Rule/RuleDelayed, Part 1 = a, Part 2 = b
      if idx == 0 {
        let head_name = if matches!(expr, Expr::Rule { .. }) {
          "Rule"
        } else {
          "RuleDelayed"
        };
        return Ok(id_expr(head_name));
      }
      let parts: [&Expr; 2] = [pattern, replacement];
      let len = 2i64;
      let actual_idx = if idx < 0 { len + idx } else { idx - 1 };
      if actual_idx >= 0 && actual_idx < len {
        Ok(parts[actual_idx as usize].clone())
      } else {
        part_missing_warn(&idx.to_string(), expr);
        Ok(part_take_unevaluated(expr, index))
      }
    }
    // Strings are atoms in Wolfram Language — Part[string, n] for n ≠ 0
    // returns unevaluated (handled by the fallback arm below).
    Expr::BinaryOp { op, left, right } => {
      // Decompose BinaryOp into head + args consistent with Head[]
      let (head_name, parts): (&str, Vec<Expr>) = match op {
        BinaryOperator::Plus => ("Plus", vec![*left.clone(), *right.clone()]),
        BinaryOperator::Minus => {
          // a - b = Plus[a, Times[-1, b]]
          let neg_right = neg(*right.clone());
          ("Plus", vec![*left.clone(), neg_right])
        }
        BinaryOperator::Times => ("Times", vec![*left.clone(), *right.clone()]),
        BinaryOperator::Divide => {
          if matches!(left.as_ref(), Expr::Integer(1)) {
            // 1/b = Power[b, -1]
            ("Power", vec![*right.clone(), Expr::Integer(-1)])
          } else {
            // a/b = Times[a, Power[b, -1]]
            let inv_right = pow(*right.clone(), Expr::Integer(-1));
            ("Times", vec![*left.clone(), inv_right])
          }
        }
        BinaryOperator::Power => ("Power", vec![*left.clone(), *right.clone()]),
        BinaryOperator::And => ("And", vec![*left.clone(), *right.clone()]),
        BinaryOperator::Or => ("Or", vec![*left.clone(), *right.clone()]),
        BinaryOperator::StringJoin => {
          ("StringJoin", vec![*left.clone(), *right.clone()])
        }
        BinaryOperator::Alternatives => {
          ("Alternatives", vec![*left.clone(), *right.clone()])
        }
      };
      if idx == 0 {
        return Ok(id_expr(head_name));
      }
      let len = parts.len() as i64;
      let actual_idx = if idx < 0 { len + idx } else { idx - 1 };
      if actual_idx >= 0 && actual_idx < len {
        Ok(parts[actual_idx as usize].clone())
      } else {
        part_missing_warn(&idx.to_string(), expr);
        Ok(part_take_unevaluated(expr, index))
      }
    }
    Expr::UnaryOp { op, operand } => {
      let (head_name, parts): (&str, Vec<Expr>) = match op {
        UnaryOperator::Minus => {
          ("Times", vec![Expr::Integer(-1), *operand.clone()])
        }
        UnaryOperator::Not => ("Not", vec![*operand.clone()]),
      };
      if idx == 0 {
        return Ok(id_expr(head_name));
      }
      let len = parts.len() as i64;
      let actual_idx = if idx < 0 { len + idx } else { idx - 1 };
      if actual_idx >= 0 && actual_idx < len {
        Ok(parts[actual_idx as usize].clone())
      } else {
        part_missing_warn(&idx.to_string(), expr);
        Ok(part_take_unevaluated(expr, index))
      }
    }
    // A pattern is an ordinary expression, not an atom: `x_Symbol` is
    // `Pattern[x, Blank[Symbol]]`, so `[[1]]` is `x` and `[[2]]` is `_Symbol`.
    // A pure function is likewise `Function[body]` (or `Function[params,
    // body]`), so `(#1 + 1 &)[[1]]` is its body.
    Expr::Pattern { .. }
    | Expr::PatternTest { .. }
    | Expr::PatternOptional { .. }
    | Expr::Function { .. }
    | Expr::NamedFunction { .. } => {
      let crate::functions::expr_form::ExprForm::Composite { head, children } =
        crate::functions::expr_form::decompose_expr(expr)
      else {
        return Ok(part_take_unevaluated(expr, index));
      };
      if idx == 0 {
        return Ok(Expr::Identifier(head));
      }
      let len = children.len() as i64;
      let actual_idx = if idx < 0 { len + idx } else { idx - 1 };
      if actual_idx >= 0 && actual_idx < len {
        Ok(children[actual_idx as usize].clone())
      } else {
        part_missing_warn(&idx.to_string(), expr);
        Ok(part_take_unevaluated(expr, index))
      }
    }
    Expr::Comparison {
      operands,
      operators,
    } => {
      // Index into the WL structure: uniform `a op b op c` is Op[a, b, c];
      // mixed `a < b <= c` is Inequality[a, Less, b, LessEqual, c].
      let (head, parts) =
        crate::syntax::comparison_head_and_args(operands, operators);
      if idx == 0 {
        return Ok(Expr::Identifier(head));
      }
      let len = parts.len() as i64;
      let actual_idx = if idx < 0 { len + idx } else { idx - 1 };
      if actual_idx >= 0 && actual_idx < len {
        Ok(parts[actual_idx as usize].clone())
      } else {
        part_missing_warn(&idx.to_string(), expr);
        Ok(part_take_unevaluated(expr, index))
      }
    }
    _ => {
      // Atoms: Part[atom, 0] returns the Head of the atom
      if idx == 0 {
        return crate::functions::predicate_ast::head_ast(
          std::slice::from_ref(expr),
        );
      }
      Ok(part_take_unevaluated(expr, index))
    }
  }
}
