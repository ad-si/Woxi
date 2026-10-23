#[allow(unused_imports)]
use super::*;

/// TuringMachine[rule, init, steps]
///
/// rule: integer n (default 2 states, 2 colors) or {n, s, k}
/// init: {{state_spec}, tape} where state_spec is {s} or {s, pos}
/// steps: number of steps to simulate
///
/// Returns list of {{state, head_pos, cumulative_shift}, tape} for each step (including initial).
/// Tape uses periodic boundaries.
pub fn turing_machine_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  // TuringMachine[rule, init] — one step, returning just the new state.
  if args.len() == 2 {
    let evolved = turing_machine_ast(&[
      args[0].clone(),
      args[1].clone(),
      Expr::Integer(1),
    ])?;
    if let Expr::List(states) = &evolved
      && states.len() == 2
    {
      let last = states[1].clone();
      // An infinite-tape run reports its tape in `{cells, background}` form.
      if let Some(bg) = infinite_tape_background(&args[1])
        && let Expr::List(pair) = &last
        && pair.len() == 2
      {
        return Ok(Expr::List(
          vec![
            pair[0].clone(),
            Expr::List(vec![pair[1].clone(), bg].into()),
          ]
          .into(),
        ));
      }
      return Ok(last);
    }
    return Ok(unevaluated("TuringMachine", args));
  }

  if args.len() != 3 {
    return Ok(unevaluated("TuringMachine", args));
  }

  // Only a plain step count is accepted; `{t}` and friends are rejected.
  if !matches!(&args[2], Expr::Integer(n) if *n >= 0) {
    crate::emit_message(&format!(
      "TuringMachine::tspec: The time specification {} must be an integer >= 0.",
      expr_to_string(&args[2])
    ));
    return Ok(unevaluated("TuringMachine", args));
  }

  // An infinite background tape runs on a pad wide enough that the head
  // never wraps, then reports only the cells it actually visited.
  if infinite_tape_background(&args[1]).is_some() {
    return turing_machine_infinite(args);
  }

  let rule = parse_rule(&args[0])?;
  let (state, start_pos, tape) = parse_init(&args[1])?;
  rule.warn_initial_state(&args[1], &state);
  let steps = match &args[2] {
    Expr::Integer(n) => *n as usize,
    _ => return Ok(unevaluated("TuringMachine", args)),
  };
  if tape.is_empty() || start_pos < 1 || start_pos > tape.len() {
    return Ok(unevaluated("TuringMachine", args));
  }

  // A cyclic tape: the head wraps around at either end.
  let len = tape.len() as i64;
  let mut tape = tape;
  let mut state = state;
  let mut pos = start_pos as i64; // 1-indexed
  let mut shift: i64 = 0;
  let row = |state: &Expr, pos: i64, shift: i64, tape: &[Expr]| {
    Expr::List(
      vec![
        Expr::List(
          vec![
            state.clone(),
            Expr::Integer(pos as i128),
            Expr::Integer(shift as i128),
          ]
          .into(),
        ),
        Expr::List(tape.to_vec().into()),
      ]
      .into(),
    )
  };
  let mut out = Vec::with_capacity(steps + 1);
  out.push(row(&state, pos, shift, &tape));
  for _ in 0..steps {
    let cell = (pos - 1) as usize;
    if let Some((new_state, new_color, offset)) =
      rule.apply(&state, &tape[cell])?
    {
      tape[cell] = new_color;
      state = new_state;
      shift += offset;
      pos = (pos - 1 + offset).rem_euclid(len) + 1;
    }
    out.push(row(&state, pos, shift, &tape));
  }
  Ok(Expr::List(out.into()))
}

/// A machine's rule: a Wolfram rule number decoded into a transition table,
/// or an explicit list of `{state, color} -> {state', color', offset}` rules
/// (patterns allowed, states and colours any expressions).
enum MachineRule {
  Numbered {
    rule_num: u64,
    transitions: Vec<Transition>,
    num_states: usize,
    num_colors: usize,
  },
  Rules(Expr),
}

impl MachineRule {
  /// The transition for `state` reading `color`, or None when no rule
  /// applies (the machine then stays put, as in Wolfram).
  fn apply(
    &self,
    state: &Expr,
    color: &Expr,
  ) -> Result<Option<(Expr, Expr, i64)>, InterpreterError> {
    match self {
      MachineRule::Numbered {
        transitions,
        num_states,
        num_colors,
        ..
      } => {
        let (Expr::Integer(s), Expr::Integer(c)) = (state, color) else {
          return Ok(None);
        };
        if *s < 1
          || *s as usize > *num_states
          || *c < 0
          || *c as usize >= *num_colors
        {
          return Ok(None);
        }
        let t =
          &transitions[(*num_states - *s as usize) * num_colors + *c as usize];
        Ok(Some((
          Expr::Integer(t.new_state as i128),
          Expr::Integer(t.new_color as i128),
          t.direction as i64,
        )))
      }
      MachineRule::Rules(rules) => {
        let subject = Expr::List(vec![state.clone(), color.clone()].into());
        let result = crate::evaluator::evaluate_function_call_ast(
          "Replace",
          &[subject, rules.clone()],
        )?;
        match &result {
          Expr::List(parts) if parts.len() == 3 => match &parts[2] {
            Expr::Integer(offset) => {
              Ok(Some((parts[0].clone(), parts[1].clone(), *offset as i64)))
            }
            _ => Ok(None),
          },
          _ => Ok(None),
        }
      }
    }
  }
}

impl MachineRule {
  /// Wolfram's warning for a numbered rule started in a state it has no
  /// transitions for (the machine then never moves).
  fn warn_initial_state(&self, init: &Expr, state: &Expr) {
    if let MachineRule::Numbered {
      rule_num,
      num_states,
      num_colors,
      ..
    } = self
      && !matches!(state, Expr::Integer(s) if *s >= 1 && *s as usize <= *num_states)
    {
      crate::emit_message(&format!(
        "TuringMachine::ninits: Warning: in the initial state {}, the state {} is not an integer in the range 1 <= state <= {num_states} indicated by the rule {{{rule_num}, {num_states}, {num_colors}, 1}}.",
        expr_to_string(init),
        expr_to_string(state)
      ));
    }
  }
}

fn is_rule(e: &Expr) -> bool {
  matches!(e, Expr::Rule { .. } | Expr::RuleDelayed { .. })
    || matches!(e, Expr::FunctionCall { name, args }
      if (name == "Rule" || name == "RuleDelayed") && args.len() == 2)
}

fn parse_rule(expr: &Expr) -> Result<MachineRule, InterpreterError> {
  let explicit = match expr {
    Expr::List(items) => !items.is_empty() && items.iter().all(is_rule),
    e => is_rule(e),
  };
  if explicit {
    return Ok(MachineRule::Rules(expr.clone()));
  }
  let (rule_num, num_states, num_colors) = parse_rule_spec(expr)?;
  Ok(MachineRule::Numbered {
    rule_num,
    transitions: decode_rule(rule_num, num_states, num_colors),
    num_states,
    num_colors,
  })
}

/// Parse rule: integer n or {n, s, k}
fn parse_rule_spec(
  expr: &Expr,
) -> Result<(u64, usize, usize), InterpreterError> {
  match expr {
    Expr::Integer(n) => Ok((*n as u64, 2, 2)),
    Expr::List(items) if items.len() == 3 => {
      let n = match &items[0] {
        Expr::Integer(n) => *n as u64,
        _ => {
          return Err(InterpreterError::EvaluationError(
            "TuringMachine: rule number must be an integer".into(),
          ));
        }
      };
      let s = match &items[1] {
        Expr::Integer(s) => *s as usize,
        _ => {
          return Err(InterpreterError::EvaluationError(
            "TuringMachine: number of states must be an integer".into(),
          ));
        }
      };
      let k = match &items[2] {
        Expr::Integer(k) => *k as usize,
        _ => {
          return Err(InterpreterError::EvaluationError(
            "TuringMachine: number of colors must be an integer".into(),
          ));
        }
      };
      Ok((n, s, k))
    }
    _ => Ok((0, 2, 2)), // fallback
  }
}

/// Parse init: `{state_spec, tape}` where state_spec is `s`, `{s}` or
/// `{s, pos}`. Returns the state, the 1-indexed head position and the tape
/// cells (empty when the tape is not a plain list).
fn parse_init(
  expr: &Expr,
) -> Result<(Expr, usize, Vec<Expr>), InterpreterError> {
  let items = match expr {
    Expr::List(items) if items.len() == 2 => items,
    _ => {
      return Err(InterpreterError::EvaluationError(
        "TuringMachine: init must be {state_spec, tape}".into(),
      ));
    }
  };
  let (state, pos) = match &items[0] {
    Expr::List(spec) if spec.len() == 1 => (spec[0].clone(), 1usize),
    Expr::List(spec) if spec.len() == 2 => match &spec[1] {
      Expr::Integer(n) if *n >= 1 => (spec[0].clone(), *n as usize),
      _ => {
        return Err(InterpreterError::EvaluationError(
          "TuringMachine: position must be a positive integer".into(),
        ));
      }
    },
    s => (s.clone(), 1usize),
  };
  let tape = match &items[1] {
    Expr::List(cells) => cells.to_vec(),
    _ => Vec::new(),
  };
  Ok((state, pos, tape))
}

/// Transition output: (new_state, new_color, direction)
/// direction: -1 (left) or +1 (right)
struct Transition {
  new_state: usize,
  new_color: u8,
  direction: i32,
}

/// Decode rule number into transition table.
/// Index: (s - state) * k + color, where state is 1-indexed.
fn decode_rule(
  rule_num: u64,
  num_states: usize,
  num_colors: usize,
) -> Vec<Transition> {
  let base = (2 * num_states * num_colors) as u64;
  let num_transitions = num_states * num_colors;
  let mut transitions = Vec::with_capacity(num_transitions);
  let mut n = rule_num;

  for _ in 0..num_transitions {
    let digit = if base > 0 { n % base } else { 0 };
    n /= if base > 0 { base } else { 1 };

    let dir_bit = digit % 2;
    let new_color = ((digit / 2) % num_colors as u64) as u8;
    let new_state =
      ((digit / (2 * num_colors as u64)) % num_states as u64) as usize + 1;
    let direction = if dir_bit == 1 { 1 } else { -1 };

    transitions.push(Transition {
      new_state,
      new_color,
      direction,
    });
  }

  transitions
}

/// The background of an `{state, {cells, background}}` initial condition, or
/// `None` when the tape is an ordinary finite (cyclic) list. The background is
/// a single colour or a non-empty list of colours repeated periodically.
fn infinite_tape_background(init: &Expr) -> Option<Expr> {
  let Expr::List(items) = init else { return None };
  if items.len() != 2 {
    return None;
  }
  let Expr::List(tape) = &items[1] else {
    return None;
  };
  // `{cells, background}`: a list of cells followed by a single colour.
  (tape.len() == 2
    && matches!(&tape[0], Expr::List(_))
    && background_pattern(&tape[1]).is_some())
  .then(|| tape[1].clone())
}

/// The colours of a background spec: one colour, or a non-empty list of
/// colours that repeats periodically in both directions.
fn background_pattern(bg: &Expr) -> Option<Vec<Expr>> {
  match bg {
    Expr::List(items) if !items.is_empty() => Some(items.to_vec()),
    Expr::List(_) => None,
    other => Some(vec![other.clone()]),
  }
}

/// Run a machine on an infinite background tape. Cells are addressed by their
/// offset from the head's starting cell; only written cells are stored, the
/// rest come from the initial content or the background. Wolfram reports the
/// window spanning every offset the head reached plus the initial content.
fn turing_machine_infinite(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let rule = parse_rule(&args[0])?;
  let steps = match &args[2] {
    Expr::Integer(n) => *n as usize,
    _ => return Ok(unevaluated("TuringMachine", args)),
  };
  let Expr::List(items) = &args[1] else {
    return Ok(unevaluated("TuringMachine", args));
  };
  let Expr::List(tape_spec) = &items[1] else {
    return Ok(unevaluated("TuringMachine", args));
  };
  let (Expr::List(content), Some(bg)) =
    (&tape_spec[0], background_pattern(&tape_spec[1]))
  else {
    return Ok(unevaluated("TuringMachine", args));
  };
  // The state spec is shared with the finite form; only the tape differs.
  let (mut state, start_pos, _) = parse_init(&Expr::List(
    vec![items[0].clone(), Expr::List(vec![].into())].into(),
  ))?;
  rule.warn_initial_state(&args[1], &state);

  // Content index of the cell at offset 0, and the initial colour anywhere:
  // the background is aligned so the content's first cell holds the
  // pattern's first colour.
  let origin = start_pos as i64 - 1;
  let initial = |x: i64| -> Expr {
    let i = x + origin;
    if i >= 0 && (i as usize) < content.len() {
      content[i as usize].clone()
    } else {
      bg[i.rem_euclid(bg.len() as i64) as usize].clone()
    }
  };

  let mut written: std::collections::HashMap<i64, Expr> =
    std::collections::HashMap::new();
  let mut offset: i64 = 0;
  // Per step: the state, the head offset, and the write made before it.
  let mut history: Vec<(Expr, i64, Option<(i64, Expr)>)> =
    vec![(state.clone(), 0, None)];
  for _ in 0..steps {
    let color = written
      .get(&offset)
      .cloned()
      .unwrap_or_else(|| initial(offset));
    let mut write = None;
    if let Some((new_state, new_color, step)) = rule.apply(&state, &color)? {
      written.insert(offset, new_color.clone());
      write = Some((offset, new_color));
      state = new_state;
      offset += step;
    }
    history.push((state.clone(), offset, write));
  }

  let lo = history.iter().map(|h| h.1).min().unwrap_or(0).min(-origin);
  let hi = history
    .iter()
    .map(|h| h.1)
    .max()
    .unwrap_or(0)
    .max(content.len() as i64 - 1 - origin);
  let mut window: Vec<Expr> = (lo..=hi).map(initial).collect();
  let mut out = Vec::with_capacity(history.len());
  for (st, off, write) in history {
    if let Some((x, c)) = write {
      window[(x - lo) as usize] = c;
    }
    out.push(Expr::List(
      vec![
        Expr::List(
          vec![
            st,
            Expr::Integer((off - lo + 1) as i128),
            Expr::Integer(off as i128),
          ]
          .into(),
        ),
        Expr::List(window.clone().into()),
      ]
      .into(),
    ));
  }
  Ok(Expr::List(out.into()))
}
