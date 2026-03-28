use crate::InterpreterError;
use crate::syntax::Expr;

/// TuringMachine[rule, init, steps]
///
/// rule: integer n (default 2 states, 2 colors) or {n, s, k}
/// init: {{state_spec}, tape} where state_spec is {s} or {s, pos}
/// steps: number of steps to simulate
///
/// Returns list of {{state, head_pos, cumulative_shift}, tape} for each step (including initial).
/// Tape uses periodic boundaries.
pub fn turing_machine_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  if args.len() != 3 {
    return Ok(Expr::FunctionCall {
      name: "TuringMachine".to_string(),
      args: args.to_vec(),
    });
  }

  // Parse rule specification
  let (rule_num, num_states, num_colors) = parse_rule_spec(&args[0])?;

  // Parse initial condition
  let (init_state, init_pos, tape) = parse_init(&args[1])?;

  // Parse steps
  let steps = match &args[2] {
    Expr::Integer(n) => *n as usize,
    _ => {
      return Ok(Expr::FunctionCall {
        name: "TuringMachine".to_string(),
        args: args.to_vec(),
      });
    }
  };

  // Decode rule number to transition table
  let transitions = decode_rule(rule_num, num_states, num_colors);

  // Simulate
  let results = simulate(
    &transitions,
    num_states,
    num_colors,
    init_state,
    init_pos,
    &tape,
    steps,
  );

  // Convert to Expr
  let result_exprs: Vec<Expr> = results
    .into_iter()
    .map(|(state, pos, shift, tape)| {
      let state_info = Expr::List(vec![
        Expr::Integer(state as i128),
        Expr::Integer(pos as i128),
        Expr::Integer(shift as i128),
      ]);
      let tape_expr = Expr::List(
        tape.into_iter().map(|c| Expr::Integer(c as i128)).collect(),
      );
      Expr::List(vec![state_info, tape_expr])
    })
    .collect();

  Ok(Expr::List(result_exprs))
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

/// Parse init: {{state_spec}, tape}
/// state_spec: {s} or {s, pos}
fn parse_init(
  expr: &Expr,
) -> Result<(usize, usize, Vec<u8>), InterpreterError> {
  let items = match expr {
    Expr::List(items) if items.len() == 2 => items,
    _ => {
      return Err(InterpreterError::EvaluationError(
        "TuringMachine: init must be {state_spec, tape}".into(),
      ));
    }
  };

  // Parse state spec
  let (state, pos) = match &items[0] {
    Expr::List(spec) if spec.len() == 1 => {
      let s = match &spec[0] {
        Expr::Integer(n) => *n as usize,
        _ => {
          return Err(InterpreterError::EvaluationError(
            "TuringMachine: state must be an integer".into(),
          ));
        }
      };
      (s, 1usize) // default position is 1
    }
    Expr::List(spec) if spec.len() == 2 => {
      let s = match &spec[0] {
        Expr::Integer(n) => *n as usize,
        _ => {
          return Err(InterpreterError::EvaluationError(
            "TuringMachine: state must be an integer".into(),
          ));
        }
      };
      let p = match &spec[1] {
        Expr::Integer(n) => *n as usize,
        _ => {
          return Err(InterpreterError::EvaluationError(
            "TuringMachine: position must be an integer".into(),
          ));
        }
      };
      (s, p)
    }
    _ => {
      return Err(InterpreterError::EvaluationError(
        "TuringMachine: state_spec must be {s} or {s, pos}".into(),
      ));
    }
  };

  // Parse tape
  let tape = match &items[1] {
    Expr::List(cells) => cells
      .iter()
      .map(|e| match e {
        Expr::Integer(n) => Ok(*n as u8),
        _ => Err(InterpreterError::EvaluationError(
          "TuringMachine: tape cells must be integers".into(),
        )),
      })
      .collect::<Result<Vec<u8>, _>>()?,
    _ => {
      return Err(InterpreterError::EvaluationError(
        "TuringMachine: tape must be a list".into(),
      ));
    }
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

/// Get transition for given (state, color).
/// Transition index = (num_states - state) * num_colors + color
fn get_transition(
  transitions: &[Transition],
  num_states: usize,
  num_colors: usize,
  state: usize,
  color: u8,
) -> &Transition {
  let index = (num_states - state) * num_colors + color as usize;
  &transitions[index]
}

/// Simulate the Turing machine.
/// Returns Vec of (state, head_position_1indexed, cumulative_shift, tape).
fn simulate(
  transitions: &[Transition],
  num_states: usize,
  num_colors: usize,
  init_state: usize,
  init_pos: usize,
  init_tape: &[u8],
  steps: usize,
) -> Vec<(usize, usize, i64, Vec<u8>)> {
  let tape_len = init_tape.len();
  let mut tape = init_tape.to_vec();
  let mut state = init_state;
  let mut pos = init_pos; // 1-indexed
  let mut cumulative_shift: i64 = 0;

  let mut results = Vec::with_capacity(steps + 1);
  results.push((state, pos, cumulative_shift, tape.clone()));

  for _ in 0..steps {
    // Read current cell (1-indexed)
    let color = tape[pos - 1];

    // Get transition
    let trans =
      get_transition(transitions, num_states, num_colors, state, color);

    // Write new color
    tape[pos - 1] = trans.new_color;

    // Update state
    state = trans.new_state;

    // Move head (periodic boundary)
    cumulative_shift += trans.direction as i64;
    let new_pos = pos as i64 + trans.direction as i64;
    pos = if new_pos < 1 {
      tape_len
    } else if new_pos > tape_len as i64 {
      1
    } else {
      new_pos as usize
    };

    results.push((state, pos, cumulative_shift, tape.clone()));
  }

  results
}
