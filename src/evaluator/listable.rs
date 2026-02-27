#[allow(unused_imports)]
use super::*;

/// Dispatch function call to built-in implementations (AST version).
/// This is the AST equivalent of the string-based function dispatch.
/// IMPORTANT: This function must NOT call interpret() to avoid infinite recursion.
/// Built-in Listable functions (thread automatically over list arguments)
pub fn is_builtin_listable(name: &str) -> bool {
  matches!(
    name,
    "Fibonacci"
      | "LucasL"
      | "Sin"
      | "Cos"
      | "Tan"
      | "Sec"
      | "Csc"
      | "Cot"
      | "Sinh"
      | "Cosh"
      | "Tanh"
      | "Coth"
      | "Sech"
      | "Csch"
      | "ArcSin"
      | "ArcCos"
      | "ArcTan"
      | "ArcSinh"
      | "ArcCosh"
      | "ArcTanh"
      | "Gudermannian"
      | "InverseGudermannian"
      | "Exp"
      | "Log"
      | "Log2"
      | "Log10"
      | "Abs"
      | "Sign"
      | "Floor"
      | "Ceiling"
      | "Round"
      | "Sqrt"
      | "Surd"
      | "Factorial"
      | "Gamma"
      | "Erf"
      | "Erfc"
      | "Prime"
      | "Power"
      | "Plus"
      | "Times"
      | "Mod"
      | "Quotient"
      | "GCD"
      | "LCM"
      | "Binomial"
      | "Multinomial"
      | "IntegerDigits"
      | "FactorInteger"
      | "IntegerLength"
      | "RealDigits"
      | "RomanNumeral"
      | "EulerPhi"
      | "MoebiusMu"
      | "DivisorSigma"
      | "BernoulliB"
      | "BellB"
      | "PrimePowerQ"
      | "CatalanNumber"
      | "StirlingS1"
      | "StirlingS2"
      | "HarmonicNumber"
      | "CoefficientList"
      | "ContinuedFraction"
      | "Boole"
      | "BitLength"
      | "EvenQ"
      | "OddQ"
      | "PrimeQ"
      | "Positive"
      | "Negative"
      | "NonPositive"
      | "NonNegative"
      | "StringLength"
      | "MixedFractionParts"
      | "Precision"
      | "Accuracy"
  )
}

pub fn is_builtin_flat(name: &str) -> bool {
  matches!(name, "Plus" | "Times" | "Max" | "Min" | "And" | "Or" | "Alternatives")
}

pub fn is_builtin_orderless(name: &str) -> bool {
  matches!(name, "Plus" | "Times" | "Max" | "Min" | "GCD" | "LCM")
}

/// Thread a Listable function over list arguments.
/// Returns Some(result) if threading was applied, None otherwise.
pub fn thread_listable(
  name: &str,
  args: &[Expr],
) -> Result<Option<Expr>, InterpreterError> {
  // Check if any argument is a list
  let has_list = args.iter().any(|a| matches!(a, Expr::List(_)));
  if !has_list {
    return Ok(None);
  }

  // Find the list length (all lists must have the same length)
  let mut list_len = None;
  for arg in args {
    if let Expr::List(items) = arg {
      match list_len {
        None => list_len = Some(items.len()),
        Some(n) if n != items.len() => {
          // Mismatched list lengths — don't thread, let the function handle it
          return Ok(None);
        }
        _ => {}
      }
    }
  }

  let len = match list_len {
    Some(n) => n,
    None => return Ok(None),
  };

  // Thread element-wise
  let mut results = Vec::with_capacity(len);
  for i in 0..len {
    let threaded_args: Vec<Expr> = args
      .iter()
      .map(|arg| {
        if let Expr::List(items) = arg {
          items[i].clone()
        } else {
          arg.clone()
        }
      })
      .collect();
    results.push(evaluate_function_call_ast(name, &threaded_args)?);
  }
  Ok(Some(Expr::List(results)))
}

/// Flatten Sequence[...] arguments into the parent function's argument list.
/// In Wolfram Language, Sequence[a, b] appearing as an argument to f produces f[..., a, b, ...].
/// Functions with the SequenceHold attribute suppress this.
/// Look up system $ variables
pub fn get_system_variable(name: &str) -> Option<Expr> {
  match name {
    "$RecursionLimit" => Some(Expr::Integer(256)),
    "$IterationLimit" => Some(Expr::Integer(4096)),
    "$MachinePrecision" => Some(Expr::Real(15.954589770191003)),
    "$MachineEpsilon" => Some(Expr::Real(2.220446049250313e-16)),
    "$MaxMachineNumber" => Some(Expr::Real(1.7976931348623157e308)),
    "$MinMachineNumber" => Some(Expr::Real(5e-324)),
    "$Assumptions" => Some(Expr::Identifier("True".to_string())),
    "$Context" => Some(Expr::String("Global`".to_string())),
    "$ContextPath" => Some(Expr::List(vec![
      Expr::String("System`".to_string()),
      Expr::String("Global`".to_string()),
    ])),
    _ => None,
  }
}

pub fn flatten_sequences(name: &str, args: &[Expr]) -> Vec<Expr> {
  // Check for SequenceHold attribute
  let has_sequence_hold = matches!(
    name,
    "Set"
      | "SetDelayed"
      | "Rule"
      | "RuleDelayed"
      | "HoldComplete"
      | "MakeBoxes"
  ) || crate::FUNC_ATTRS.with(|m| {
    m.borrow()
      .get(name)
      .is_some_and(|attrs| attrs.contains(&"SequenceHold".to_string()))
  });

  if has_sequence_hold {
    return args.to_vec();
  }

  let mut result = Vec::with_capacity(args.len());
  let mut had_sequence = false;
  for arg in args {
    if let Expr::FunctionCall {
      name: seq_name,
      args: seq_args,
    } = arg
      && seq_name == "Sequence"
    {
      result.extend(seq_args.iter().cloned());
      had_sequence = true;
      continue;
    }
    result.push(arg.clone());
  }
  if had_sequence { result } else { args.to_vec() }
}
