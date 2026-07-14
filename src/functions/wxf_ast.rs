//! BinarySerialize / BinaryDeserialize — the WXF binary serialization
//! format (header "8:"). Token bytes, varint lengths, and the choice of
//! integer width all match wolframscript's output byte for byte.

use crate::InterpreterError;
use crate::syntax::{BinaryOperator, ComparisonOp, Expr, UnaryOperator};

// WXF token bytes.
const T_FUNCTION: u8 = b'f'; // 102
const T_SYMBOL: u8 = b's'; // 115
const T_STRING: u8 = b'S'; // 83
const T_INT8: u8 = b'C'; // 67
const T_INT16: u8 = b'j'; // 106
const T_INT32: u8 = b'i'; // 105
const T_INT64: u8 = b'L'; // 76
const T_BIGINT: u8 = b'I'; // 73
const T_REAL64: u8 = b'r'; // 114
const T_BINARY: u8 = b'B'; // 66
const T_ASSOC: u8 = b'A'; // 65
const T_RULE: u8 = b'-'; // 45
const T_RULE_DELAYED: u8 = b':'; // 58
const T_PACKED: u8 = 0xC1; // 193

fn write_varint(out: &mut Vec<u8>, mut n: usize) {
  loop {
    let byte = (n & 0x7F) as u8;
    n >>= 7;
    if n > 0 {
      out.push(byte | 0x80);
    } else {
      out.push(byte);
      break;
    }
  }
}

fn read_varint(bytes: &[u8], pos: &mut usize) -> Option<usize> {
  let mut result: usize = 0;
  let mut shift = 0;
  loop {
    let byte = *bytes.get(*pos)?;
    *pos += 1;
    result |= ((byte & 0x7F) as usize) << shift;
    if byte & 0x80 == 0 {
      return Some(result);
    }
    shift += 7;
    if shift > 56 {
      return None;
    }
  }
}

/// The WXF name of a symbol: System symbols (anything in functions.csv)
/// are written bare, user symbols get the Global` context. Names that
/// already carry an explicit context are written as-is.
fn symbol_name(name: &str) -> String {
  if name.contains('`')
    || crate::evaluator::get_builtin_function_info(name).is_some()
  {
    name.to_string()
  } else {
    format!("Global`{}", name)
  }
}

fn write_symbol(out: &mut Vec<u8>, name: &str) {
  let full = symbol_name(name);
  out.push(T_SYMBOL);
  write_varint(out, full.len());
  out.extend_from_slice(full.as_bytes());
}

fn write_integer(out: &mut Vec<u8>, n: i128) {
  if let Ok(v) = i8::try_from(n) {
    out.push(T_INT8);
    out.push(v as u8);
  } else if let Ok(v) = i16::try_from(n) {
    out.push(T_INT16);
    out.extend_from_slice(&v.to_le_bytes());
  } else if let Ok(v) = i32::try_from(n) {
    out.push(T_INT32);
    out.extend_from_slice(&v.to_le_bytes());
  } else if let Ok(v) = i64::try_from(n) {
    out.push(T_INT64);
    out.extend_from_slice(&v.to_le_bytes());
  } else {
    write_big_integer(out, &n.to_string());
  }
}

fn write_big_integer(out: &mut Vec<u8>, digits: &str) {
  out.push(T_BIGINT);
  write_varint(out, digits.len());
  out.extend_from_slice(digits.as_bytes());
}

fn write_function_header(out: &mut Vec<u8>, head: &str, argc: usize) {
  out.push(T_FUNCTION);
  write_varint(out, argc);
  write_symbol(out, head);
}

/// A numeric complex number split into (re, im) literals, promoted to
/// Real when either part is machine precision — matching wolframscript's
/// Complex[0., 2.5] for 2.5*I. Returns None for purely real values and
/// for symbolic parts.
fn numeric_complex_parts(expr: &Expr) -> Option<(Expr, Expr)> {
  fn is_numeric_literal(e: &Expr) -> bool {
    match e {
      Expr::Integer(_) | Expr::BigInteger(_) | Expr::Real(_) => true,
      Expr::FunctionCall { name, args } if name == "Rational" => {
        args.len() == 2 && args.iter().all(is_numeric_literal)
      }
      Expr::UnaryOp {
        op: UnaryOperator::Minus,
        operand,
      } => is_numeric_literal(operand),
      _ => false,
    }
  }
  let (re, im) =
    crate::evaluator::dispatch::complex_and_special::split_real_imag_symbolic(
      expr,
    )?;
  if matches!(im, Expr::Integer(0)) || matches!(im, Expr::Real(z) if z == 0.0) {
    return None;
  }
  if !is_numeric_literal(&re) || !is_numeric_literal(&im) {
    return None;
  }
  // Machine precision is contagious across the pair.
  if matches!(re, Expr::Real(_)) || matches!(im, Expr::Real(_)) {
    let re_f = crate::functions::math_ast::expr_to_num(&re)?;
    let im_f = crate::functions::math_ast::expr_to_num(&im)?;
    return Some((Expr::Real(re_f), Expr::Real(im_f)));
  }
  Some((re, im))
}

/// A purely real numeric literal (Integer/BigInteger/Real/Rational, possibly
/// negated). Used to identify the numeric factors of a Times that fold into a
/// leading Complex atom alongside the complex factors.
fn is_real_numeric_literal(e: &Expr) -> bool {
  match e {
    Expr::Integer(_) | Expr::BigInteger(_) | Expr::Real(_) => true,
    Expr::FunctionCall { name, args } if name == "Rational" => {
      args.len() == 2 && args.iter().all(is_real_numeric_literal)
    }
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => is_real_numeric_literal(operand),
    _ => false,
  }
}

/// Serialize one expression. Returns None for expression kinds outside the
/// supported subset (the caller then leaves the whole call unevaluated
/// rather than emitting wrong bytes).
fn write_expr(out: &mut Vec<u8>, expr: &Expr) -> Option<()> {
  // Numeric complex values serialize as the atomic Complex[re, im] pair.
  if matches!(
    expr,
    Expr::BinaryOp { .. }
      | Expr::UnaryOp { .. }
      | Expr::FunctionCall { .. }
      | Expr::Identifier(_)
      | Expr::Constant(_)
  ) && let Some((re, im)) = numeric_complex_parts(expr)
  {
    write_function_header(out, "Complex", 2);
    write_expr(out, &re)?;
    write_expr(out, &im)?;
    return Some(());
  }
  match expr {
    Expr::Integer(n) => write_integer(out, *n),
    Expr::BigInteger(b) => {
      use num_traits::ToPrimitive;
      match b.to_i128() {
        Some(i) => write_integer(out, i),
        None => write_big_integer(out, &b.to_string()),
      }
    }
    Expr::Real(f) => {
      out.push(T_REAL64);
      out.extend_from_slice(&f.to_le_bytes());
    }
    Expr::String(s) => {
      out.push(T_STRING);
      write_varint(out, s.len());
      out.extend_from_slice(s.as_bytes());
    }
    Expr::Identifier(name) | Expr::Constant(name) => {
      // A symbolic expression containing a bare I would serialize with
      // Woxi's internal Plus/Times ordering of complex terms, which
      // diverges from wolframscript's canonical Complex folding — bail
      // out instead of emitting wrong bytes. (Pure numeric complexes are
      // already handled above.)
      if name == "I" {
        return None;
      }
      write_symbol(out, name)
    }
    Expr::List(items) => {
      write_function_header(out, "List", items.len());
      for item in items.iter() {
        write_expr(out, item)?;
      }
    }
    Expr::FunctionCall { name, args } => {
      // A ByteArray serializes as a WXF binary string.
      if name == "ByteArray"
        && args.len() == 1
        && let Some(bytes) =
          crate::functions::string_ast::byte_array_bytes(expr)
      {
        out.push(T_BINARY);
        write_varint(out, bytes.len());
        out.extend_from_slice(&bytes);
        return Some(());
      }
      // A product with a numeric-complex factor folds ALL its numeric
      // factors INTO a single leading Complex atom in wolframscript
      // (3*I*x is Times[Complex[0, 3], x]) whereas Woxi keeps them separate
      // internally (Times[3, Complex[0, 1], x]). Re-fold here so the bytes
      // match: multiply every numeric factor (reals and complexes alike)
      // into one Complex[re, im] atom and emit it ahead of the symbolic
      // factors, in their original order.
      if name == "Times"
        && args.iter().any(|a| numeric_complex_parts(a).is_some())
      {
        let (numeric, symbolic): (Vec<Expr>, Vec<Expr>) =
          args.iter().cloned().partition(|a| {
            is_real_numeric_literal(a) || numeric_complex_parts(a).is_some()
          });
        let folded = crate::functions::math_ast::times_ast(&numeric).ok()?;
        // If the imaginary parts cancelled (e.g. I*I), `folded` is real and
        // the ordinary path below serializes it correctly.
        if numeric_complex_parts(&folded).is_some() {
          if symbolic.is_empty() {
            return write_expr(out, &folded);
          }
          write_function_header(out, "Times", 1 + symbolic.len());
          write_expr(out, &folded)?;
          for arg in &symbolic {
            write_expr(out, arg)?;
          }
          return Some(());
        }
        // Fall through: reconstruct the fully-real product and serialize it.
        let mut rebuilt = vec![folded];
        rebuilt.extend(symbolic);
        write_function_header(out, "Times", rebuilt.len());
        for arg in &rebuilt {
          write_expr(out, arg)?;
        }
        return Some(());
      }
      write_function_header(out, name, args.len());
      // wolframscript's canonical Plus order sorts the numeric Complex atom
      // ahead of the symbolic terms (Plus[Complex[0, 3], x]); Woxi keeps it
      // last (Plus[x, Complex[0, 3]]). Emit the complex atom(s) first so the
      // bytes match. Real numeric terms already sort first in both engines.
      if name == "Plus"
        && args.iter().any(|a| numeric_complex_parts(a).is_some())
      {
        let (cplx, rest): (Vec<Expr>, Vec<Expr>) = args
          .iter()
          .cloned()
          .partition(|a| numeric_complex_parts(a).is_some());
        for arg in cplx.iter().chain(rest.iter()) {
          write_expr(out, arg)?;
        }
      } else {
        for arg in args.iter() {
          write_expr(out, arg)?;
        }
      }
    }
    Expr::CurriedCall { func, args } => {
      out.push(T_FUNCTION);
      write_varint(out, args.len());
      write_expr(out, func)?;
      for arg in args {
        write_expr(out, arg)?;
      }
    }
    Expr::Association(pairs) => {
      out.push(T_ASSOC);
      write_varint(out, pairs.len());
      for (k, v) in pairs {
        out.push(T_RULE);
        write_expr(out, k)?;
        write_expr(out, v)?;
      }
    }
    Expr::Rule {
      pattern,
      replacement,
    } => {
      write_function_header(out, "Rule", 2);
      write_expr(out, pattern)?;
      write_expr(out, replacement)?;
    }
    Expr::RuleDelayed {
      pattern,
      replacement,
    } => {
      write_function_header(out, "RuleDelayed", 2);
      write_expr(out, pattern)?;
      write_expr(out, replacement)?;
    }
    Expr::BinaryOp { op, left, right } => {
      // Rewrite operator forms to their FullForm heads.
      // See the FunctionCall "Plus" arm: complex terms in sums and
      // products serialize in a different shape in wolframscript.
      if matches!(
        op,
        BinaryOperator::Plus
          | BinaryOperator::Minus
          | BinaryOperator::Times
          | BinaryOperator::Divide
      ) && (numeric_complex_parts(left).is_some()
        || numeric_complex_parts(right).is_some())
      {
        return None;
      }
      match op {
        BinaryOperator::Plus => {
          write_function_header(out, "Plus", 2);
          write_expr(out, left)?;
          write_expr(out, right)?;
        }
        BinaryOperator::Minus => {
          // a - b  ==  Plus[a, Times[-1, b]]
          write_function_header(out, "Plus", 2);
          write_expr(out, left)?;
          write_function_header(out, "Times", 2);
          write_integer(out, -1);
          write_expr(out, right)?;
        }
        BinaryOperator::Times => {
          write_function_header(out, "Times", 2);
          write_expr(out, left)?;
          write_expr(out, right)?;
        }
        BinaryOperator::Divide => {
          // a / b  ==  Times[a, Power[b, -1]]
          write_function_header(out, "Times", 2);
          write_expr(out, left)?;
          write_function_header(out, "Power", 2);
          write_expr(out, right)?;
          write_integer(out, -1);
        }
        BinaryOperator::Power => {
          write_function_header(out, "Power", 2);
          write_expr(out, left)?;
          write_expr(out, right)?;
        }
        BinaryOperator::And => {
          write_function_header(out, "And", 2);
          write_expr(out, left)?;
          write_expr(out, right)?;
        }
        BinaryOperator::Or => {
          write_function_header(out, "Or", 2);
          write_expr(out, left)?;
          write_expr(out, right)?;
        }
        BinaryOperator::StringJoin => {
          write_function_header(out, "StringJoin", 2);
          write_expr(out, left)?;
          write_expr(out, right)?;
        }
        BinaryOperator::Alternatives => {
          write_function_header(out, "Alternatives", 2);
          write_expr(out, left)?;
          write_expr(out, right)?;
        }
      }
    }
    Expr::UnaryOp {
      op: UnaryOperator::Minus,
      operand,
    } => match operand.as_ref() {
      Expr::Integer(n) => write_integer(out, -n),
      Expr::Real(f) => {
        out.push(T_REAL64);
        out.extend_from_slice(&(-f).to_le_bytes());
      }
      _ => {
        // -x  ==  Times[-1, x]
        write_function_header(out, "Times", 2);
        write_integer(out, -1);
        write_expr(out, operand)?;
      }
    },
    Expr::Comparison {
      operands,
      operators,
    } if operands.len() == 2 && operators.len() == 1 => {
      let head = match operators[0] {
        ComparisonOp::Equal => "Equal",
        ComparisonOp::NotEqual => "Unequal",
        ComparisonOp::Less => "Less",
        ComparisonOp::LessEqual => "LessEqual",
        ComparisonOp::Greater => "Greater",
        ComparisonOp::GreaterEqual => "GreaterEqual",
        ComparisonOp::SameQ => "SameQ",
        ComparisonOp::UnsameQ => "UnsameQ",
      };
      write_function_header(out, head, 2);
      write_expr(out, &operands[0])?;
      write_expr(out, &operands[1])?;
    }
    _ => return None,
  }
  Some(())
}

/// BinarySerialize[expr] — WXF bytes as a ByteArray.
pub fn binary_serialize_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || {
    Ok(Expr::FunctionCall {
      name: "BinarySerialize".to_string(),
      args: args.to_vec().into(),
    })
  };
  if args.len() != 1 {
    return unevaluated();
  }
  let mut out: Vec<u8> = vec![b'8', b':'];
  if write_expr(&mut out, &args[0]).is_none() {
    return unevaluated();
  }
  use base64::Engine;
  let b64 = base64::engine::general_purpose::STANDARD.encode(&out);
  Ok(Expr::FunctionCall {
    name: "ByteArray".to_string(),
    args: vec![Expr::String(b64)].into(),
  })
}

fn read_exact<'a>(
  bytes: &'a [u8],
  pos: &mut usize,
  n: usize,
) -> Option<&'a [u8]> {
  let slice = bytes.get(*pos..*pos + n)?;
  *pos += n;
  Some(slice)
}

fn int_expr(n: i128) -> Expr {
  Expr::Integer(n)
}

fn read_expr(bytes: &[u8], pos: &mut usize) -> Option<Expr> {
  let token = *bytes.get(*pos)?;
  *pos += 1;
  match token {
    T_INT8 => {
      let b = read_exact(bytes, pos, 1)?;
      Some(int_expr(b[0] as i8 as i128))
    }
    T_INT16 => {
      let b = read_exact(bytes, pos, 2)?;
      Some(int_expr(i16::from_le_bytes([b[0], b[1]]) as i128))
    }
    T_INT32 => {
      let b = read_exact(bytes, pos, 4)?;
      Some(int_expr(
        i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as i128
      ))
    }
    T_INT64 => {
      let b = read_exact(bytes, pos, 8)?;
      let mut arr = [0u8; 8];
      arr.copy_from_slice(b);
      Some(int_expr(i64::from_le_bytes(arr) as i128))
    }
    T_BIGINT => {
      let len = read_varint(bytes, pos)?;
      let digits = std::str::from_utf8(read_exact(bytes, pos, len)?).ok()?;
      let big = num_bigint::BigInt::parse_bytes(digits.as_bytes(), 10)?;
      use num_traits::ToPrimitive;
      Some(match big.to_i128() {
        Some(i) => Expr::Integer(i),
        None => Expr::BigInteger(big),
      })
    }
    T_REAL64 => {
      let b = read_exact(bytes, pos, 8)?;
      let mut arr = [0u8; 8];
      arr.copy_from_slice(b);
      Some(Expr::Real(f64::from_le_bytes(arr)))
    }
    T_STRING => {
      let len = read_varint(bytes, pos)?;
      let s = std::str::from_utf8(read_exact(bytes, pos, len)?).ok()?;
      Some(Expr::String(s.to_string()))
    }
    T_SYMBOL => {
      let len = read_varint(bytes, pos)?;
      let full = std::str::from_utf8(read_exact(bytes, pos, len)?).ok()?;
      let name = full
        .strip_prefix("Global`")
        .or_else(|| full.strip_prefix("System`"))
        .unwrap_or(full);
      Some(Expr::Identifier(name.to_string()))
    }
    T_FUNCTION => {
      let argc = read_varint(bytes, pos)?;
      let head = read_expr(bytes, pos)?;
      let mut args = Vec::with_capacity(argc);
      for _ in 0..argc {
        args.push(read_expr(bytes, pos)?);
      }
      Some(match head {
        Expr::Identifier(ref name) if name == "List" => Expr::List(args.into()),
        Expr::Identifier(ref name) => Expr::FunctionCall {
          name: name.clone(),
          args: args.into(),
        },
        other => Expr::CurriedCall {
          func: Box::new(other),
          args,
        },
      })
    }
    T_ASSOC => {
      let count = read_varint(bytes, pos)?;
      let mut pairs = Vec::with_capacity(count);
      for _ in 0..count {
        let rule = *bytes.get(*pos)?;
        if rule != T_RULE && rule != T_RULE_DELAYED {
          return None;
        }
        *pos += 1;
        let k = read_expr(bytes, pos)?;
        let v = read_expr(bytes, pos)?;
        pairs.push((k, v));
      }
      Some(Expr::Association(pairs))
    }
    T_BINARY => {
      let len = read_varint(bytes, pos)?;
      let data = read_exact(bytes, pos, len)?;
      use base64::Engine;
      let b64 = base64::engine::general_purpose::STANDARD.encode(data);
      Some(Expr::FunctionCall {
        name: "ByteArray".to_string(),
        args: vec![Expr::String(b64)].into(),
      })
    }
    T_PACKED => {
      let dtype = *bytes.get(*pos)?;
      *pos += 1;
      let rank = read_varint(bytes, pos)?;
      if rank == 0 || rank > 16 {
        return None;
      }
      let mut dims = Vec::with_capacity(rank);
      for _ in 0..rank {
        dims.push(read_varint(bytes, pos)?);
      }
      read_packed_level(bytes, pos, dtype, &dims)
    }
    _ => None,
  }
}

/// One level of a packed array: recurse over the leading dimension.
fn read_packed_level(
  bytes: &[u8],
  pos: &mut usize,
  dtype: u8,
  dims: &[usize],
) -> Option<Expr> {
  let (first, rest) = dims.split_first()?;
  let mut items = Vec::with_capacity(*first);
  for _ in 0..*first {
    let item = if rest.is_empty() {
      match dtype {
        0 => int_expr(read_exact(bytes, pos, 1)?[0] as i8 as i128),
        1 => {
          let b = read_exact(bytes, pos, 2)?;
          int_expr(i16::from_le_bytes([b[0], b[1]]) as i128)
        }
        2 => {
          let b = read_exact(bytes, pos, 4)?;
          int_expr(i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as i128)
        }
        3 => {
          let b = read_exact(bytes, pos, 8)?;
          let mut arr = [0u8; 8];
          arr.copy_from_slice(b);
          int_expr(i64::from_le_bytes(arr) as i128)
        }
        0x23 => {
          let b = read_exact(bytes, pos, 8)?;
          let mut arr = [0u8; 8];
          arr.copy_from_slice(b);
          Expr::Real(f64::from_le_bytes(arr))
        }
        _ => return None,
      }
    } else {
      read_packed_level(bytes, pos, dtype, rest)?
    };
    items.push(item);
  }
  Some(Expr::List(items.into()))
}

/// BinaryDeserialize[ByteArray[…]] — parse WXF bytes back into an
/// expression; corrupt data emits BinaryDeserialize::corrupt and $Failed.
pub fn binary_deserialize_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let unevaluated = || {
    Ok(Expr::FunctionCall {
      name: "BinaryDeserialize".to_string(),
      args: args.to_vec().into(),
    })
  };
  if args.len() != 1 {
    return unevaluated();
  }
  let Some(bytes) = crate::functions::string_ast::byte_array_bytes(&args[0])
  else {
    return unevaluated();
  };
  let corrupt = || {
    crate::emit_message(&format!(
      "BinaryDeserialize::corrupt: Serialized data ByteArray[<{}>] is corrupt and does not represent an expression.",
      bytes.len()
    ));
    Ok(Expr::Identifier("$Failed".to_string()))
  };
  if bytes.len() < 2 || bytes[0] != b'8' || bytes[1] != b':' {
    return corrupt();
  }
  let mut pos = 2;
  let Some(expr) = read_expr(&bytes, &mut pos) else {
    return corrupt();
  };
  if pos != bytes.len() {
    return corrupt();
  }
  crate::evaluator::evaluate_expr_to_expr(&expr)
}
