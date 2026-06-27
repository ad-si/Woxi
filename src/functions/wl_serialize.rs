//! Reader for Wolfram Language's binary serialization format, the byte stream
//! produced by `Compress` / consumed by `Uncompress` (and embedded in
//! `CompressedData[...]`). After the `"1:"` prefix is stripped, base64-decoded
//! and zlib-inflated, the payload begins with the magic `!boR` followed by a
//! tree of tagged tokens. This module turns that tree back into an [`Expr`].
//!
//! Ground-truth tokens (single leading byte), verified against `wolframscript`:
//!
//! | tag | meaning                                                            |
//! |-----|--------------------------------------------------------------------|
//! | `i` | machine integer: `<i32 LE>`                                         |
//! | `I` | big integer: `<i32 len><len ASCII decimal digits>`                  |
//! | `r` | machine real: `<f64 LE>`                                            |
//! | `S` | string: `<i32 len><len UTF-8 bytes>`                                |
//! | `s` | symbol (used as a head): `<i32 len><len UTF-8 bytes>`               |
//! | `f` | normal expression: `<i32 nargs><head expr><arg expr>*nargs`         |
//! | `n` | integer raw array: `<i32 type><i32 rank><i32 dims><packed ints>`    |
//! | `e` | real packed array: `<i32 rank><i32 dims><packed f64>`               |

use crate::syntax::Expr;

/// Decode a WL binary serialization payload (must start with `!boR`).
/// Returns `None` when the data is not in this format or is malformed.
pub fn deserialize(data: &[u8]) -> Option<Expr> {
  if data.len() < 5 || &data[0..4] != b"!boR" {
    return None;
  }
  let mut pos = 4;
  let expr = read_expr(data, &mut pos)?;
  Some(expr)
}

fn read_i32(data: &[u8], pos: &mut usize) -> Option<i32> {
  let end = pos.checked_add(4)?;
  if end > data.len() {
    return None;
  }
  let v = i32::from_le_bytes([
    data[*pos],
    data[*pos + 1],
    data[*pos + 2],
    data[*pos + 3],
  ]);
  *pos = end;
  Some(v)
}

fn read_f64(data: &[u8], pos: &mut usize) -> Option<f64> {
  let end = pos.checked_add(8)?;
  if end > data.len() {
    return None;
  }
  let mut b = [0u8; 8];
  b.copy_from_slice(&data[*pos..end]);
  *pos = end;
  Some(f64::from_le_bytes(b))
}

fn read_len_bytes<'a>(data: &'a [u8], pos: &mut usize) -> Option<&'a [u8]> {
  let len = read_i32(data, pos)?;
  if len < 0 {
    return None;
  }
  let len = len as usize;
  let end = pos.checked_add(len)?;
  if end > data.len() {
    return None;
  }
  let slice = &data[*pos..end];
  *pos = end;
  Some(slice)
}

fn read_expr(data: &[u8], pos: &mut usize) -> Option<Expr> {
  let tag = *data.get(*pos)?;
  *pos += 1;
  match tag {
    b'i' => Some(Expr::Integer(read_i32(data, pos)? as i128)),
    b'r' => Some(Expr::Real(read_f64(data, pos)?)),
    b'I' => {
      let s = std::str::from_utf8(read_len_bytes(data, pos)?).ok()?;
      Some(parse_integer(s))
    }
    b'S' => {
      let s = std::str::from_utf8(read_len_bytes(data, pos)?).ok()?;
      Some(Expr::String(s.to_string()))
    }
    b's' => {
      let s = std::str::from_utf8(read_len_bytes(data, pos)?).ok()?;
      Some(Expr::Identifier(s.to_string()))
    }
    b'f' => {
      let nargs = read_i32(data, pos)?;
      if nargs < 0 {
        return None;
      }
      let head = read_expr(data, pos)?;
      let mut args = Vec::with_capacity(nargs as usize);
      for _ in 0..nargs {
        args.push(read_expr(data, pos)?);
      }
      Some(build_normal(head, args))
    }
    b'n' => read_integer_array(data, pos),
    b'e' => read_real_array(data, pos),
    _ => None,
  }
}

/// `f`-token reconstruction: a `List` head becomes [`Expr::List`], any other
/// symbol head becomes a [`Expr::FunctionCall`]. A non-symbol head is wrapped
/// as `head[args...]` via its rendered name so nothing is silently dropped.
fn build_normal(head: Expr, args: Vec<Expr>) -> Expr {
  match &head {
    Expr::Identifier(name) if name == "List" => Expr::List(args.into()),
    Expr::Identifier(name) => Expr::FunctionCall {
      name: name.clone(),
      args: args.into(),
    },
    other => Expr::FunctionCall {
      name: crate::syntax::expr_to_string(other),
      args: args.into(),
    },
  }
}

fn parse_integer(s: &str) -> Expr {
  if let Ok(i) = s.parse::<i128>() {
    Expr::Integer(i)
  } else if let Ok(big) = s.parse::<num_bigint::BigInt>() {
    Expr::BigInteger(big)
  } else {
    Expr::Integer(0)
  }
}

/// Build a (possibly nested) list of `count` leaves over the given dimensions.
fn nest(dims: &[usize], leaves: &mut std::vec::IntoIter<Expr>) -> Expr {
  match dims.split_first() {
    None => leaves.next().unwrap_or(Expr::Integer(0)),
    Some((&first, rest)) => Expr::List(
      (0..first)
        .map(|_| nest(rest, leaves))
        .collect::<Vec<_>>()
        .into(),
    ),
  }
}

fn read_dims(data: &[u8], pos: &mut usize, rank: i32) -> Option<Vec<usize>> {
  if rank < 0 {
    return None;
  }
  let mut dims = Vec::with_capacity(rank as usize);
  for _ in 0..rank {
    let d = read_i32(data, pos)?;
    if d < 0 {
      return None;
    }
    dims.push(d as usize);
  }
  Some(dims)
}

/// `n` token — raw integer array. The element width is taken from the trailing
/// payload (signed little-endian ints of 1/2/4/8 bytes), which makes the reader
/// independent of the exact element-type code.
fn read_integer_array(data: &[u8], pos: &mut usize) -> Option<Expr> {
  let _typ = read_i32(data, pos)?;
  let rank = read_i32(data, pos)?;
  let dims = read_dims(data, pos, rank)?;
  let count: usize = dims.iter().product();
  let values = if count == 0 {
    Vec::new()
  } else {
    let rest = data.len().checked_sub(*pos)?;
    if rest % count != 0 {
      return None;
    }
    let width = rest / count;
    if !matches!(width, 1 | 2 | 4 | 8) {
      return None;
    }
    let mut vals = Vec::with_capacity(count);
    for _ in 0..count {
      let mut buf = [0u8; 8];
      buf[..width].copy_from_slice(&data[*pos..*pos + width]);
      // Sign-extend a little-endian signed integer of `width` bytes.
      let raw = u64::from_le_bytes(buf);
      let shift = 64 - width * 8;
      let signed = ((raw << shift) as i64) >> shift;
      vals.push(Expr::Integer(signed as i128));
      *pos += width;
    }
    vals
  };
  Some(nest(&dims, &mut values.into_iter()))
}

/// `e` token — packed real (f64) array.
fn read_real_array(data: &[u8], pos: &mut usize) -> Option<Expr> {
  let rank = read_i32(data, pos)?;
  let dims = read_dims(data, pos, rank)?;
  let count: usize = dims.iter().product();
  let mut vals = Vec::with_capacity(count);
  for _ in 0..count {
    vals.push(Expr::Real(read_f64(data, pos)?));
  }
  Some(nest(&dims, &mut vals.into_iter()))
}

#[cfg(test)]
mod tests {
  use super::*;

  fn render(data: &[u8]) -> String {
    crate::syntax::expr_to_string(&deserialize(data).unwrap())
  }

  #[test]
  fn reads_machine_integer_list() {
    // Compress[{1, 2, 3}] payload
    let data = b"!boRf\x03\x00\x00\x00s\x04\x00\x00\x00Listi\x01\x00\x00\x00i\x02\x00\x00\x00i\x03\x00\x00\x00";
    assert_eq!(render(data), "{1, 2, 3}");
  }

  #[test]
  fn reads_string_token() {
    assert_eq!(render(b"!boRS\x02\x00\x00\x00hi"), "\"hi\"");
  }

  #[test]
  fn reads_big_integer_token() {
    // I token: length-prefixed ASCII decimal
    assert_eq!(render(b"!boRI\x02\x00\x00\x0010"), "10");
  }

  #[test]
  fn reads_machine_real() {
    // r token: f64 little-endian 1.5
    assert_eq!(render(b"!boRr\x00\x00\x00\x00\x00\x00\xf8?"), "1.5");
  }

  #[test]
  fn rejects_non_magic() {
    assert!(deserialize(b"nope").is_none());
  }
}
