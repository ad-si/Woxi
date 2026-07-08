//! Reader for CERN ROOT files (<https://root.cern/>).
//!
//! `Import["file.root"]` walks the file's directory structure and returns an
//! Association mapping each stored object's name to a decoded value:
//!
//! - `TObjString` → the contained String
//! - `TH1C` / `TH1S` / `TH1I` / `TH1F` / `TH1D` → an Association with the
//!   axis definition, entry count, and bin contents
//! - `TH2C` / `TH2S` / `TH2I` / `TH2F` / `TH2D` → an Association with both
//!   axis definitions, entry count, and the 2-D bin-content matrix
//! - `TTree` → an Association with the entry count and the branch
//!   name → type (leaf specification or element class) mapping
//! - `TDirectory` → a nested Association of the directory's contents
//! - any other class → an Association with `"ClassName"` and `"Title"` so
//!   the object is at least visible
//!
//! Listing a whole file stays cheap: a histogram whose bin count exceeds the
//! walk's cell budget ([`MAX_LISTING_CELLS`]) keeps its metadata (axes, entry
//! count) but omits the bin-content array, which is still read in full through
//! its element path (`Import[file, {"ROOT", "name"}]`). Without this bound a
//! detector file's large histograms turned a 14 MB file into gigabytes of
//! decoded values and aborted the WASM heap.
//!
//! Branch *data* is available through element paths, so a multi-GB tree
//! never has to be materialized just to list a file's contents:
//!
//! - `Import[file, {"ROOT", "dir/tree", "branch"}]` → one branch's values
//! - `Import[file, {"ROOT", "dir/tree", {"b1", "b2"}}]` → an Association
//!   of the selected columns
//! - `Import[file, {"ROOT", "dir/tree", "Data"}]` → an Association of
//!   every branch's values (may be very large — prefer selecting branches)
//!
//! Decoded branch layouts: flat basic types (`TLeafB/S/I/L/F/D/O`, signed
//! and unsigned, fixed-size arrays), variable-length leaf-count arrays,
//! `std::vector<basic type>`, and `TLorentzVector` (returned as
//! `<|"Px" -> …, "Py" -> …, "Pz" -> …, "E" -> …|>`).
//!
//! Wolfram Language itself has no ROOT importer, so this is a Woxi
//! extension; the output shape follows Import's Association-producing
//! formats (e.g. JSON) rather than any wolframscript reference output.
//!
//! Supported compression codecs: uncompressed, zlib (`ZL`), and LZ4 (`L4`).
//! LZMA (`XZ`) and ZSTD (`ZS`) records produce a descriptive error.

use crate::InterpreterError;
use crate::syntax::Expr;

/// Flag bit marking a 4-byte length prefix ("byte count") in streamed data.
const K_BYTE_COUNT_MASK: u32 = 0x4000_0000;
/// Low 30 bits of a length-prefix word: the byte count itself.
const K_BYTE_COUNT_VALUE: u32 = 0x3FFF_FFFF;
/// Tag announcing that a class name string follows in an object stream.
const K_NEW_CLASS_TAG: u32 = 0xFFFF_FFFF;
/// Flag bit on an object tag marking a class reference (vs. an object ref).
const K_CLASS_MASK: u32 = 0x8000_0000;
/// Streamed class references point 2 bytes past the tag word they cite.
const K_MAP_OFFSET: u32 = 2;
/// `TObject::fBits` flag: a 2-byte process id follows the bits field.
const K_IS_REFERENCED: u32 = 1 << 4;
/// Directories may nest; guard against reference cycles in corrupt files.
const MAX_DIR_DEPTH: usize = 16;
/// Upper bound for `Vec::with_capacity` calls driven by on-file counts, so
/// a corrupt length field fails with a read error instead of a huge
/// allocation. Vectors still grow past this if the data really is there.
const MAX_PREALLOC: usize = 65_536;
/// Cumulative cap on the number of histogram cells a whole-directory walk
/// (`Import[file]`) materializes. Detector files carry histograms with tens of
/// millions of bins; fully decoding every one turned a 14 MB file into >7 GB
/// of `Expr` values and aborted the (32-bit) WASM heap. Beyond this many
/// cells, further histograms degrade to their metadata Association (see
/// [`decode_object`]). The cap is also kept low enough that the listing stays
/// small: an assigned value is stored and re-parsed through its string form on
/// every access (`StoredValue::Raw`), so a multi-million-cell Association makes
/// even `data = Import[file]; Head[data]` crawl (the stored string is
/// re-parsed on every access by an O(n²) parser). Only genuinely small
/// histograms keep their bin contents inline in a whole-file listing; larger
/// ones show up as metadata and are read in full through their element path.
const MAX_LISTING_CELLS: usize = 4_096;
/// Cell budget for an explicit single-object element path
/// (`Import[file, {"ROOT", "hist"}]`). Much larger than the whole-file listing
/// cap because the caller asked for that one object by name, but still finite
/// so a single pathological histogram can't exhaust the WASM heap.
const MAX_OBJECT_CELLS: usize = 10_000_000;

#[cfg(not(target_arch = "wasm32"))]
pub fn root_import_file(path: &str) -> Result<Expr, InterpreterError> {
  let bytes = std::fs::read(path).map_err(|e| {
    InterpreterError::EvaluationError(format!(
      "Import: cannot open \"{}\": {}",
      path, e
    ))
  })?;
  root_import_bytes(&bytes)
}

pub fn root_import_bytes(bytes: &[u8]) -> Result<Expr, InterpreterError> {
  parse_root(bytes)
    .map_err(|e| InterpreterError::EvaluationError(format!("Import: {}", e)))
}

#[cfg(not(target_arch = "wasm32"))]
pub fn root_import_file_element(
  path: &str,
  elements: &[Expr],
) -> Result<Expr, InterpreterError> {
  let bytes = std::fs::read(path).map_err(|e| {
    InterpreterError::EvaluationError(format!(
      "Import: cannot open \"{}\": {}",
      path, e
    ))
  })?;
  root_import_bytes_element(&bytes, elements)
}

/// Element access below the file level, mirroring Import's element-path
/// convention: `{"ROOT", "dir/obj"}` returns one decoded object,
/// `{"ROOT", "dir/tree", "branch"}` one column, `{"ROOT", "dir/tree",
/// {"b1", "b2"}}` an Association of the selected columns, and
/// `{"ROOT", "dir/tree", "Data"}` every column.
pub fn root_import_bytes_element(
  bytes: &[u8],
  elements: &[Expr],
) -> Result<Expr, InterpreterError> {
  root_element(bytes, elements)
    .map_err(|e| InterpreterError::EvaluationError(format!("Import: {}", e)))
}

fn root_element(data: &[u8], elements: &[Expr]) -> Result<Expr, String> {
  let [path_expr, rest @ ..] = elements else {
    return parse_root(data);
  };
  let Expr::String(path) = path_expr else {
    return Err("the object path element must be a string".into());
  };
  let mut seek_keys = top_dir_seek_keys(data)?;
  let components: Vec<&str> = path.split('/').collect();
  for (i, component) in components.iter().enumerate() {
    // An explicit cycle (`name;2`) selects one revision of the object;
    // otherwise the highest cycle wins, matching the directory walk.
    let (name, cycle) = match component.rsplit_once(';') {
      Some((n, c))
        if !c.is_empty() && c.bytes().all(|b| b.is_ascii_digit()) =>
      {
        (n, c.parse::<u16>().ok())
      }
      _ => (*component, None),
    };
    let key = read_dir_keys(data, seek_keys)?
      .into_iter()
      .filter(|k| k.name == name && cycle.is_none_or(|c| k.cycle == c))
      .max_by_key(|k| k.cycle)
      .ok_or_else(|| format!("object \"{}\" not found in file", component))?;
    let is_last = i + 1 == components.len();
    if key.class_name == "TDirectory" || key.class_name == "TDirectoryFile" {
      let sub =
        dir_seek_keys(data, key.seek_key as usize + key.key_len as usize)?;
      if is_last {
        if !rest.is_empty() {
          return Err(format!(
            "\"{}\" is a directory and has no branch elements",
            path
          ));
        }
        let mut budget = MAX_LISTING_CELLS;
        return read_directory(data, sub, 0, &mut budget);
      }
      seek_keys = sub;
    } else if !is_last {
      return Err(format!("\"{}\" is not a directory", component));
    } else {
      let payload = object_payload(data, &key)?;
      return decode_element(data, &key, &payload, rest);
    }
  }
  Err("empty object path".into())
}

/// Decode the object a path resolved to, honoring an optional branch
/// selector (only meaningful for trees).
fn decode_element(
  data: &[u8],
  key: &KeyInfo,
  payload: &[u8],
  rest: &[Expr],
) -> Result<Expr, String> {
  if key.class_name != "TTree" {
    if !rest.is_empty() {
      return Err(format!("class {} has no sub-elements", key.class_name));
    }
    // Explicit single-object access gets its own, larger budget so a large
    // histogram requested by name is still decoded in full.
    let mut budget = MAX_OBJECT_CELLS;
    return Ok(decode_object(key, payload, &mut budget));
  }
  let selector = match rest {
    [] => return decode_ttree(key, payload),
    [sel] => sel,
    _ => return Err("too many elements for a TTree".into()),
  };
  let (tree, branch_err) = parse_ttree(payload, key.key_len as u32)?;
  if let Some(e) = branch_err {
    return Err(e);
  }
  let find = |name: &str| -> Result<&BranchInfo, String> {
    tree
      .branches
      .iter()
      .find(|b| b.name == name)
      .ok_or_else(|| format!("branch \"{}\" not found in tree", name))
  };
  match selector {
    // A single branch name yields its bare column.
    Expr::String(name) if name != "Data" => {
      Ok(Expr::List(branch_values(data, find(name)?)?.into()))
    }
    // "Data" yields every branch, degrading per branch like the tree walk.
    Expr::String(_) => {
      let mut pairs: Vec<(Expr, Expr)> = Vec::new();
      for branch in &tree.branches {
        let value = match branch_values(data, branch) {
          Ok(values) => Expr::List(values.into()),
          Err(e) => Expr::Association(vec![(
            Expr::String("Error".into()),
            Expr::String(e),
          )]),
        };
        pairs.push((Expr::String(branch.name.clone()), value));
      }
      Ok(Expr::Association(pairs))
    }
    // A list of branch names yields an Association of those columns.
    Expr::List(names) => {
      let mut pairs: Vec<(Expr, Expr)> = Vec::new();
      for item in names.iter() {
        let Expr::String(name) = item else {
          return Err("branch names must be strings".into());
        };
        let values = branch_values(data, find(name)?)?;
        pairs.push((Expr::String(name.clone()), Expr::List(values.into())));
      }
      Ok(Expr::Association(pairs))
    }
    _ => Err("the branch selector must be a string or list of strings".into()),
  }
}

/// Big-endian reader over the raw file bytes with bounds-checked accessors.
/// ROOT streams all multi-byte scalars big-endian regardless of platform.
struct Reader<'a> {
  data: &'a [u8],
  pos: usize,
}

impl<'a> Reader<'a> {
  fn new(data: &'a [u8]) -> Self {
    Reader { data, pos: 0 }
  }

  fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
    let end = self
      .pos
      .checked_add(n)
      .filter(|e| *e <= self.data.len())
      .ok_or("unexpected end of file")?;
    let slice = &self.data[self.pos..end];
    self.pos = end;
    Ok(slice)
  }

  fn seek(&mut self, pos: usize) -> Result<(), String> {
    if pos > self.data.len() {
      return Err("seek past end of file".into());
    }
    self.pos = pos;
    Ok(())
  }

  fn read_u8(&mut self) -> Result<u8, String> {
    Ok(self.take(1)?[0])
  }

  fn read_u16(&mut self) -> Result<u16, String> {
    Ok(u16::from_be_bytes(self.take(2)?.try_into().unwrap()))
  }

  fn read_u32(&mut self) -> Result<u32, String> {
    Ok(u32::from_be_bytes(self.take(4)?.try_into().unwrap()))
  }

  fn read_i32(&mut self) -> Result<i32, String> {
    Ok(i32::from_be_bytes(self.take(4)?.try_into().unwrap()))
  }

  fn read_u64(&mut self) -> Result<u64, String> {
    Ok(u64::from_be_bytes(self.take(8)?.try_into().unwrap()))
  }

  fn read_i64(&mut self) -> Result<i64, String> {
    Ok(i64::from_be_bytes(self.take(8)?.try_into().unwrap()))
  }

  fn read_i16(&mut self) -> Result<i16, String> {
    Ok(i16::from_be_bytes(self.take(2)?.try_into().unwrap()))
  }

  fn read_f32(&mut self) -> Result<f32, String> {
    Ok(f32::from_be_bytes(self.take(4)?.try_into().unwrap()))
  }

  fn read_f64(&mut self) -> Result<f64, String> {
    Ok(f64::from_be_bytes(self.take(8)?.try_into().unwrap()))
  }

  /// TString: a 1-byte length, or 255 followed by a 4-byte length.
  fn read_tstring(&mut self) -> Result<String, String> {
    let short_len = self.read_u8()?;
    let len = if short_len == 255 {
      self.read_u32()? as usize
    } else {
      short_len as usize
    };
    Ok(String::from_utf8_lossy(self.take(len)?).into_owned())
  }

  /// NUL-terminated class name emitted after a "new class" tag.
  fn read_cstring(&mut self) -> Result<String, String> {
    let start = self.pos;
    while self.read_u8()? != 0 {}
    Ok(String::from_utf8_lossy(&self.data[start..self.pos - 1]).into_owned())
  }

  /// Streamed-class version header: an optional byte count (4 bytes with
  /// `K_BYTE_COUNT_MASK` set, counting everything that follows it) and a
  /// 2-byte class version. Returns the version and, when a byte count was
  /// present, the absolute position just past this object segment.
  fn read_version(&mut self) -> Result<(u16, Option<usize>), String> {
    let end = self
      .pos
      .checked_add(4)
      .filter(|e| *e <= self.data.len())
      .ok_or("unexpected end of file")?;
    let raw = u32::from_be_bytes(self.data[self.pos..end].try_into().unwrap());
    if raw & K_BYTE_COUNT_MASK != 0 {
      let seg_end = self.pos + 4 + (raw & K_BYTE_COUNT_VALUE) as usize;
      self.pos = end;
      let version = self.read_u16()?;
      Ok((version, Some(seg_end)))
    } else {
      Ok((self.read_u16()?, None))
    }
  }

  /// Skip a byte-count-delimited class segment (attribute classes, unused
  /// axes, …) without interpreting its contents.
  fn skip_versioned(&mut self) -> Result<(), String> {
    let (_, end) = self.read_version()?;
    self.seek(end.ok_or("missing byte count in object stream")?)
  }

  /// TObject base data: version, unique id, bits (+ process id when the
  /// object is referenced).
  fn skip_tobject(&mut self) -> Result<(), String> {
    let _version = self.read_u16()?;
    let _unique_id = self.read_u32()?;
    let bits = self.read_u32()?;
    if bits & K_IS_REFERENCED != 0 {
      let _pidf = self.read_u16()?;
    }
    Ok(())
  }

  /// TNamed base data: TObject followed by the name and title strings.
  fn read_tnamed(&mut self) -> Result<(String, String), String> {
    let (_, end) = self.read_version()?;
    self.skip_tobject()?;
    let name = self.read_tstring()?;
    let title = self.read_tstring()?;
    if let Some(e) = end {
      self.seek(e)?;
    }
    Ok((name, title))
  }
}

/// The directory entry ("key") preceding every stored object.
struct KeyInfo {
  n_bytes: u32,
  obj_len: u32,
  key_len: u16,
  cycle: u16,
  seek_key: u64,
  class_name: String,
  name: String,
  title: String,
}

fn read_key(r: &mut Reader) -> Result<KeyInfo, String> {
  let n_bytes = r.read_i32()?;
  if n_bytes <= 0 {
    // A negative length marks a deleted (gap) record in the key list.
    return Err("empty key slot".into());
  }
  let version = r.read_u16()?;
  let obj_len = r.read_u32()?;
  let _datime = r.read_u32()?;
  let key_len = r.read_u16()?;
  let cycle = r.read_u16()?;
  let (seek_key, _seek_pdir) = if version > 1000 {
    (r.read_u64()?, r.read_u64()?)
  } else {
    (r.read_u32()? as u64, r.read_u32()? as u64)
  };
  let class_name = r.read_tstring()?;
  let name = r.read_tstring()?;
  let title = r.read_tstring()?;
  Ok(KeyInfo {
    n_bytes: n_bytes as u32,
    obj_len,
    key_len,
    cycle,
    seek_key,
    class_name,
    name,
    title,
  })
}

fn parse_root(data: &[u8]) -> Result<Expr, String> {
  let mut budget = MAX_LISTING_CELLS;
  read_directory(data, top_dir_seek_keys(data)?, 0, &mut budget)
}

/// Validate the file header and return the top directory's key-list offset.
fn top_dir_seek_keys(data: &[u8]) -> Result<u64, String> {
  if !data.starts_with(b"root") {
    return Err("not a ROOT file (missing \"root\" magic)".into());
  }
  let mut r = Reader::new(data);
  r.seek(4)?;
  let version = r.read_u32()?;
  let begin = r.read_u32()? as u64;
  // Files larger than 2 GB store 8-byte seek pointers and add 1,000,000
  // to the version number.
  let large = version >= 1_000_000;
  if large {
    let _end = r.read_u64()?;
    let _seek_free = r.read_u64()?;
  } else {
    let _end = r.read_u32()?;
    let _seek_free = r.read_u32()?;
  }
  let _nbytes_free = r.read_u32()?;
  let _nfree = r.read_u32()?;
  let nbytes_name = r.read_u32()?;
  // The top-level TDirectory data sits right after the file's name record.
  dir_seek_keys(data, begin as usize + nbytes_name as usize)
}

/// Parse streamed TDirectory data at `pos` and return its key-list offset.
fn dir_seek_keys(data: &[u8], pos: usize) -> Result<u64, String> {
  let mut r = Reader::new(data);
  r.seek(pos)?;
  let version = r.read_u16()?;
  let _datime_c = r.read_u32()?;
  let _datime_m = r.read_u32()?;
  let _nbytes_keys = r.read_i32()?;
  let _nbytes_name = r.read_i32()?;
  let seek_keys = if version > 1000 {
    let _seek_dir = r.read_u64()?;
    let _seek_parent = r.read_u64()?;
    r.read_u64()?
  } else {
    let _seek_dir = r.read_u32()?;
    let _seek_parent = r.read_u32()?;
    r.read_u32()? as u64
  };
  Ok(seek_keys)
}

/// Read the raw key list stored at `seek_keys` (all cycles included).
fn read_dir_keys(data: &[u8], seek_keys: u64) -> Result<Vec<KeyInfo>, String> {
  let mut r = Reader::new(data);
  r.seek(seek_keys as usize)?;
  // The key list is itself stored as a record with a key header.
  let header = read_key(&mut r)?;
  r.seek(seek_keys as usize + header.key_len as usize)?;
  let n_keys = r.read_i32()?;
  if n_keys < 0 {
    return Err("negative key count in directory".into());
  }
  // Cap the pre-allocation: a corrupt count fails on read, not on alloc.
  let mut keys: Vec<KeyInfo> =
    Vec::with_capacity((n_keys as usize).min(MAX_PREALLOC));
  for _ in 0..n_keys {
    keys.push(read_key(&mut r)?);
  }
  Ok(keys)
}

/// Read the key list at `seek_keys` and decode every object in the
/// directory into an Association, recursing into subdirectories.
fn read_directory(
  data: &[u8],
  seek_keys: u64,
  depth: usize,
  budget: &mut usize,
) -> Result<Expr, String> {
  if depth > MAX_DIR_DEPTH {
    return Err("directory nesting too deep".into());
  }
  let keys = read_dir_keys(data, seek_keys)?;
  // A file may hold several cycles of the same object (name;1, name;2, …).
  // Keep only the highest cycle of each name, in first-seen order.
  let mut selected: Vec<KeyInfo> = Vec::with_capacity(keys.len());
  for key in keys {
    match selected.iter_mut().find(|k| k.name == key.name) {
      Some(existing) => {
        if key.cycle > existing.cycle {
          *existing = key;
        }
      }
      None => selected.push(key),
    }
  }
  let mut pairs: Vec<(Expr, Expr)> = Vec::with_capacity(selected.len());
  for key in &selected {
    let value =
      if key.class_name == "TDirectory" || key.class_name == "TDirectoryFile" {
        let sub_seek =
          dir_seek_keys(data, key.seek_key as usize + key.key_len as usize)?;
        read_directory(data, sub_seek, depth + 1, budget)?
      } else {
        match object_payload(data, key) {
          Ok(payload) => decode_object(key, &payload, budget),
          Err(e) => {
            // Undecodable payloads (e.g. an unsupported compression codec for
            // this one record) degrade to metadata plus the reason.
            Expr::Association(vec![
              (
                Expr::String("ClassName".into()),
                Expr::String(key.class_name.clone()),
              ),
              (
                Expr::String("Title".into()),
                Expr::String(key.title.clone()),
              ),
              (Expr::String("Error".into()), Expr::String(e)),
            ])
          }
        }
      };
    pairs.push((Expr::String(key.name.clone()), value));
  }
  Ok(Expr::Association(pairs))
}

/// Extract (and decompress if needed) the streamed object bytes of a key.
fn object_payload(data: &[u8], key: &KeyInfo) -> Result<Vec<u8>, String> {
  let start = key.seek_key as usize + key.key_len as usize;
  let end = key.seek_key as usize + key.n_bytes as usize;
  let record = data.get(start..end).ok_or("object record out of bounds")?;
  if key.obj_len as usize == record.len() {
    return Ok(record.to_vec());
  }
  // Compressed records are a sequence of blocks, each with a 9-byte header:
  // 2-byte codec tag, 1-byte method, then compressed and uncompressed sizes
  // as 3-byte little-endian integers.
  let mut out: Vec<u8> =
    Vec::with_capacity((key.obj_len as usize).min(MAX_PREALLOC));
  let mut rest = record;
  while out.len() < key.obj_len as usize {
    if rest.len() < 9 {
      return Err("truncated compressed block".into());
    }
    let codec = &rest[0..2];
    let c_size =
      rest[3] as usize | (rest[4] as usize) << 8 | (rest[5] as usize) << 16;
    let u_size =
      rest[6] as usize | (rest[7] as usize) << 8 | (rest[8] as usize) << 16;
    let payload = rest
      .get(9..9 + c_size)
      .ok_or("truncated compressed block payload")?;
    match codec {
      b"ZL" => {
        use std::io::Read;
        let mut decoder = flate2::read::ZlibDecoder::new(payload);
        let before = out.len();
        decoder
          .by_ref()
          .take(u_size as u64)
          .read_to_end(&mut out)
          .map_err(|e| format!("zlib decompression failed: {}", e))?;
        if out.len() - before != u_size {
          return Err("zlib block decompressed to unexpected size".into());
        }
      }
      b"L4" => {
        // The compressed payload starts with an 8-byte xxhash64 checksum.
        let lz4_data = payload.get(8..).ok_or("truncated LZ4 block")?;
        let decoded = lz4_flex::block::decompress(lz4_data, u_size)
          .map_err(|e| format!("LZ4 decompression failed: {}", e))?;
        out.extend_from_slice(&decoded);
      }
      b"XZ" => {
        return Err("LZMA-compressed ROOT records are not supported".into());
      }
      b"ZS" => {
        return Err("ZSTD-compressed ROOT records are not supported".into());
      }
      other => {
        return Err(format!(
          "unsupported ROOT compression codec {:?}",
          String::from_utf8_lossy(other)
        ));
      }
    }
    rest = &rest[9 + c_size..];
  }
  Ok(out)
}

/// Decode one object's streamed bytes. Any parse failure degrades to the
/// metadata Association so a single odd object never breaks the import.
fn decode_object(key: &KeyInfo, payload: &[u8], budget: &mut usize) -> Expr {
  let decoded = match key.class_name.as_str() {
    "TObjString" => decode_tobjstring(payload),
    "TH1C" | "TH1S" | "TH1I" | "TH1F" | "TH1D" => {
      decode_th1(&key.class_name, payload, budget)
    }
    "TH2C" | "TH2S" | "TH2I" | "TH2F" | "TH2D" => {
      decode_th2(&key.class_name, payload, budget)
    }
    "TTree" => decode_ttree(key, payload),
    _ => Err("class not decoded".into()),
  };
  decoded.unwrap_or_else(|_| {
    Expr::Association(vec![
      (
        Expr::String("ClassName".into()),
        Expr::String(key.class_name.clone()),
      ),
      (
        Expr::String("Title".into()),
        Expr::String(key.title.clone()),
      ),
    ])
  })
}

fn decode_tobjstring(payload: &[u8]) -> Result<Expr, String> {
  let mut r = Reader::new(payload);
  let (_, _) = r.read_version()?;
  r.skip_tobject()?;
  Ok(Expr::String(r.read_tstring()?))
}

/// Bin definition of one histogram axis.
struct AxisInfo {
  nbins: i32,
  min: f64,
  max: f64,
  /// Explicit bin edges for variable-width binning; empty otherwise.
  edges: Vec<f64>,
}

/// TAxis: extract the bin definition, then skip the remaining attributes.
fn read_taxis(r: &mut Reader) -> Result<AxisInfo, String> {
  let (_, end) = r.read_version()?;
  let end = end.ok_or("missing byte count on TAxis")?;
  let _ = r.read_tnamed()?;
  r.skip_versioned()?; // TAttAxis
  let nbins = r.read_i32()?;
  let min = r.read_f64()?;
  let max = r.read_f64()?;
  // fXbins: a TArrayD of bin edges; empty for fixed-width binning.
  let n_edges = r.read_i32()?;
  let mut edges =
    Vec::with_capacity((n_edges.max(0) as usize).min(MAX_PREALLOC));
  for _ in 0..n_edges.max(0) {
    edges.push(r.read_f64()?);
  }
  r.seek(end)?;
  Ok(AxisInfo {
    nbins,
    min,
    max,
    edges,
  })
}

/// Parse the shared TH1 base-class segment: title, x/y axis definitions,
/// and entry count. Leaves the reader just past the segment, at the start
/// of the concrete class's members.
fn read_th1_base(
  r: &mut Reader,
) -> Result<(String, AxisInfo, AxisInfo, f64), String> {
  let (_, base_end) = r.read_version()?; // TH1 base class
  let base_end = base_end.ok_or("missing byte count on TH1")?;
  let (_name, title) = r.read_tnamed()?;
  r.skip_versioned()?; // TAttLine
  r.skip_versioned()?; // TAttFill
  r.skip_versioned()?; // TAttMarker
  let _ncells = r.read_i32()?;
  let x_axis = read_taxis(r)?;
  let y_axis = read_taxis(r)?;
  r.skip_versioned()?; // fZaxis
  let _bar_offset = r.read_u16()?;
  let _bar_width = r.read_u16()?;
  let entries = r.read_f64()?;
  // Remaining statistics, contours, sum-of-weights, and function list are
  // not part of the basic import; the byte count jumps straight past them.
  r.seek(base_end)?;
  Ok((title, x_axis, y_axis, entries))
}

/// The concrete histogram class contributes its TArray base: a count plus
/// the cell values, typed by the class suffix (C/S/I/F/D).
fn read_bin_array(
  r: &mut Reader,
  class_name: &str,
  budget: &mut usize,
) -> Result<Option<Vec<Expr>>, String> {
  let n = r.read_i32()?;
  if n < 0 {
    return Err("negative bin array length".into());
  }
  // Charge the shared listing budget so one file's histograms can't
  // collectively materialize gigabytes of cells. A histogram larger than the
  // remaining budget keeps its metadata but drops the bin array (`None`); its
  // full contents stay reachable through the object's element path.
  if n as usize > *budget {
    return Ok(None);
  }
  *budget -= n as usize;
  let kind = match class_name.as_bytes().last() {
    Some(b'C') => BasicKind::I8,
    Some(b'S') => BasicKind::I16,
    Some(b'I') => BasicKind::I32,
    Some(b'F') => BasicKind::F32,
    _ => BasicKind::F64,
  };
  let mut contents: Vec<Expr> =
    Vec::with_capacity((n as usize).min(MAX_PREALLOC));
  for _ in 0..n {
    contents.push(kind.read(r)?);
  }
  Ok(Some(contents))
}

/// TH1 family: name/title, x-axis definition, entry count, and the bin
/// content array (which includes underflow and overflow cells).
fn decode_th1(
  class_name: &str,
  payload: &[u8],
  budget: &mut usize,
) -> Result<Expr, String> {
  let mut r = Reader::new(payload);
  let (_, _) = r.read_version()?; // TH1x wrapper
  let (title, axis, _y_axis, entries) = read_th1_base(&mut r)?;
  let contents = read_bin_array(&mut r, class_name, budget)?;
  let AxisInfo {
    nbins,
    min: xmin,
    max: xmax,
    edges,
  } = axis;
  let mut pairs: Vec<(Expr, Expr)> = vec![
    (
      Expr::String("ClassName".into()),
      Expr::String(class_name.to_string()),
    ),
    (Expr::String("Title".into()), Expr::String(title)),
    (Expr::String("NBins".into()), Expr::Integer(nbins as i128)),
    (Expr::String("XMin".into()), Expr::Real(xmin)),
    (Expr::String("XMax".into()), Expr::Real(xmax)),
  ];
  if !edges.is_empty() {
    pairs.push((
      Expr::String("BinEdges".into()),
      Expr::List(edges.into_iter().map(Expr::Real).collect::<Vec<_>>().into()),
    ));
  }
  pairs.push((Expr::String("Entries".into()), Expr::Real(entries)));
  // A histogram over the listing budget keeps the metadata above but omits its
  // bin contents (fetch them via the object's element path).
  if let Some(mut contents) = contents {
    // The bin array holds nbins + 2 cells: index 0 is the underflow bin and
    // the last index the overflow bin.
    if contents.len() == nbins as usize + 2 {
      let overflow = contents.pop().expect("len >= 2");
      let underflow = contents.remove(0);
      pairs.push((
        Expr::String("BinContents".into()),
        Expr::List(contents.into()),
      ));
      pairs.push((Expr::String("Underflow".into()), underflow));
      pairs.push((Expr::String("Overflow".into()), overflow));
    } else {
      pairs.push((
        Expr::String("BinContents".into()),
        Expr::List(contents.into()),
      ));
    }
  }
  Ok(Expr::Association(pairs))
}

/// TH2 family: both axis definitions, entry count, and the bin contents as
/// an `NBinsX × NBinsY` matrix (row `i` holds x-bin `i` over all y-bins;
/// the underflow/overflow border cells are dropped).
fn decode_th2(
  class_name: &str,
  payload: &[u8],
  budget: &mut usize,
) -> Result<Expr, String> {
  let mut r = Reader::new(payload);
  let (_, _) = r.read_version()?; // TH2x wrapper
  let (_, th2_end) = r.read_version()?; // TH2 base class
  let th2_end = th2_end.ok_or("missing byte count on TH2")?;
  let (title, x_axis, y_axis, entries) = read_th1_base(&mut r)?;
  let _scale_factor = r.read_f64()?;
  let _tsumwy = r.read_f64()?;
  let _tsumwy2 = r.read_f64()?;
  let _tsumwxy = r.read_f64()?;
  r.seek(th2_end)?;
  let contents = read_bin_array(&mut r, class_name, budget)?;
  let (nx, ny) = (x_axis.nbins.max(0) as usize, y_axis.nbins.max(0) as usize);
  let mut pairs: Vec<(Expr, Expr)> = vec![
    (
      Expr::String("ClassName".into()),
      Expr::String(class_name.to_string()),
    ),
    (Expr::String("Title".into()), Expr::String(title)),
    (
      Expr::String("NBinsX".into()),
      Expr::Integer(x_axis.nbins as i128),
    ),
    (Expr::String("XMin".into()), Expr::Real(x_axis.min)),
    (Expr::String("XMax".into()), Expr::Real(x_axis.max)),
    (
      Expr::String("NBinsY".into()),
      Expr::Integer(y_axis.nbins as i128),
    ),
    (Expr::String("YMin".into()), Expr::Real(y_axis.min)),
    (Expr::String("YMax".into()), Expr::Real(y_axis.max)),
  ];
  if !x_axis.edges.is_empty() {
    pairs.push((
      Expr::String("XBinEdges".into()),
      Expr::List(
        x_axis
          .edges
          .into_iter()
          .map(Expr::Real)
          .collect::<Vec<_>>()
          .into(),
      ),
    ));
  }
  if !y_axis.edges.is_empty() {
    pairs.push((
      Expr::String("YBinEdges".into()),
      Expr::List(
        y_axis
          .edges
          .into_iter()
          .map(Expr::Real)
          .collect::<Vec<_>>()
          .into(),
      ),
    ));
  }
  pairs.push((Expr::String("Entries".into()), Expr::Real(entries)));
  // A histogram over the listing budget keeps the metadata above but omits its
  // bin contents (fetch them via the object's element path).
  if let Some(contents) = contents {
    if contents.len() != (nx + 2) * (ny + 2) {
      return Err("2-D bin array size mismatch".into());
    }
    // ROOT stores cells as a flat array indexed binx + (nbinsx+2)*biny,
    // where index 0 of each axis is the underflow bin.
    let mut rows: Vec<Expr> = Vec::with_capacity(nx);
    for x in 1..=nx {
      let mut row: Vec<Expr> = Vec::with_capacity(ny);
      for y in 1..=ny {
        row.push(contents[x + (nx + 2) * y].clone());
      }
      rows.push(Expr::List(row.into()));
    }
    pairs.push((Expr::String("BinContents".into()), Expr::List(rows.into())));
  }
  Ok(Expr::Association(pairs))
}

/// One leaf of a branch: the concrete `TLeaf` subclass plus the fields
/// deciding how basket bytes map to values.
struct LeafInfo {
  class: String,
  /// Fixed number of values per entry (`fLen`), 1 for scalars.
  len: i32,
  /// `fIsUnsigned`: reinterpret the integer type as its unsigned twin.
  unsigned: bool,
  /// A counter leaf drives this leaf's per-entry length (jagged data).
  has_count: bool,
}

/// Everything needed to both list a branch and read its basket data.
struct BranchInfo {
  name: String,
  title: String,
  /// `fClassName` of a `TBranchElement` (e.g. `vector<double>`).
  element_class: Option<String>,
  entries: i64,
  /// Absolute file offset of each written basket's key record.
  basket_seek: Vec<u64>,
  leaves: Vec<LeafInfo>,
  sub_branches: Vec<BranchInfo>,
}

impl BranchInfo {
  /// The user-facing type string: the leaf specification title for plain
  /// `TBranch`es (`"Run/I"`), the streamed class for `TBranchElement`s.
  fn type_string(&self) -> String {
    match &self.element_class {
      Some(c) => c.clone(),
      None => self.title.clone(),
    }
  }
}

/// Registry of "new class" tags already seen in this object stream, so
/// class *references* (`tag & K_CLASS_MASK`) can be resolved to a name.
/// Keys are stream positions shifted by the enclosing key length plus
/// `K_MAP_OFFSET`, matching TBufferFile's bookkeeping.
struct ClassMap {
  map: std::collections::HashMap<u32, String>,
  key_len: u32,
}

impl ClassMap {
  fn new(key_len: u32) -> Self {
    ClassMap {
      map: std::collections::HashMap::new(),
      key_len,
    }
  }

  fn record(&mut self, tag_pos: usize, name: &str) {
    self.map.insert(
      tag_pos as u32 + self.key_len + K_MAP_OFFSET,
      name.to_string(),
    );
  }

  fn resolve(&self, tag: u32) -> Option<&str> {
    self.map.get(&(tag & !K_CLASS_MASK)).map(|s| s.as_str())
  }
}

/// Read the class tag of the next object slot in a `TObjArray` and return
/// `(class name, end-of-object position)`, or `None` for null slots and
/// bare object back-references (which carry no payload).
fn read_object_slot(
  r: &mut Reader,
  classes: &mut ClassMap,
) -> Result<Option<(String, usize)>, String> {
  let raw = r.read_u32()?;
  if raw == 0 || raw & K_BYTE_COUNT_MASK == 0 {
    return Ok(None);
  }
  let obj_end = r.pos + (raw & K_BYTE_COUNT_VALUE) as usize;
  let tag_pos = r.pos;
  let tag = r.read_u32()?;
  let class = if tag == K_NEW_CLASS_TAG {
    let name = r.read_cstring()?;
    classes.record(tag_pos, &name);
    name
  } else if tag & K_CLASS_MASK != 0 {
    classes
      .resolve(tag)
      .ok_or("unresolved class reference in object stream")?
      .to_string()
  } else {
    return Err("unexpected object reference in class slot".into());
  };
  Ok(Some((class, obj_end)))
}

/// Skip an object *pointer* member (e.g. `fLeafCount`, `fBranchCount`):
/// null tag, back-reference, or a full inline object. Returns whether the
/// pointer was non-null.
fn skip_object_pointer(r: &mut Reader) -> Result<bool, String> {
  let raw = r.read_u32()?;
  if raw == 0 {
    return Ok(false);
  }
  if raw & K_BYTE_COUNT_MASK != 0 {
    // Inline object: jump past its byte-counted payload.
    r.seek(r.pos + (raw & K_BYTE_COUNT_VALUE) as usize)?;
  }
  Ok(true)
}

/// Parse the `fLeaves` TObjArray of a branch.
fn parse_leaves(
  r: &mut Reader,
  classes: &mut ClassMap,
) -> Result<Vec<LeafInfo>, String> {
  let (_, end) = r.read_version()?;
  let end = end.ok_or("missing byte count on fLeaves")?;
  r.skip_tobject()?;
  let _array_name = r.read_tstring()?;
  let size = r.read_i32()?;
  let _lower_bound = r.read_i32()?;
  let mut leaves = Vec::with_capacity((size.max(0) as usize).min(16));
  for _ in 0..size.max(0) {
    let Some((class, obj_end)) = read_object_slot(r, classes)? else {
      continue;
    };
    // TLeafX wrapper version, then the TLeaf base segment.
    let (_, _) = r.read_version()?;
    let (_, _) = r.read_version()?;
    let (_name, _title) = r.read_tnamed()?;
    let len = r.read_i32()?;
    let _len_type = r.read_i32()?;
    let _offset = r.read_i32()?;
    let _is_range = r.read_u8()?;
    let unsigned = r.read_u8()? != 0;
    let has_count = skip_object_pointer(r)?;
    leaves.push(LeafInfo {
      class,
      len,
      unsigned,
      has_count,
    });
    r.seek(obj_end)?;
  }
  r.seek(end)?;
  Ok(leaves)
}

/// Parse the `TBranch` base-class segment (layout of class versions 12+,
/// i.e. ROOT 6). Positions the reader at the segment end on success.
fn parse_tbranch_base(
  r: &mut Reader,
  classes: &mut ClassMap,
) -> Result<BranchInfo, String> {
  let (version, end) = r.read_version()?;
  let end = end.ok_or("missing byte count on TBranch")?;
  if version < 12 {
    return Err(format!("TBranch version {} not supported", version));
  }
  let (name, title) = r.read_tnamed()?;
  r.skip_versioned()?; // TAttFill
  let _compress = r.read_i32()?;
  let _basket_size = r.read_i32()?;
  let _entry_offset_len = r.read_i32()?;
  let write_basket = r.read_i32()?;
  let _entry_number = r.read_i64()?;
  if version >= 13 {
    r.skip_versioned()?; // fIOFeatures
  }
  let _offset = r.read_i32()?;
  let max_baskets = r.read_i32()?;
  let _split_level = r.read_i32()?;
  let entries = r.read_i64()?;
  let _first_entry = r.read_i64()?;
  let _tot_bytes = r.read_i64()?;
  let _zip_bytes = r.read_i64()?;
  let sub_branches = parse_branch_array(r, classes)?;
  let leaves = parse_leaves(r, classes)?;
  r.skip_versioned()?; // fBaskets (in-memory baskets; not needed)
  if max_baskets < 0 || write_basket < 0 || write_basket > max_baskets {
    return Err("inconsistent basket counts".into());
  }
  // fBasketBytes / fBasketEntry / fBasketSeek: counted basic-type arrays,
  // each preceded by a 1-byte instance marker.
  r.take(1)?;
  r.take(4 * max_baskets as usize)?;
  r.take(1)?;
  r.take(8 * max_baskets as usize)?;
  r.take(1)?;
  let mut basket_seek =
    Vec::with_capacity((write_basket as usize).min(MAX_PREALLOC));
  for i in 0..max_baskets {
    let seek = r.read_i64()?;
    if i < write_basket {
      basket_seek.push(seek as u64);
    }
  }
  let _file_name = r.read_tstring()?;
  r.seek(end)?;
  Ok(BranchInfo {
    name,
    title,
    element_class: None,
    entries,
    basket_seek,
    leaves,
    sub_branches,
  })
}

/// Parse one branch object (either a plain `TBranch` or a
/// `TBranchElement`, which wraps a `TBranch` base segment).
fn parse_branch(
  r: &mut Reader,
  class: &str,
  classes: &mut ClassMap,
) -> Result<BranchInfo, String> {
  match class {
    "TBranch" => parse_tbranch_base(r, classes),
    "TBranchElement" => {
      let (_, end) = r.read_version()?;
      let end = end.ok_or("missing byte count on TBranchElement")?;
      let mut info = parse_tbranch_base(r, classes)?;
      let class_name = r.read_tstring()?;
      let _parent_name = r.read_tstring()?;
      let _clones_name = r.read_tstring()?;
      let _checksum = r.read_u32()?;
      let _class_version = r.read_u16()?;
      let _id = r.read_i32()?;
      let _type = r.read_i32()?;
      let _streamer_type = r.read_i32()?;
      let _maximum = r.read_i32()?;
      let _branch_count = skip_object_pointer(r)?;
      let _branch_count2 = skip_object_pointer(r)?;
      info.element_class = Some(class_name);
      r.seek(end)?;
      Ok(info)
    }
    other => Err(format!("branch class {} not supported", other)),
  }
}

/// Parse a `TObjArray` of branches (`fBranches` of a tree or branch).
fn parse_branch_array(
  r: &mut Reader,
  classes: &mut ClassMap,
) -> Result<Vec<BranchInfo>, String> {
  let (_, end) = r.read_version()?;
  let end = end.ok_or("missing byte count on fBranches")?;
  r.skip_tobject()?;
  let _array_name = r.read_tstring()?;
  let size = r.read_i32()?;
  let _lower_bound = r.read_i32()?;
  if size < 0 {
    return Err("negative branch count".into());
  }
  let mut branches = Vec::with_capacity((size as usize).min(MAX_PREALLOC));
  for _ in 0..size {
    let Some((class, obj_end)) = read_object_slot(r, classes)? else {
      continue;
    };
    branches.push(parse_branch(r, &class, classes)?);
    r.seek(obj_end)?;
  }
  r.seek(end)?;
  Ok(branches)
}

/// Fully parsed tree header: entry count plus recursive branch metadata.
struct TreeInfo {
  title: String,
  entries: i64,
  branches: Vec<BranchInfo>,
}

/// Parse a streamed `TTree` object down to its branch list. The scalar
/// block between fEntries and fBranches is only walked for the layouts
/// shipped with ROOT 6 (class versions 19 and 20); older versions still
/// report their entry count, with an error in place of the branches.
fn parse_ttree(
  payload: &[u8],
  key_len: u32,
) -> Result<(TreeInfo, Option<String>), String> {
  let mut r = Reader::new(payload);
  let (version, _) = r.read_version()?;
  let (_name, title) = r.read_tnamed()?;
  r.skip_versioned()?; // TAttLine
  r.skip_versioned()?; // TAttFill
  r.skip_versioned()?; // TAttMarker
  let entries: i64 = if version >= 16 {
    r.read_i64()?
  } else {
    r.read_f64()? as i64
  };
  let mut info = TreeInfo {
    title,
    entries,
    branches: Vec::new(),
  };
  if !(19..=20).contains(&version) {
    return Ok((
      info,
      Some(format!("TTree version {} branches not decoded", version)),
    ));
  }
  let branches = (|| -> Result<Vec<BranchInfo>, String> {
    // fTotBytes, fZipBytes, fSavedBytes, fFlushedBytes
    r.take(4 * 8)?;
    let _weight = r.read_f64()?;
    // fTimerInterval, fScanField, fUpdate, fDefaultEntryOffsetLen
    r.take(4 * 4)?;
    let n_cluster_range = r.read_i32()?;
    if n_cluster_range < 0 {
      return Err("negative cluster range count".into());
    }
    // fMaxEntries, fMaxEntryLoop, fMaxVirtualSize, fAutoSave, fAutoFlush,
    // fEstimate
    r.take(6 * 8)?;
    // fClusterRangeEnd and fClusterSize: counted basic-type arrays, each
    // preceded by a 1-byte instance marker.
    for _ in 0..2 {
      r.take(1)?;
      r.take(8 * n_cluster_range as usize)?;
    }
    if version >= 20 {
      r.skip_versioned()?; // fIOFeatures
    }
    let mut classes = ClassMap::new(key_len);
    parse_branch_array(&mut r, &mut classes)
  })();
  match branches {
    Ok(b) => {
      info.branches = b;
      Ok((info, None))
    }
    Err(e) => Ok((info, Some(e))),
  }
}

/// TTree → metadata Association (entry count and branch types). Column
/// values are only produced through explicit element selectors (see
/// [`decode_element`]) so listing a file with a multi-GB tree stays cheap.
fn decode_ttree(key: &KeyInfo, payload: &[u8]) -> Result<Expr, String> {
  let (tree, branch_err) = parse_ttree(payload, key.key_len as u32)?;
  let mut pairs: Vec<(Expr, Expr)> = vec![
    (
      Expr::String("ClassName".into()),
      Expr::String("TTree".into()),
    ),
    (
      Expr::String("Title".into()),
      Expr::String(tree.title.clone()),
    ),
    (
      Expr::String("Entries".into()),
      Expr::Integer(tree.entries as i128),
    ),
  ];
  if branch_err.is_none() {
    pairs.push((
      Expr::String("Branches".into()),
      Expr::Association(
        tree
          .branches
          .iter()
          .map(|b| {
            (Expr::String(b.name.clone()), Expr::String(b.type_string()))
          })
          .collect(),
      ),
    ));
  }
  Ok(Expr::Association(pairs))
}

/// The basic value types a leaf or vector element can stream as.
#[derive(Clone, Copy, PartialEq)]
enum BasicKind {
  I8,
  U8,
  I16,
  U16,
  I32,
  U32,
  I64,
  U64,
  F32,
  F64,
  Bool,
}

impl BasicKind {
  fn size(self) -> usize {
    match self {
      BasicKind::I8 | BasicKind::U8 | BasicKind::Bool => 1,
      BasicKind::I16 | BasicKind::U16 => 2,
      BasicKind::I32 | BasicKind::U32 | BasicKind::F32 => 4,
      BasicKind::I64 | BasicKind::U64 | BasicKind::F64 => 8,
    }
  }

  fn read(self, r: &mut Reader) -> Result<Expr, String> {
    Ok(match self {
      BasicKind::I8 => Expr::Integer(r.read_u8()? as i8 as i128),
      BasicKind::U8 => Expr::Integer(r.read_u8()? as i128),
      BasicKind::I16 => Expr::Integer(r.read_i16()? as i128),
      BasicKind::U16 => Expr::Integer(r.read_u16()? as i128),
      BasicKind::I32 => Expr::Integer(r.read_i32()? as i128),
      BasicKind::U32 => Expr::Integer(r.read_u32()? as i128),
      BasicKind::I64 => Expr::Integer(r.read_i64()? as i128),
      BasicKind::U64 => Expr::Integer(r.read_i64()? as u64 as i128),
      BasicKind::F32 => Expr::Real(r.read_f32()? as f64),
      BasicKind::F64 => Expr::Real(r.read_f64()?),
      BasicKind::Bool => Expr::Identifier(
        if r.read_u8()? != 0 { "True" } else { "False" }.to_string(),
      ),
    })
  }
}

/// How a branch's basket bytes map to per-entry values.
enum Interp {
  /// Leaf of a basic type: `len` values per entry; `jagged` when a counter
  /// leaf makes the per-entry length variable (then the basket's entry
  /// offsets determine each length).
  Basic {
    kind: BasicKind,
    len: usize,
    jagged: bool,
  },
  /// `std::vector<basic>`: per-entry 10-byte header, then the elements.
  VectorBasic { kind: BasicKind },
  /// A streamed `TLorentzVector` object per entry.
  LorentzVector,
}

fn leaf_basic_kind(leaf: &LeafInfo) -> Result<BasicKind, String> {
  Ok(match (leaf.class.as_str(), leaf.unsigned) {
    ("TLeafB", false) => BasicKind::I8,
    ("TLeafB", true) => BasicKind::U8,
    ("TLeafS", false) => BasicKind::I16,
    ("TLeafS", true) => BasicKind::U16,
    ("TLeafI", false) => BasicKind::I32,
    ("TLeafI", true) => BasicKind::U32,
    ("TLeafL", false) => BasicKind::I64,
    ("TLeafL", true) => BasicKind::U64,
    ("TLeafF", _) => BasicKind::F32,
    ("TLeafD", _) => BasicKind::F64,
    ("TLeafO", _) => BasicKind::Bool,
    (other, _) => return Err(format!("leaf class {} not decoded", other)),
  })
}

fn vector_element_kind(element: &str) -> Result<BasicKind, String> {
  Ok(match element.trim() {
    "double" | "Double_t" => BasicKind::F64,
    "float" | "Float_t" => BasicKind::F32,
    "int" | "Int_t" => BasicKind::I32,
    "unsigned int" | "UInt_t" => BasicKind::U32,
    "short" | "Short_t" => BasicKind::I16,
    "unsigned short" | "UShort_t" => BasicKind::U16,
    "long long" | "Long64_t" | "long" => BasicKind::I64,
    "unsigned long long" | "ULong64_t" | "unsigned long" => BasicKind::U64,
    "char" | "Char_t" => BasicKind::I8,
    "unsigned char" | "UChar_t" => BasicKind::U8,
    "bool" | "Bool_t" => BasicKind::Bool,
    other => {
      return Err(format!("vector element type {} not decoded", other));
    }
  })
}

/// Decide how to decode a branch's baskets from its metadata.
fn resolve_interp(branch: &BranchInfo) -> Result<Interp, String> {
  if !branch.sub_branches.is_empty() {
    return Err("split branches are not decoded".into());
  }
  if let Some(class) = &branch.element_class {
    if class == "TLorentzVector" {
      return Ok(Interp::LorentzVector);
    }
    if let Some(inner) = class
      .strip_prefix("vector<")
      .and_then(|s| s.strip_suffix('>'))
    {
      return Ok(Interp::VectorBasic {
        kind: vector_element_kind(inner)?,
      });
    }
    return Err(format!("branch class {} not decoded", class));
  }
  let [leaf] = branch.leaves.as_slice() else {
    return Err("multi-leaf branches are not decoded".into());
  };
  Ok(Interp::Basic {
    kind: leaf_basic_kind(leaf)?,
    len: leaf.len.max(1) as usize,
    jagged: leaf.has_count,
  })
}

/// One decompressed basket: the entry bytes, and the byte range of each
/// entry when the basket carries an entry-offset table.
struct BasketData {
  payload: Vec<u8>,
  /// End of the entry bytes (`fLast - fKeyLen`).
  border: usize,
  /// Per-entry start positions within `payload[..border]`.
  offsets: Option<Vec<usize>>,
  n_entries: usize,
}

impl BasketData {
  /// Byte range of entry `i` (requires an offset table).
  fn entry_range(&self, i: usize) -> Result<(usize, usize), String> {
    let offsets = self.offsets.as_ref().ok_or("basket has no offset table")?;
    let start = offsets[i];
    let end = if i + 1 < offsets.len() {
      offsets[i + 1]
    } else {
      self.border
    };
    if start > end || end > self.border {
      return Err("corrupt basket entry offsets".into());
    }
    Ok((start, end))
  }
}

/// Read and decompress the basket stored at file offset `seek`.
fn read_basket(file: &[u8], seek: u64) -> Result<BasketData, String> {
  let mut r = Reader::new(file);
  r.seek(seek as usize)?;
  let key = read_key(&mut r)?;
  if key.class_name != "TBasket" {
    return Err(format!("expected TBasket, found {}", key.class_name));
  }
  // TBasket appends its own members to the key header.
  let _basket_version = r.read_i16()?;
  let _buffer_size = r.read_i32()?;
  let _nev_buf_size = r.read_i32()?;
  let n_entries = r.read_i32()?;
  let last = r.read_i32()?;
  let _flag = r.read_u8()?;
  if n_entries < 0 || last < key.key_len as i32 {
    return Err("corrupt basket header".into());
  }
  let mut key = key;
  key.seek_key = seek;
  let payload = object_payload(file, &key)?;
  let border = (last as usize) - key.key_len as usize;
  if border > payload.len() {
    return Err("basket data extends past payload".into());
  }
  let n = n_entries as usize;
  // Entries of variable size carry a table of per-entry displacements
  // after the data: a count word, then `n` absolute positions (offset by
  // the key length).
  let offsets = if payload.len() > border {
    let mut or = Reader::new(&payload);
    or.seek(border)?;
    let count = or.read_i32()?;
    if (count as usize) < n {
      return Err("basket offset table too short".into());
    }
    let mut offsets = Vec::with_capacity(n.min(MAX_PREALLOC));
    for _ in 0..n {
      let off = or.read_i32()? as i64 - key.key_len as i64;
      if off < 0 || off as usize > border {
        return Err("basket entry offset out of range".into());
      }
      offsets.push(off as usize);
    }
    if !offsets.is_sorted() {
      return Err("basket entry offsets not ascending".into());
    }
    Some(offsets)
  } else {
    None
  };
  Ok(BasketData {
    payload,
    border,
    offsets,
    n_entries: n,
  })
}

/// Decode a `std::vector<basic>` entry: byte count, version word, element
/// count, then the elements.
fn decode_vector_entry(bytes: &[u8], kind: BasicKind) -> Result<Expr, String> {
  let mut r = Reader::new(bytes);
  let _byte_count = r.read_u32()?;
  let version = r.read_u16()?;
  if version & 0x4000 != 0 {
    return Err("member-wise streamed vectors are not decoded".into());
  }
  let n = r.read_i32()?;
  if n < 0 || bytes.len() < 10 + n as usize * kind.size() {
    return Err("corrupt vector entry".into());
  }
  let mut values = Vec::with_capacity((n as usize).min(MAX_PREALLOC));
  for _ in 0..n {
    values.push(kind.read(&mut r)?);
  }
  Ok(Expr::List(values.into()))
}

/// Decode a streamed `TLorentzVector`: TObject base, a `TVector3` with the
/// momentum components, then the energy.
fn decode_lorentz_entry(bytes: &[u8]) -> Result<Expr, String> {
  let mut r = Reader::new(bytes);
  let (_, _) = r.read_version()?; // TLorentzVector
  r.skip_tobject()?;
  let (_, _) = r.read_version()?; // TVector3
  r.skip_tobject()?;
  let px = r.read_f64()?;
  let py = r.read_f64()?;
  let pz = r.read_f64()?;
  let e = r.read_f64()?;
  Ok(Expr::Association(vec![
    (Expr::String("Px".into()), Expr::Real(px)),
    (Expr::String("Py".into()), Expr::Real(py)),
    (Expr::String("Pz".into()), Expr::Real(pz)),
    (Expr::String("E".into()), Expr::Real(e)),
  ]))
}

/// Read every basket of a branch and decode its column of values.
fn branch_values(
  file: &[u8],
  branch: &BranchInfo,
) -> Result<Vec<Expr>, String> {
  let interp = resolve_interp(branch)?;
  let mut values: Vec<Expr> =
    Vec::with_capacity((branch.entries.max(0) as usize).min(MAX_PREALLOC));
  for &seek in &branch.basket_seek {
    if seek == 0 {
      return Err("branch basket missing from file".into());
    }
    let basket = read_basket(file, seek)?;
    match &interp {
      Interp::Basic { kind, len, jagged } => {
        if *jagged {
          for i in 0..basket.n_entries {
            let (start, end) = basket.entry_range(i)?;
            let n_bytes = end - start;
            if n_bytes % kind.size() != 0 {
              return Err("entry size not a multiple of the leaf size".into());
            }
            let mut r = Reader::new(&basket.payload[start..end]);
            let n = n_bytes / kind.size();
            let mut entry = Vec::with_capacity(n.min(MAX_PREALLOC));
            for _ in 0..n {
              entry.push(kind.read(&mut r)?);
            }
            values.push(Expr::List(entry.into()));
          }
        } else {
          let mut r = Reader::new(&basket.payload[..basket.border]);
          for _ in 0..basket.n_entries {
            if *len == 1 {
              values.push(kind.read(&mut r)?);
            } else {
              let mut entry = Vec::with_capacity(*len);
              for _ in 0..*len {
                entry.push(kind.read(&mut r)?);
              }
              values.push(Expr::List(entry.into()));
            }
          }
        }
      }
      Interp::VectorBasic { kind } => {
        for i in 0..basket.n_entries {
          let (start, end) = basket.entry_range(i)?;
          values.push(decode_vector_entry(&basket.payload[start..end], *kind)?);
        }
      }
      Interp::LorentzVector => {
        for i in 0..basket.n_entries {
          let (start, end) = basket.entry_range(i)?;
          values.push(decode_lorentz_entry(&basket.payload[start..end])?);
        }
      }
    }
  }
  Ok(values)
}
