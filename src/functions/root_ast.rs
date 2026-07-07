//! Minimal reader for CERN ROOT files (<https://root.cern/>).
//!
//! `Import["file.root"]` walks the file's directory structure and returns an
//! Association mapping each stored object's name to a decoded value:
//!
//! - `TObjString` → the contained String
//! - `TH1C` / `TH1S` / `TH1I` / `TH1F` / `TH1D` → an Association with the
//!   axis definition, entry count, and bin contents
//! - `TTree` → an Association with the entry count and the branch
//!   name → title (leaf specification) mapping
//! - `TDirectory` → a nested Association of the directory's contents
//! - any other class → an Association with `"ClassName"` and `"Title"` so
//!   the object is at least visible
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
/// `TObject::fBits` flag: a 2-byte process id follows the bits field.
const K_IS_REFERENCED: u32 = 1 << 4;
/// Directories may nest; guard against reference cycles in corrupt files.
const MAX_DIR_DEPTH: usize = 16;
/// Upper bound for `Vec::with_capacity` calls driven by on-file counts, so
/// a corrupt length field fails with a read error instead of a huge
/// allocation. Vectors still grow past this if the data really is there.
const MAX_PREALLOC: usize = 65_536;

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
  let seek_keys = dir_seek_keys(data, begin as usize + nbytes_name as usize)?;
  read_directory(data, seek_keys, 0)
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

/// Read the key list at `seek_keys` and decode every object in the
/// directory into an Association, recursing into subdirectories.
fn read_directory(
  data: &[u8],
  seek_keys: u64,
  depth: usize,
) -> Result<Expr, String> {
  if depth > MAX_DIR_DEPTH {
    return Err("directory nesting too deep".into());
  }
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
        read_directory(data, sub_seek, depth + 1)?
      } else {
        match object_payload(data, key) {
          Ok(payload) => decode_object(key, &payload),
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
fn decode_object(key: &KeyInfo, payload: &[u8]) -> Expr {
  let decoded = match key.class_name.as_str() {
    "TObjString" => decode_tobjstring(payload),
    "TH1C" | "TH1S" | "TH1I" | "TH1F" | "TH1D" => {
      decode_th1(&key.class_name, payload)
    }
    "TTree" => decode_ttree(payload),
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

/// TAxis: extract the bin definition, then skip the remaining attributes.
/// Returns (nbins, xmin, xmax, explicit bin edges for variable-width axes).
fn read_taxis(r: &mut Reader) -> Result<(i32, f64, f64, Vec<f64>), String> {
  let (_, end) = r.read_version()?;
  let end = end.ok_or("missing byte count on TAxis")?;
  let _ = r.read_tnamed()?;
  r.skip_versioned()?; // TAttAxis
  let nbins = r.read_i32()?;
  let xmin = r.read_f64()?;
  let xmax = r.read_f64()?;
  // fXbins: a TArrayD of bin edges; empty for fixed-width binning.
  let n_edges = r.read_i32()?;
  let mut edges =
    Vec::with_capacity((n_edges.max(0) as usize).min(MAX_PREALLOC));
  for _ in 0..n_edges.max(0) {
    edges.push(r.read_f64()?);
  }
  r.seek(end)?;
  Ok((nbins, xmin, xmax, edges))
}

/// TH1 family: name/title, x-axis definition, entry count, and the bin
/// content array (which includes underflow and overflow cells).
fn decode_th1(class_name: &str, payload: &[u8]) -> Result<Expr, String> {
  let mut r = Reader::new(payload);
  let (_, _) = r.read_version()?; // TH1x wrapper
  let (_, base_end) = r.read_version()?; // TH1 base class
  let base_end = base_end.ok_or("missing byte count on TH1")?;
  let (_name, title) = r.read_tnamed()?;
  r.skip_versioned()?; // TAttLine
  r.skip_versioned()?; // TAttFill
  r.skip_versioned()?; // TAttMarker
  let _ncells = r.read_i32()?;
  let (nbins, xmin, xmax, edges) = read_taxis(&mut r)?;
  r.skip_versioned()?; // fYaxis
  r.skip_versioned()?; // fZaxis
  let _bar_offset = r.read_u16()?;
  let _bar_width = r.read_u16()?;
  let entries = r.read_f64()?;
  // Remaining statistics, contours, sum-of-weights, and function list are
  // not part of the basic import; the byte count jumps straight past them.
  r.seek(base_end)?;
  // The concrete class contributes its TArray base: a count plus values.
  let n = r.read_i32()?;
  if n < 0 {
    return Err("negative bin array length".into());
  }
  let mut contents: Vec<Expr> =
    Vec::with_capacity((n as usize).min(MAX_PREALLOC));
  for _ in 0..n {
    contents.push(match class_name {
      "TH1C" => Expr::Integer(r.read_u8()? as i8 as i128),
      "TH1S" => Expr::Integer(r.read_u16()? as i16 as i128),
      "TH1I" => Expr::Integer(r.read_i32()? as i128),
      "TH1F" => Expr::Real(r.read_f32()? as f64),
      _ => Expr::Real(r.read_f64()?),
    });
  }
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
  Ok(Expr::Association(pairs))
}

/// TTree: entry count plus the branch name → title (leaf specification)
/// mapping. Branch *data* (baskets) is beyond this basic importer.
fn decode_ttree(payload: &[u8]) -> Result<Expr, String> {
  let mut r = Reader::new(payload);
  let (version, _) = r.read_version()?;
  let (_name, title) = r.read_tnamed()?;
  r.skip_versioned()?; // TAttLine
  r.skip_versioned()?; // TAttFill
  r.skip_versioned()?; // TAttMarker
  let entries: i128 = if version >= 16 {
    r.read_i64()? as i128
  } else {
    r.read_f64()? as i128
  };
  let mut pairs: Vec<(Expr, Expr)> = vec![
    (
      Expr::String("ClassName".into()),
      Expr::String("TTree".into()),
    ),
    (Expr::String("Title".into()), Expr::String(title)),
    (Expr::String("Entries".into()), Expr::Integer(entries)),
  ];
  // The scalar block between fEntries and fBranches is only walked for the
  // layouts shipped with ROOT 6 (class versions 19 and 20); older files
  // still report their entry count above.
  if version >= 19
    && let Ok(branches) = read_tree_branches(&mut r, version)
  {
    pairs.push((
      Expr::String("Branches".into()),
      Expr::Association(
        branches
          .into_iter()
          .map(|(n, t)| (Expr::String(n), Expr::String(t)))
          .collect(),
      ),
    ));
  }
  Ok(Expr::Association(pairs))
}

fn read_tree_branches(
  r: &mut Reader,
  version: u16,
) -> Result<Vec<(String, String)>, String> {
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
  // fClusterRangeEnd and fClusterSize: each a counted basic-type array
  // preceded by a 1-byte instance marker.
  for _ in 0..2 {
    r.take(1)?;
    r.take(8 * n_cluster_range as usize)?;
  }
  if version >= 20 {
    r.skip_versioned()?; // fIOFeatures
  }
  // fBranches: a TObjArray of TBranch objects.
  let (_, end) = r.read_version()?;
  let end = end.ok_or("missing byte count on TObjArray")?;
  r.skip_tobject()?;
  let _array_name = r.read_tstring()?;
  let size = r.read_i32()?;
  let _lower_bound = r.read_i32()?;
  if size < 0 {
    return Err("negative branch count".into());
  }
  let mut branches = Vec::with_capacity((size as usize).min(MAX_PREALLOC));
  for _ in 0..size {
    let raw = r.read_u32()?;
    if raw == 0 {
      // Null slot in the array.
      continue;
    }
    if raw & K_BYTE_COUNT_MASK == 0 {
      // Back-reference to an already streamed object: no payload follows.
      continue;
    }
    let obj_end = r.pos + (raw & K_BYTE_COUNT_VALUE) as usize;
    let tag = r.read_u32()?;
    if tag == K_NEW_CLASS_TAG {
      let _class_name = r.read_cstring()?;
    }
    // The branch's own data: version header, then its TNamed base carrying
    // the name and the leaf-specification title. Everything after that is
    // skipped via the enclosing byte count.
    let (_, _) = r.read_version()?;
    let (name, title) = r.read_tnamed()?;
    branches.push((name, title));
    r.seek(obj_end)?;
  }
  r.seek(end)?;
  Ok(branches)
}
