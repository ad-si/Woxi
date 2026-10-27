//! Generate `resources/gazetteer.tsv.gz`, the country / subdivision gazetteer
//! that `src/functions/geographics.rs` embeds to resolve geographic
//! `Entity[…]` specifications to positions.
//!
//! The data comes from the keshvar crate (ISO 3166 names, translations and
//! centers). keshvar itself is not a dependency of Woxi: it compiles its data
//! into code (one constructor function per country), which made it ~16 MiB of
//! the release binary. This dump holds only the fields Woxi uses.
//!
//! Format: gzipped UTF-8, one tab-separated record per line.
//!
//!     C <alpha2> <lat> <lon> <iso short name> <iso long name> <translation>…
//!     S <code> <lat> <lon> <local name> <translation>…
//!
//! An `S` line is a subdivision of the nearest preceding `C` line. Countries
//! keep keshvar's iteration order (later ones win name collisions, as with
//! keshvar's own index); subdivisions are sorted by code, translations sorted
//! and deduplicated. Subdivisions without coordinates are left out.
//!
//! Run from the repository root:
//!
//!     cargo run --release --manifest-path scripts/gen_gazetteer/Cargo.toml

use flate2::{Compression, write::GzEncoder};
use std::io::Write;

fn field(s: &str) -> &str {
  assert!(!s.contains(['\t', '\n', '\r']), "unexpected separator in {s:?}");
  s
}

fn push_names<'a>(line: &mut String, names: impl Iterator<Item = &'a str>) {
  let mut names: Vec<&str> = names.collect();
  names.sort_unstable();
  names.dedup();
  for n in names {
    line.push('\t');
    line.push_str(field(n));
  }
}

fn main() {
  let mut out = String::new();
  for c in keshvar::CountryIterator::new() {
    let g = c.geo();
    out.push_str(&format!(
      "C\t{}\t{}\t{}\t{}\t{}",
      c.alpha2().to_string(),
      g.latitude(),
      g.longitude(),
      field(c.iso_short_name()),
      field(c.iso_long_name()),
    ));
    push_names(&mut out, c.translations().values().copied());
    out.push('\n');

    let mut subs: Vec<_> = c.subdivisions().iter().collect();
    subs.sort_unstable_by_key(|(code, _)| **code);
    for (code, s) in subs {
      let Some(g) = s.geo() else { continue };
      let (Some(lat), Some(lon)) = (g.latitude(), g.longitude()) else {
        continue;
      };
      out.push_str(&format!(
        "S\t{}\t{lat}\t{lon}\t{}",
        field(code),
        field(s.name())
      ));
      push_names(&mut out, s.translations().values().copied());
      out.push('\n');
    }
  }

  let path = "resources/gazetteer.tsv.gz";
  let mut enc = GzEncoder::new(Vec::new(), Compression::best());
  enc.write_all(out.as_bytes()).unwrap();
  std::fs::write(path, enc.finish().unwrap()).unwrap();
  eprintln!("wrote {path} ({} bytes uncompressed)", out.len());
}
