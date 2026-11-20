//! Geometry of time-zone entities (`Entity["TimeZone", "GMT+9"]`) for
//! `GeoGraphics`.
//!
//! A zone `GMT±h` is modeled as the places that keep that standard time:
//!   * the open-ocean part of its nautical zone — the 15°-wide longitude band
//!     centered on `15·h` degrees (half bands for ±12), with every country's
//!     land cut out (Antarctica keeps nautical time, so it stays in the band);
//!   * the land of every country (or part of a country) whose standard time
//!     is `GMT±h`.
//!
//! Countries spanning several zones are split into approximate latitude /
//! longitude boxes. The country geometry is the embedded Natural Earth
//! 1:110m data set, so the result is a coarse but faithful overview map.

/// A closed ring of `(lon, lat)` degrees.
pub type Ring = Vec<(f64, f64)>;

/// The UTC offset in hours named by a time-zone entity canonical name:
/// `"GMT"`, `"GMT+9"`, `"GMT-3"`, `"GMT+5:30"` (also with a `UTC` prefix).
pub fn parse_gmt_offset(name: &str) -> Option<f64> {
  let rest = name
    .strip_prefix("GMT")
    .or_else(|| name.strip_prefix("UTC"))?;
  if rest.is_empty() {
    return Some(0.0);
  }
  let (sign, digits) = match rest.as_bytes()[0] {
    b'+' => (1.0, &rest[1..]),
    b'-' => (-1.0, &rest[1..]),
    _ => return None,
  };
  let (hours, minutes) = match digits.split_once(':') {
    Some((h, m)) => (h, m),
    None => (digits, "0"),
  };
  if hours.is_empty()
    || !hours.bytes().all(|b| b.is_ascii_digit())
    || !minutes.bytes().all(|b| b.is_ascii_digit())
  {
    return None;
  }
  let h: f64 = hours.parse().ok()?;
  let m: f64 = minutes.parse().ok()?;
  if h > 14.0 || m >= 60.0 {
    return None;
  }
  Some(sign * (h + m / 60.0))
}

/// Standard UTC offsets of countries that keep a single time throughout,
/// keyed by their Natural Earth name.
const SINGLE_ZONE: &[(&str, f64)] = &[
  // Europe
  ("Albania", 1.0),
  ("Austria", 1.0),
  ("Belarus", 3.0),
  ("Belgium", 1.0),
  ("Bosnia and Herz.", 1.0),
  ("Bulgaria", 2.0),
  ("Croatia", 1.0),
  ("Cyprus", 2.0),
  ("Czechia", 1.0),
  ("Denmark", 1.0),
  ("Estonia", 2.0),
  ("Finland", 2.0),
  ("Germany", 1.0),
  ("Greece", 2.0),
  ("Hungary", 1.0),
  ("Iceland", 0.0),
  ("Ireland", 0.0),
  ("Italy", 1.0),
  ("Kosovo", 1.0),
  ("Latvia", 2.0),
  ("Lithuania", 2.0),
  ("Luxembourg", 1.0),
  ("Moldova", 2.0),
  ("Montenegro", 1.0),
  ("N. Cyprus", 2.0),
  ("Netherlands", 1.0),
  ("North Macedonia", 1.0),
  ("Norway", 1.0),
  ("Poland", 1.0),
  ("Portugal", 0.0),
  ("Romania", 2.0),
  ("Serbia", 1.0),
  ("Slovakia", 1.0),
  ("Slovenia", 1.0),
  ("Sweden", 1.0),
  ("Switzerland", 1.0),
  ("Turkey", 3.0),
  ("Ukraine", 2.0),
  ("United Kingdom", 0.0),
  // Africa
  ("Algeria", 1.0),
  ("Angola", 1.0),
  ("Benin", 1.0),
  ("Botswana", 2.0),
  ("Burkina Faso", 0.0),
  ("Burundi", 2.0),
  ("Cameroon", 1.0),
  ("Central African Rep.", 1.0),
  ("Chad", 1.0),
  ("Congo", 1.0),
  ("Côte d'Ivoire", 0.0),
  ("Djibouti", 3.0),
  ("Egypt", 2.0),
  ("Eq. Guinea", 1.0),
  ("Eritrea", 3.0),
  ("eSwatini", 2.0),
  ("Ethiopia", 3.0),
  ("Gabon", 1.0),
  ("Gambia", 0.0),
  ("Ghana", 0.0),
  ("Guinea", 0.0),
  ("Guinea-Bissau", 0.0),
  ("Kenya", 3.0),
  ("Lesotho", 2.0),
  ("Liberia", 0.0),
  ("Libya", 2.0),
  ("Madagascar", 3.0),
  ("Malawi", 2.0),
  ("Mali", 0.0),
  ("Mauritania", 0.0),
  ("Morocco", 1.0),
  ("Mozambique", 2.0),
  ("Namibia", 2.0),
  ("Niger", 1.0),
  ("Nigeria", 1.0),
  ("Rwanda", 2.0),
  ("S. Sudan", 2.0),
  ("Senegal", 0.0),
  ("Sierra Leone", 0.0),
  ("Somalia", 3.0),
  ("Somaliland", 3.0),
  ("South Africa", 2.0),
  ("Sudan", 2.0),
  ("Tanzania", 3.0),
  ("Togo", 0.0),
  ("Tunisia", 1.0),
  ("Uganda", 3.0),
  ("W. Sahara", 1.0),
  ("Zambia", 2.0),
  ("Zimbabwe", 2.0),
  // Asia
  ("Afghanistan", 4.5),
  ("Armenia", 4.0),
  ("Azerbaijan", 4.0),
  ("Bangladesh", 6.0),
  ("Bhutan", 6.0),
  ("Brunei", 8.0),
  ("Cambodia", 7.0),
  ("China", 8.0),
  ("Georgia", 4.0),
  ("India", 5.5),
  ("Iran", 3.5),
  ("Iraq", 3.0),
  ("Israel", 2.0),
  ("Japan", 9.0),
  ("Jordan", 3.0),
  ("Kazakhstan", 5.0),
  ("Kuwait", 3.0),
  ("Kyrgyzstan", 6.0),
  ("Laos", 7.0),
  ("Lebanon", 2.0),
  ("Malaysia", 8.0),
  ("Myanmar", 6.5),
  ("Nepal", 5.75),
  ("North Korea", 9.0),
  ("Oman", 4.0),
  ("Pakistan", 5.0),
  ("Palestine", 2.0),
  ("Philippines", 8.0),
  ("Qatar", 3.0),
  ("Saudi Arabia", 3.0),
  ("South Korea", 9.0),
  ("Sri Lanka", 5.5),
  ("Syria", 3.0),
  ("Taiwan", 8.0),
  ("Tajikistan", 5.0),
  ("Thailand", 7.0),
  ("Timor-Leste", 9.0),
  ("Turkmenistan", 5.0),
  ("United Arab Emirates", 4.0),
  ("Uzbekistan", 5.0),
  ("Vietnam", 7.0),
  ("Yemen", 3.0),
  // Americas
  ("Argentina", -3.0),
  ("Bahamas", -5.0),
  ("Belize", -6.0),
  ("Bolivia", -4.0),
  ("Colombia", -5.0),
  ("Costa Rica", -6.0),
  ("Cuba", -5.0),
  ("Dominican Rep.", -4.0),
  ("El Salvador", -6.0),
  ("Falkland Is.", -3.0),
  ("Greenland", -2.0),
  ("Guatemala", -6.0),
  ("Guyana", -4.0),
  ("Haiti", -5.0),
  ("Honduras", -6.0),
  ("Jamaica", -5.0),
  ("Nicaragua", -6.0),
  ("Panama", -5.0),
  ("Paraguay", -3.0),
  ("Peru", -5.0),
  ("Puerto Rico", -4.0),
  ("Suriname", -3.0),
  ("Trinidad and Tobago", -4.0),
  ("Uruguay", -3.0),
  ("Venezuela", -4.0),
  // Oceania and the southern ocean
  ("Fiji", 12.0),
  ("Fr. S. Antarctic Lands", 5.0),
  ("New Caledonia", 11.0),
  ("New Zealand", 12.0),
  ("Papua New Guinea", 10.0),
  ("Solomon Is.", 11.0),
  ("Vanuatu", 11.0),
];

/// A disjoint `(lon_min, lon_max, lat_min, lat_max, offset)` box of a
/// country that spans several zones.
type ZoneBox = (f64, f64, f64, f64, f64);

/// Approximate zone boxes of countries spanning several time zones.
const MULTI_ZONE: &[(&str, &[ZoneBox])] = &[
  (
    "United States of America",
    &[
      (-180.0, -169.0, 30.0, 90.0, -10.0),
      (-180.0, -150.0, -90.0, 30.0, -10.0),
      (-169.0, -129.0, 50.0, 90.0, -9.0),
      (-129.0, -114.5, 24.0, 50.0, -8.0),
      (-114.5, -102.0, 24.0, 50.0, -7.0),
      (-102.0, -87.5, 24.0, 50.0, -6.0),
      (-87.5, -60.0, 24.0, 50.0, -5.0),
    ],
  ),
  (
    "Canada",
    &[
      (-141.0, -120.0, 40.0, 60.0, -8.0),
      (-141.0, -102.0, 60.0, 90.0, -7.0),
      (-120.0, -110.0, 40.0, 60.0, -7.0),
      (-110.0, -90.0, 40.0, 60.0, -6.0),
      (-102.0, -90.0, 60.0, 90.0, -6.0),
      (-90.0, -69.0, 40.0, 90.0, -5.0),
      (-69.0, -64.0, 48.1, 90.0, -5.0),
      (-69.0, -63.5, 40.0, 48.1, -4.0),
      (-63.5, -59.5, 40.0, 50.0, -4.0),
      (-64.0, -55.5, 52.0, 61.0, -4.0),
      (-59.5, -52.0, 46.0, 52.0, -3.5),
    ],
  ),
  (
    "Mexico",
    &[
      (-118.0, -114.8, 28.0, 33.0, -8.0),
      (-114.8, -108.5, 26.0, 33.0, -7.0),
      (-118.0, -114.8, 26.0, 28.0, -7.0),
      (-118.0, -105.5, 14.0, 26.0, -7.0),
      (-108.5, -89.3, 26.0, 33.0, -6.0),
      (-105.5, -89.3, 14.0, 26.0, -6.0),
      (-89.3, -86.0, 14.0, 22.0, -5.0),
    ],
  ),
  (
    "Brazil",
    &[
      (-75.0, -66.6, -11.2, -7.0, -5.0),
      (-74.0, -58.5, -7.0, 6.0, -4.0),
      (-66.6, -50.5, -24.0, -7.0, -4.0),
      (-58.5, -30.0, -7.0, 6.0, -3.0),
      (-50.5, -30.0, -34.0, -7.0, -3.0),
      (-58.5, -50.5, -34.0, -24.0, -3.0),
    ],
  ),
  (
    "Chile",
    &[
      (-76.0, -66.0, -49.0, -17.0, -4.0),
      (-76.0, -66.0, -56.0, -49.0, -3.0),
    ],
  ),
  (
    "Ecuador",
    &[
      (-92.0, -88.0, -2.0, 2.0, -6.0),
      (-81.5, -75.0, -5.5, 2.0, -5.0),
    ],
  ),
  (
    "France",
    &[
      (-55.0, -51.0, 2.0, 6.0, -3.0),
      (-6.0, 10.0, 41.0, 52.0, 1.0),
    ],
  ),
  (
    "Spain",
    &[
      (-19.0, -13.0, 27.0, 30.0, 0.0),
      (-10.0, 5.0, 35.0, 44.0, 1.0),
    ],
  ),
  (
    "Dem. Rep. Congo",
    &[
      (12.0, 20.0, -14.0, 6.0, 1.0),
      (20.0, 24.5, -1.5, 6.0, 1.0),
      (20.0, 32.0, -14.0, -1.5, 2.0),
      (24.5, 32.0, -1.5, 6.0, 2.0),
    ],
  ),
  (
    "Russia",
    &[
      (19.0, 23.0, 53.0, 56.0, 2.0),
      (23.0, 48.0, 40.0, 90.0, 3.0),
      (48.0, 53.0, 56.0, 61.5, 3.0),
      (48.0, 60.0, 61.5, 90.0, 3.0),
      (48.0, 53.0, 40.0, 56.0, 4.0),
      (53.0, 60.0, 40.0, 61.5, 5.0),
      (60.0, 70.0, 40.0, 90.0, 5.0),
      (70.0, 85.0, 60.0, 90.0, 5.0),
      (70.0, 76.0, 40.0, 60.0, 6.0),
      (76.0, 97.0, 40.0, 60.0, 7.0),
      (85.0, 108.0, 60.0, 90.0, 7.0),
      (97.0, 108.0, 40.0, 60.0, 8.0),
      (108.0, 130.0, 40.0, 90.0, 9.0),
      (130.0, 135.0, 55.0, 90.0, 9.0),
      (130.0, 141.6, 40.0, 55.0, 10.0),
      (135.0, 147.0, 55.0, 90.0, 10.0),
      (141.6, 147.0, 40.0, 55.0, 11.0),
      (147.0, 155.0, 40.0, 90.0, 11.0),
      (155.0, 162.0, 59.0, 90.0, 11.0),
      (155.0, 162.0, 40.0, 59.0, 12.0),
      (162.0, 180.0, 40.0, 90.0, 12.0),
      (-180.0, -160.0, 40.0, 90.0, 12.0),
    ],
  ),
  (
    "Mongolia",
    &[
      (87.0, 95.5, 40.0, 53.0, 7.0),
      (95.5, 120.0, 40.0, 53.0, 8.0),
    ],
  ),
  (
    "Indonesia",
    &[
      (95.0, 114.5, -11.0, 6.0, 7.0),
      (114.5, 126.0, -11.0, 6.0, 8.0),
      (126.0, 141.1, -11.0, 6.0, 9.0),
    ],
  ),
  (
    "Australia",
    &[
      (110.0, 129.0, -45.0, -10.0, 8.0),
      (129.0, 138.0, -26.0, -10.0, 9.5),
      (129.0, 141.0, -40.0, -26.0, 9.5),
      (138.0, 154.0, -26.0, -9.0, 10.0),
      (141.0, 154.0, -45.0, -26.0, 10.0),
    ],
  ),
];

/// The `(lon_min, lon_max, lat_min, lat_max)` boxes of a Natural Earth
/// country that keep standard time `offset` (empty when none does). A
/// single-zone country is one box covering the whole globe.
fn country_boxes(name: &str, offset: f64) -> Vec<(f64, f64, f64, f64)> {
  let same = |o: f64| (o - offset).abs() < 1e-9;
  if let Some(&(_, o)) = SINGLE_ZONE.iter().find(|(n, _)| *n == name) {
    return if same(o) {
      vec![(-180.0, 180.0, -90.0, 90.0)]
    } else {
      Vec::new()
    };
  }
  MULTI_ZONE
    .iter()
    .find(|(n, _)| *n == name)
    .map_or_default(|(_, boxes)| {
      boxes
        .iter()
        .filter(|b| same(b.4))
        .map(|&(lo0, lo1, la0, la1, _)| (lo0, lo1, la0, la1))
        .collect()
    })
}

/// Every country name the zone tables use (they must all be Natural Earth
/// names, which the geographics tests check).
#[cfg(test)]
pub fn table_country_names() -> impl Iterator<Item = &'static str> {
  SINGLE_ZONE
    .iter()
    .map(|(n, _)| *n)
    .chain(MULTI_ZONE.iter().map(|(n, _)| *n))
}

/// The open-ocean longitude band `(lon_min, lon_max)` of the nautical zone
/// with this offset, or `None` for offsets no nautical zone has.
pub fn nautical_band(offset: f64) -> Option<(f64, f64)> {
  if offset.fract() != 0.0 || offset.abs() > 12.0 {
    return None;
  }
  let center = 15.0 * offset;
  Some(((center - 7.5).max(-180.0), (center + 7.5).min(180.0)))
}

/// Clip a ring to an axis-aligned box (Sutherland–Hodgman; the box is
/// convex, so any ring clips to a single ring, possibly degenerate).
pub fn clip_ring(
  ring: &[(f64, f64)],
  (lon0, lon1, lat0, lat1): (f64, f64, f64, f64),
) -> Ring {
  // Each edge: inside test and intersection with the boundary line.
  type Inside = fn(&(f64, f64), f64) -> bool;
  let edges: [(Inside, f64, bool); 4] = [
    (|p, v| p.0 >= v, lon0, true),
    (|p, v| p.0 <= v, lon1, true),
    (|p, v| p.1 >= v, lat0, false),
    (|p, v| p.1 <= v, lat1, false),
  ];
  let mut out: Ring = ring.to_vec();
  for (inside, v, vertical) in edges {
    if out.is_empty() {
      break;
    }
    let input = std::mem::take(&mut out);
    let cross = |a: &(f64, f64), b: &(f64, f64)| {
      if vertical {
        let t = (v - a.0) / (b.0 - a.0);
        (v, a.1 + t * (b.1 - a.1))
      } else {
        let t = (v - a.1) / (b.1 - a.1);
        (a.0 + t * (b.0 - a.0), v)
      }
    };
    for (i, cur) in input.iter().enumerate() {
      let prev = &input[(i + input.len() - 1) % input.len()];
      match (inside(cur, v), inside(prev, v)) {
        (true, true) => out.push(*cur),
        (true, false) => {
          out.push(cross(prev, cur));
          out.push(*cur);
        }
        (false, true) => out.push(cross(prev, cur)),
        (false, false) => {}
      }
    }
  }
  if out.len() < 3 { Vec::new() } else { out }
}

/// The rings (filled even-odd) of the region keeping standard time `offset`,
/// given every country as `(Natural Earth name, polygons)` where a polygon's
/// first ring is its exterior.
pub fn zone_rings<'a>(
  offset: f64,
  countries: impl Iterator<Item = (&'a str, &'a [Vec<Vec<(f32, f32)>>])>,
) -> Vec<Ring> {
  let band = nautical_band(offset).map(|(lo0, lo1)| (lo0, lo1, -90.0, 90.0));
  let mut band_rings: Vec<Ring> = Vec::new();
  let mut land_rings: Vec<Ring> = Vec::new();
  if let Some((lo0, lo1, la0, la1)) = band {
    band_rings.push(vec![(lo0, la0), (lo1, la0), (lo1, la1), (lo0, la1)]);
  }
  for (name, polys) in countries {
    let as_f64 = |r: &Vec<(f32, f32)>| -> Ring {
      r.iter().map(|&(x, y)| (x as f64, y as f64)).collect()
    };
    // Cut the country's land out of the ocean band (Antarctica, which has
    // no civil time, stays part of it).
    if let Some(b) = band
      && name != "Antarctica"
    {
      for exterior in polys.iter().filter_map(|p| p.first()) {
        let clipped = clip_ring(&as_f64(exterior), b);
        if !clipped.is_empty() {
          band_rings.push(clipped);
        }
      }
    }
    for b in country_boxes(name, offset) {
      for ring in polys.iter().flatten() {
        let clipped = clip_ring(&as_f64(ring), b);
        if !clipped.is_empty() {
          land_rings.push(clipped);
        }
      }
    }
  }
  band_rings.extend(land_rings);
  band_rings
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn parses_gmt_offsets() {
    assert_eq!(parse_gmt_offset("GMT"), Some(0.0));
    assert_eq!(parse_gmt_offset("GMT+9"), Some(9.0));
    assert_eq!(parse_gmt_offset("GMT-3"), Some(-3.0));
    assert_eq!(parse_gmt_offset("GMT+5:30"), Some(5.5));
    assert_eq!(parse_gmt_offset("UTC-10"), Some(-10.0));
    assert_eq!(parse_gmt_offset("GMT+"), None);
    assert_eq!(parse_gmt_offset("Asia/Tokyo"), None);
    assert_eq!(parse_gmt_offset("GMT+99"), None);
  }

  #[test]
  fn nautical_bands_are_15_degrees_wide() {
    assert_eq!(nautical_band(0.0), Some((-7.5, 7.5)));
    assert_eq!(nautical_band(9.0), Some((127.5, 142.5)));
    assert_eq!(nautical_band(-5.0), Some((-82.5, -67.5)));
    assert_eq!(nautical_band(12.0), Some((172.5, 180.0)));
    assert_eq!(nautical_band(-12.0), Some((-180.0, -172.5)));
    assert_eq!(nautical_band(5.5), None);
  }

  #[test]
  fn clips_rings_to_a_box() {
    let square = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)];
    let clipped = clip_ring(&square, (5.0, 20.0, -5.0, 20.0));
    let xs: Vec<f64> = clipped.iter().map(|p| p.0).collect();
    assert_eq!(clipped.len(), 4);
    assert!(xs.iter().all(|&x| (5.0..=10.0).contains(&x)));
    assert!(clip_ring(&square, (20.0, 30.0, 0.0, 10.0)).is_empty());
  }

  #[test]
  fn country_boxes_follow_standard_time() {
    assert_eq!(country_boxes("Japan", 9.0).len(), 1);
    assert!(country_boxes("Japan", 8.0).is_empty());
    // Only the eastern provinces of Indonesia keep GMT+9.
    assert_eq!(
      country_boxes("Indonesia", 9.0),
      vec![(126.0, 141.1, -11.0, 6.0)]
    );
    // Half-hour zones never match a whole-hour one.
    assert!(country_boxes("India", 5.0).is_empty());
    assert_eq!(country_boxes("India", 5.5).len(), 1);
  }

  #[test]
  fn multi_zone_boxes_are_disjoint() {
    for (name, boxes) in MULTI_ZONE {
      for (i, a) in boxes.iter().enumerate() {
        for b in &boxes[i + 1..] {
          let overlap = a.0 < b.1 && b.0 < a.1 && a.2 < b.3 && b.2 < a.3;
          assert!(!overlap, "{name}: {a:?} overlaps {b:?}");
        }
      }
    }
  }
}
