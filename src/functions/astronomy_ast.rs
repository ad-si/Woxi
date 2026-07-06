//! Astronomical computation functions (MoonPhase, SunPosition, Sunrise, …).
//!
//! All ephemerides are computed from the algorithms in Jean Meeus,
//! "Astronomical Algorithms" (2nd edition): solar coordinates (ch. 25),
//! lunar coordinates (ch. 47, truncated ELP-2000/82), nutation and
//! obliquity (ch. 22), sidereal time (ch. 12), rise/set times (ch. 15),
//! illuminated fraction (ch. 48), lunar phases (ch. 49), and eclipses
//! (ch. 54). Accuracy is on the order of arcseconds for the Sun, ~10″
//! for the Moon, and better than a minute for phase and eclipse times —
//! wolframscript uses full ephemerides, so the last displayed digit can
//! differ.
//!
//! Following Woxi's datetime convention, all date arguments are read as
//! UTC instants and all returned DateObjects carry TimeZone 0 (matching
//! `$TimeZone`), whereas wolframscript localizes to the geo location's
//! time zone.

use crate::InterpreterError;
use crate::functions::datetime_ast::{
  absolute_seconds_to_date, date_to_absolute_seconds, extract_date_components,
  resolve_date_to_list,
};
use crate::syntax::Expr;

// ─── Angle helpers ──────────────────────────────────────────────────

const DEG: f64 = std::f64::consts::PI / 180.0;

fn sin_d(x: f64) -> f64 {
  (x * DEG).sin()
}
fn cos_d(x: f64) -> f64 {
  (x * DEG).cos()
}

/// Reduce an angle in degrees to [0, 360).
fn norm360(x: f64) -> f64 {
  x.rem_euclid(360.0)
}

// ─── Time scales ────────────────────────────────────────────────────

/// JD of the Woxi absolute-time epoch 1900-01-01 00:00 UTC.
const JD_1900: f64 = 2415020.5;

/// Absolute seconds (Woxi's AbsoluteTime, seconds since 1900) → Julian date.
fn abs_seconds_to_jd(abs: f64) -> f64 {
  JD_1900 + abs / 86400.0
}

fn jd_to_abs_seconds(jd: f64) -> f64 {
  (jd - JD_1900) * 86400.0
}

/// Current instant as a Julian date (UTC).
fn now_jd() -> f64 {
  use web_time::{SystemTime, UNIX_EPOCH};
  let unix = SystemTime::now()
    .duration_since(UNIX_EPOCH)
    .map(|d| d.as_secs_f64())
    .unwrap_or(0.0);
  2440587.5 + unix / 86400.0
}

/// ΔT = TT − UT1 in seconds, from the piecewise polynomial fits of
/// Espenak & Meeus (2006) used by the NASA eclipse pages. `year` is a
/// decimal year.
fn delta_t_seconds(year: f64) -> f64 {
  let y = year;
  if y < -500.0 {
    let u = (y - 1820.0) / 100.0;
    -20.0 + 32.0 * u * u
  } else if y < 500.0 {
    let u = y / 100.0;
    10583.6
      + u
        * (-1014.41
          + u
            * (33.78311
              + u
                * (-5.952053
                  + u * (-0.1798452 + u * (0.022174192 + u * 0.0090316521)))))
  } else if y < 1600.0 {
    let u = (y - 1000.0) / 100.0;
    1574.2
      + u
        * (-556.01
          + u
            * (71.23472
              + u
                * (0.319781
                  + u * (-0.8503463 + u * (-0.005050998 + u * 0.0083572073)))))
  } else if y < 1700.0 {
    let t = y - 1600.0;
    120.0 + t * (-0.9808 + t * (-0.01532 + t / 7129.0))
  } else if y < 1800.0 {
    let t = y - 1700.0;
    8.83 + t * (0.1603 + t * (-0.0059285 + t * (0.00013336 - t / 1174000.0)))
  } else if y < 1860.0 {
    let t = y - 1800.0;
    13.72
      + t
        * (-0.332447
          + t
            * (0.0068612
              + t
                * (0.0041116
                  + t
                    * (-0.00037436
                      + t
                        * (0.0000121272
                          + t * (-0.0000001699 + t * 0.000000000875))))))
  } else if y < 1900.0 {
    let t = y - 1860.0;
    7.62
      + t
        * (0.5737
          + t
            * (-0.251754
              + t * (0.01680668 + t * (-0.0004473624 + t / 233174.0))))
  } else if y < 1920.0 {
    let t = y - 1900.0;
    -2.79 + t * (1.494119 + t * (-0.0598939 + t * (0.0061966 - t * 0.000197)))
  } else if y < 1941.0 {
    let t = y - 1920.0;
    21.20 + t * (0.84493 + t * (-0.076100 + t * 0.0020936))
  } else if y < 1961.0 {
    let t = y - 1950.0;
    29.07 + t * (0.407 + t * (-1.0 / 233.0 + t / 2547.0))
  } else if y < 1986.0 {
    let t = y - 1975.0;
    45.45 + t * (1.067 + t * (-1.0 / 260.0 - t / 718.0))
  } else if y < 2005.0 {
    let t = y - 2000.0;
    63.86
      + t
        * (0.3345
          + t
            * (-0.060374
              + t * (0.0017275 + t * (0.000651814 + t * 0.00002373599))))
  } else if y < 2050.0 {
    let t = y - 2000.0;
    62.92 + t * (0.32217 + t * 0.005589)
  } else if y < 2150.0 {
    let u = (y - 1820.0) / 100.0;
    -20.0 + 32.0 * u * u - 0.5628 * (2150.0 - y)
  } else {
    let u = (y - 1820.0) / 100.0;
    -20.0 + 32.0 * u * u
  }
}

/// Decimal year of a Julian date.
fn jd_to_decimal_year(jd: f64) -> f64 {
  2000.0 + (jd - 2451544.5) / 365.2425
}

/// UTC Julian date → Julian ephemeris date (TT).
fn jd_utc_to_jde(jd: f64) -> f64 {
  jd + delta_t_seconds(jd_to_decimal_year(jd)) / 86400.0
}

/// Julian ephemeris date (TT) → UTC Julian date.
fn jde_to_jd_utc(jde: f64) -> f64 {
  jde - delta_t_seconds(jd_to_decimal_year(jde)) / 86400.0
}

// ─── Nutation and obliquity (Meeus ch. 22) ──────────────────────────

/// Nutation in longitude and obliquity (degrees), main terms (~0.5″).
fn nutation(t: f64) -> (f64, f64) {
  // Longitude of the ascending node of the Moon's mean orbit
  let omega = 125.04452 - 1934.136261 * t + 0.0020708 * t * t;
  // Mean longitudes of Sun and Moon
  let l_sun = 280.4665 + 36000.7698 * t;
  let l_moon = 218.3165 + 481267.8813 * t;
  let dpsi = (-17.20 * sin_d(omega)
    - 1.32 * sin_d(2.0 * l_sun)
    - 0.23 * sin_d(2.0 * l_moon)
    + 0.21 * sin_d(2.0 * omega))
    / 3600.0;
  let deps = (9.20 * cos_d(omega)
    + 0.57 * cos_d(2.0 * l_sun)
    + 0.10 * cos_d(2.0 * l_moon)
    - 0.09 * cos_d(2.0 * omega))
    / 3600.0;
  (dpsi, deps)
}

/// Mean obliquity of the ecliptic in degrees (Meeus 22.2).
fn mean_obliquity(t: f64) -> f64 {
  23.0 + 26.0 / 60.0 + 21.448 / 3600.0
    - (46.8150 * t + 0.00059 * t * t - 0.001813 * t * t * t) / 3600.0
}

// ─── Solar coordinates (Meeus ch. 25) ───────────────────────────────

/// Geometric mean and apparent solar longitude plus distance.
/// Returns (apparent λ in degrees, true geometric λ, distance in AU)
/// for a Julian ephemeris date.
fn sun_coordinates(jde: f64) -> (f64, f64, f64) {
  let t = (jde - 2451545.0) / 36525.0;
  let l0 = 280.46646 + 36000.76983 * t + 0.0003032 * t * t;
  let m = 357.52911 + 35999.05029 * t - 0.0001537 * t * t;
  let e = 0.016708634 - 0.000042037 * t - 0.0000001267 * t * t;
  let c = (1.914602 - 0.004817 * t - 0.000014 * t * t) * sin_d(m)
    + (0.019993 - 0.000101 * t) * sin_d(2.0 * m)
    + 0.000289 * sin_d(3.0 * m);
  let true_lon = l0 + c;
  let nu = m + c;
  let r = 1.000001018 * (1.0 - e * e) / (1.0 + e * cos_d(nu));
  let omega = 125.04 - 1934.136 * t;
  let apparent = true_lon - 0.00569 - 0.00478 * sin_d(omega);
  (norm360(apparent), norm360(true_lon), r)
}

/// Apparent right ascension and declination of the Sun in degrees.
fn sun_ra_dec(jde: f64) -> (f64, f64) {
  let t = (jde - 2451545.0) / 36525.0;
  let (lambda, _, _) = sun_coordinates(jde);
  let (_, deps) = nutation(t);
  // Apparent positions use the true obliquity; the 0.00256 cos Ω term of
  // Meeus 25.8 is folded into deps well enough at this accuracy.
  let eps = mean_obliquity(t) + deps;
  let ra = norm360(f64::atan2(cos_d(eps) * sin_d(lambda), cos_d(lambda)) / DEG);
  let dec = (sin_d(eps) * sin_d(lambda)).asin() / DEG;
  (ra, dec)
}

// ─── Lunar coordinates (Meeus ch. 47) ───────────────────────────────

/// Periodic terms for the Moon's longitude (Σl, 1e-6 deg) and distance
/// (Σr, 1e-3 km): multiples of (D, M, M′, F) with both coefficients.
#[rustfmt::skip]
const MOON_LR: [(i8, i8, i8, i8, i64, i64); 60] = [
  (0, 0, 1, 0, 6288774, -20905355),
  (2, 0, -1, 0, 1274027, -3699111),
  (2, 0, 0, 0, 658314, -2955968),
  (0, 0, 2, 0, 213618, -569925),
  (0, 1, 0, 0, -185116, 48888),
  (0, 0, 0, 2, -114332, -3149),
  (2, 0, -2, 0, 58793, 246158),
  (2, -1, -1, 0, 57066, -152138),
  (2, 0, 1, 0, 53322, -170733),
  (2, -1, 0, 0, 45758, -204586),
  (0, 1, -1, 0, -40923, -129620),
  (1, 0, 0, 0, -34720, 108743),
  (0, 1, 1, 0, -30383, 104755),
  (2, 0, 0, -2, 15327, 10321),
  (0, 0, 1, 2, -12528, 0),
  (0, 0, 1, -2, 10980, 79661),
  (4, 0, -1, 0, 10675, -34782),
  (0, 0, 3, 0, 10034, -23210),
  (4, 0, -2, 0, 8548, -21636),
  (2, 1, -1, 0, -7888, 24208),
  (2, 1, 0, 0, -6766, 30824),
  (1, 0, -1, 0, -5163, -8379),
  (1, 1, 0, 0, 4987, -16675),
  (2, -1, 1, 0, 4036, -12831),
  (2, 0, 2, 0, 3994, -10445),
  (4, 0, 0, 0, 3861, -11650),
  (2, 0, -3, 0, 3665, 14403),
  (0, 1, -2, 0, -2689, -7003),
  (2, 0, -1, 2, -2602, 0),
  (2, -1, -2, 0, 2390, 10056),
  (1, 0, 1, 0, -2348, 6322),
  (2, -2, 0, 0, 2236, -9884),
  (0, 1, 2, 0, -2120, 5751),
  (0, 2, 0, 0, -2069, 0),
  (2, -2, -1, 0, 2048, -4950),
  (2, 0, 1, -2, -1773, 4130),
  (2, 0, 0, 2, -1595, 0),
  (4, -1, -1, 0, 1215, -3958),
  (0, 0, 2, 2, -1110, 0),
  (3, 0, -1, 0, -892, 3258),
  (2, 1, 1, 0, -810, 2616),
  (4, -1, -2, 0, 759, -1897),
  (0, 2, -1, 0, -713, -2117),
  (2, 2, -1, 0, -700, 2354),
  (2, 1, -2, 0, 691, 0),
  (2, -1, 0, -2, 596, 0),
  (4, 0, 1, 0, 549, -1423),
  (0, 0, 4, 0, 537, -1117),
  (4, -1, 0, 0, 520, -1571),
  (1, 0, -2, 0, -487, -1739),
  (2, 1, 0, -2, -399, 0),
  (0, 0, 2, -2, -381, -4421),
  (1, 1, 1, 0, 351, 0),
  (3, 0, -2, 0, -340, 0),
  (4, 0, -3, 0, 330, 0),
  (2, -1, 2, 0, 327, 0),
  (0, 2, 1, 0, -323, 1165),
  (1, 1, -1, 0, 299, 0),
  (2, 0, 3, 0, 294, 0),
  (2, 0, -1, -2, 0, 8752),
];

/// Periodic terms for the Moon's latitude (Σb, 1e-6 deg).
#[rustfmt::skip]
const MOON_B: [(i8, i8, i8, i8, i64); 60] = [
  (0, 0, 0, 1, 5128122),
  (0, 0, 1, 1, 280602),
  (0, 0, 1, -1, 277693),
  (2, 0, 0, -1, 173237),
  (2, 0, -1, 1, 55413),
  (2, 0, -1, -1, 46271),
  (2, 0, 0, 1, 32573),
  (0, 0, 2, 1, 17198),
  (2, 0, 1, -1, 9266),
  (0, 0, 2, -1, 8822),
  (2, -1, 0, -1, 8216),
  (2, 0, -2, -1, 4324),
  (2, 0, 1, 1, 4200),
  (2, 1, 0, -1, -3359),
  (2, -1, -1, 1, 2463),
  (2, -1, 0, 1, 2211),
  (2, -1, -1, -1, 2065),
  (0, 1, -1, -1, -1870),
  (4, 0, -1, -1, 1828),
  (0, 1, 0, 1, -1794),
  (0, 0, 0, 3, -1749),
  (0, 1, -1, 1, -1565),
  (1, 0, 0, 1, -1491),
  (0, 1, 1, 1, -1475),
  (0, 1, 1, -1, -1410),
  (0, 1, 0, -1, -1344),
  (1, 0, 0, -1, -1335),
  (0, 0, 3, 1, 1107),
  (4, 0, 0, -1, 1021),
  (4, 0, -1, 1, 833),
  (0, 0, 1, -3, 777),
  (4, 0, -2, 1, 671),
  (2, 0, 0, -3, 607),
  (2, 0, 2, -1, 596),
  (2, -1, 1, -1, 491),
  (2, 0, -2, 1, -451),
  (0, 0, 3, -1, 439),
  (2, 0, 2, 1, 422),
  (2, 0, -3, -1, 421),
  (2, 1, -1, 1, -366),
  (2, 1, 0, 1, -351),
  (4, 0, 0, 1, 331),
  (2, -1, 1, 1, 315),
  (2, -2, 0, -1, 302),
  (0, 0, 1, 3, -283),
  (2, 1, 1, -1, -229),
  (1, 1, 0, -1, 223),
  (1, 1, 0, 1, 223),
  (0, 1, -2, -1, -220),
  (2, 1, -1, -1, -220),
  (1, 0, 1, 1, -185),
  (2, -1, -2, -1, 181),
  (0, 1, 2, 1, -177),
  (4, 0, -2, -1, 176),
  (4, -1, -1, -1, 166),
  (1, 0, 1, -1, -164),
  (4, 0, 1, -1, 132),
  (1, 0, -1, -1, -119),
  (4, -1, 0, -1, 115),
  (2, -2, 0, 1, 107),
];

/// Geocentric ecliptic longitude, latitude (degrees, apparent — nutation
/// included in the longitude) and distance (km) of the Moon.
fn moon_coordinates(jde: f64) -> (f64, f64, f64) {
  let t = (jde - 2451545.0) / 36525.0;
  let t2 = t * t;
  let t3 = t2 * t;
  let t4 = t3 * t;

  // Mean longitude, elongation, anomalies, argument of latitude
  let lp = norm360(
    218.3164477 + 481267.88123421 * t - 0.0015786 * t2 + t3 / 538841.0
      - t4 / 65194000.0,
  );
  let d = norm360(
    297.8501921 + 445267.1114034 * t - 0.0018819 * t2 + t3 / 545868.0
      - t4 / 113065000.0,
  );
  let m =
    norm360(357.5291092 + 35999.0502909 * t - 0.0001536 * t2 + t3 / 24490000.0);
  let mp = norm360(
    134.9633964 + 477198.8675055 * t + 0.0087414 * t2 + t3 / 69699.0
      - t4 / 14712000.0,
  );
  let f = norm360(
    93.2720950 + 483202.0175233 * t - 0.0036539 * t2 - t3 / 3526000.0
      + t4 / 863310000.0,
  );

  let a1 = norm360(119.75 + 131.849 * t);
  let a2 = norm360(53.09 + 479264.290 * t);
  let a3 = norm360(313.45 + 481266.484 * t);
  let e = 1.0 - 0.002516 * t - 0.0000074 * t2;

  let mut sum_l = 0.0;
  let mut sum_r = 0.0;
  for &(cd, cm, cmp, cf, sl, sr) in MOON_LR.iter() {
    let arg = cd as f64 * d + cm as f64 * m + cmp as f64 * mp + cf as f64 * f;
    let e_factor = match cm.abs() {
      1 => e,
      2 => e * e,
      _ => 1.0,
    };
    sum_l += sl as f64 * e_factor * sin_d(arg);
    sum_r += sr as f64 * e_factor * cos_d(arg);
  }
  sum_l += 3958.0 * sin_d(a1) + 1962.0 * sin_d(lp - f) + 318.0 * sin_d(a2);

  let mut sum_b = 0.0;
  for &(cd, cm, cmp, cf, sb) in MOON_B.iter() {
    let arg = cd as f64 * d + cm as f64 * m + cmp as f64 * mp + cf as f64 * f;
    let e_factor = match cm.abs() {
      1 => e,
      2 => e * e,
      _ => 1.0,
    };
    sum_b += sb as f64 * e_factor * sin_d(arg);
  }
  sum_b += -2235.0 * sin_d(lp)
    + 382.0 * sin_d(a3)
    + 175.0 * sin_d(a1 - f)
    + 175.0 * sin_d(a1 + f)
    + 127.0 * sin_d(lp - mp)
    - 115.0 * sin_d(lp + mp);

  let (dpsi, _) = nutation(t);
  let lambda = norm360(lp + sum_l / 1_000_000.0 + dpsi);
  let beta = sum_b / 1_000_000.0;
  let delta = 385000.56 + sum_r / 1000.0;
  (lambda, beta, delta)
}

/// Apparent right ascension and declination of the Moon in degrees.
fn moon_ra_dec(jde: f64) -> (f64, f64) {
  let t = (jde - 2451545.0) / 36525.0;
  let (lambda, beta, _) = moon_coordinates(jde);
  let (_, deps) = nutation(t);
  let eps = mean_obliquity(t) + deps;
  ecliptic_to_equatorial(lambda, beta, eps)
}

/// Ecliptic (λ, β) → equatorial (α, δ), all in degrees.
fn ecliptic_to_equatorial(lambda: f64, beta: f64, eps: f64) -> (f64, f64) {
  let ra = f64::atan2(
    sin_d(lambda) * cos_d(eps) - (beta * DEG).tan() * sin_d(eps),
    cos_d(lambda),
  ) / DEG;
  let dec = (sin_d(beta) * cos_d(eps)
    + cos_d(beta) * sin_d(eps) * sin_d(lambda))
  .asin()
    / DEG;
  (norm360(ra), dec)
}

// ─── Sidereal time and horizontal coordinates ───────────────────────

/// Mean sidereal time at Greenwich in degrees (Meeus 12.4).
fn gmst_deg(jd_ut: f64) -> f64 {
  let t = (jd_ut - 2451545.0) / 36525.0;
  norm360(
    280.46061837 + 360.98564736629 * (jd_ut - 2451545.0) + 0.000387933 * t * t
      - t * t * t / 38710000.0,
  )
}

/// Apparent sidereal time at Greenwich in degrees (mean + equation of
/// the equinoxes).
fn apparent_gst_deg(jd_ut: f64) -> f64 {
  let t = (jd_ut - 2451545.0) / 36525.0;
  let (dpsi, deps) = nutation(t);
  let eps = mean_obliquity(t) + deps;
  norm360(gmst_deg(jd_ut) + dpsi * cos_d(eps))
}

/// Equatorial (α, δ) → horizontal (azimuth from North through East,
/// altitude), all in degrees, for an observer at (lat, lon) — east
/// longitudes positive — at UTC Julian date `jd_ut`.
fn equatorial_to_horizontal(
  ra: f64,
  dec: f64,
  lat: f64,
  lon: f64,
  jd_ut: f64,
) -> (f64, f64) {
  let lst = apparent_gst_deg(jd_ut) + lon;
  let h = lst - ra; // local hour angle
  let alt =
    (sin_d(lat) * sin_d(dec) + cos_d(lat) * cos_d(dec) * cos_d(h)).asin() / DEG;
  // Meeus measures azimuth from South; add 180° for the from-North
  // convention used by SunPosition.
  let az_south = f64::atan2(
    sin_d(h),
    cos_d(h) * sin_d(lat) - (dec * DEG).tan() * cos_d(lat),
  ) / DEG;
  (norm360(az_south + 180.0), alt)
}

// ─── Lunar phases (Meeus ch. 49) ────────────────────────────────────

/// Phase selector: 0 = new, 1 = first quarter, 2 = full, 3 = last quarter.
#[derive(Clone, Copy, PartialEq)]
pub enum Phase {
  New = 0,
  FirstQuarter = 1,
  Full = 2,
  LastQuarter = 3,
}

/// JDE (TT) of the lunar phase at series index k (k integer for new
/// moons; the phase adds its quarter offset internally).
fn phase_jde(k_int: f64, phase: Phase) -> f64 {
  let k = k_int + (phase as i64 as f64) * 0.25;
  let t = k / 1236.85;
  let t2 = t * t;
  let t3 = t2 * t;
  let t4 = t3 * t;

  let mut jde = 2451550.09766 + 29.530588861 * k + 0.00015437 * t2
    - 0.000000150 * t3
    + 0.00000000073 * t4;

  let e = 1.0 - 0.002516 * t - 0.0000074 * t2;
  let m = norm360(2.5534 + 29.10535670 * k - 0.0000014 * t2 - 0.00000011 * t3);
  let mp = norm360(
    201.5643 + 385.81693528 * k + 0.0107582 * t2 + 0.00001238 * t3
      - 0.000000058 * t4,
  );
  let f = norm360(
    160.7108 + 390.67050284 * k - 0.0016118 * t2 - 0.00000227 * t3
      + 0.000000011 * t4,
  );
  let om =
    norm360(124.7746 - 1.56375588 * k + 0.0020672 * t2 + 0.00000215 * t3);

  match phase {
    Phase::New => {
      jde += -0.40720 * sin_d(mp)
        + 0.17241 * e * sin_d(m)
        + 0.01608 * sin_d(2.0 * mp)
        + 0.01039 * sin_d(2.0 * f)
        + 0.00739 * e * sin_d(mp - m)
        - 0.00514 * e * sin_d(mp + m)
        + 0.00208 * e * e * sin_d(2.0 * m)
        - 0.00111 * sin_d(mp - 2.0 * f)
        - 0.00057 * sin_d(mp + 2.0 * f)
        + 0.00056 * e * sin_d(2.0 * mp + m)
        - 0.00042 * sin_d(3.0 * mp)
        + 0.00042 * e * sin_d(m + 2.0 * f)
        + 0.00038 * e * sin_d(m - 2.0 * f)
        - 0.00024 * e * sin_d(2.0 * mp - m)
        - 0.00017 * sin_d(om)
        - 0.00007 * sin_d(mp + 2.0 * m)
        + 0.00004 * sin_d(2.0 * mp - 2.0 * f)
        + 0.00004 * sin_d(3.0 * m)
        + 0.00003 * sin_d(mp + m - 2.0 * f)
        + 0.00003 * sin_d(2.0 * mp + 2.0 * f)
        - 0.00003 * sin_d(mp + m + 2.0 * f)
        + 0.00003 * sin_d(mp - m + 2.0 * f)
        - 0.00002 * sin_d(mp - m - 2.0 * f)
        - 0.00002 * sin_d(3.0 * mp + m)
        + 0.00002 * sin_d(4.0 * mp);
    }
    Phase::Full => {
      jde += -0.40614 * sin_d(mp)
        + 0.17302 * e * sin_d(m)
        + 0.01614 * sin_d(2.0 * mp)
        + 0.01043 * sin_d(2.0 * f)
        + 0.00734 * e * sin_d(mp - m)
        - 0.00514 * e * sin_d(mp + m)
        + 0.00209 * e * e * sin_d(2.0 * m)
        - 0.00111 * sin_d(mp - 2.0 * f)
        - 0.00057 * sin_d(mp + 2.0 * f)
        + 0.00056 * e * sin_d(2.0 * mp + m)
        - 0.00042 * sin_d(3.0 * mp)
        + 0.00042 * e * sin_d(m + 2.0 * f)
        + 0.00038 * e * sin_d(m - 2.0 * f)
        - 0.00024 * e * sin_d(2.0 * mp - m)
        - 0.00017 * sin_d(om)
        - 0.00007 * sin_d(mp + 2.0 * m)
        + 0.00004 * sin_d(2.0 * mp - 2.0 * f)
        + 0.00004 * sin_d(3.0 * m)
        + 0.00003 * sin_d(mp + m - 2.0 * f)
        + 0.00003 * sin_d(2.0 * mp + 2.0 * f)
        - 0.00003 * sin_d(mp + m + 2.0 * f)
        + 0.00003 * sin_d(mp - m + 2.0 * f)
        - 0.00002 * sin_d(mp - m - 2.0 * f)
        - 0.00002 * sin_d(3.0 * mp + m)
        + 0.00002 * sin_d(4.0 * mp);
    }
    Phase::FirstQuarter | Phase::LastQuarter => {
      jde += -0.62801 * sin_d(mp) + 0.17172 * e * sin_d(m)
        - 0.01183 * e * sin_d(mp + m)
        + 0.00862 * sin_d(2.0 * mp)
        + 0.00804 * sin_d(2.0 * f)
        + 0.00454 * e * sin_d(mp - m)
        + 0.00204 * e * e * sin_d(2.0 * m)
        - 0.00180 * sin_d(mp - 2.0 * f)
        - 0.00070 * sin_d(mp + 2.0 * f)
        - 0.00040 * sin_d(3.0 * mp)
        - 0.00034 * e * sin_d(2.0 * mp - m)
        + 0.00032 * e * sin_d(m + 2.0 * f)
        + 0.00032 * e * sin_d(m - 2.0 * f)
        - 0.00028 * e * e * sin_d(mp + 2.0 * m)
        + 0.00027 * e * sin_d(2.0 * mp + m)
        - 0.00017 * sin_d(om)
        - 0.00005 * sin_d(mp - m - 2.0 * f)
        + 0.00004 * sin_d(2.0 * mp + 2.0 * f)
        - 0.00004 * sin_d(mp + m + 2.0 * f)
        + 0.00004 * sin_d(mp - 2.0 * m)
        + 0.00003 * sin_d(mp + m - 2.0 * f)
        + 0.00003 * sin_d(3.0 * m)
        + 0.00002 * sin_d(2.0 * mp - 2.0 * f)
        + 0.00002 * sin_d(mp - m + 2.0 * f)
        - 0.00002 * sin_d(3.0 * mp + m);
      let w = 0.00306 - 0.00038 * e * cos_d(m) + 0.00026 * cos_d(mp)
        - 0.00002 * cos_d(mp - m)
        + 0.00002 * cos_d(mp + m)
        + 0.00002 * cos_d(2.0 * f);
      jde += if matches!(phase, Phase::FirstQuarter) {
        w
      } else {
        -w
      };
    }
  }

  // Planetary perturbation terms (all phases).
  #[rustfmt::skip]
  let planetary: [(f64, f64, f64); 14] = [
    (0.000325, 299.77, 0.107408),
    (0.000165, 251.88, 0.016321),
    (0.000164, 251.83, 26.651886),
    (0.000126, 349.42, 36.412478),
    (0.000110, 84.66, 18.206239),
    (0.000062, 141.74, 53.303771),
    (0.000060, 207.14, 2.453732),
    (0.000056, 154.84, 7.306860),
    (0.000047, 34.52, 27.261239),
    (0.000042, 207.19, 0.121824),
    (0.000040, 291.34, 1.844379),
    (0.000037, 161.72, 24.198154),
    (0.000035, 239.56, 25.513099),
    (0.000023, 331.55, 3.592518),
  ];
  // The first term's argument includes the T² correction (A1 of ch. 49).
  jde +=
    planetary[0].0 * sin_d(planetary[0].1 + planetary[0].2 * k - 0.009173 * t2);
  for &(amp, a, b) in planetary.iter().skip(1) {
    jde += amp * sin_d(a + b * k);
  }
  jde
}

/// UTC Julian date of the first occurrence of `phase` strictly after
/// the UTC Julian date `jd`.
fn next_phase_jd(jd: f64, phase: Phase) -> f64 {
  let year = jd_to_decimal_year(jd);
  let k_guess = ((year - 2000.0) * 12.3685).floor() - 2.0;
  let mut k = k_guess;
  loop {
    let jde = phase_jde(k, phase);
    let jd_event = jde_to_jd_utc(jde);
    if jd_event > jd {
      return jd_event;
    }
    k += 1.0;
  }
}

// ─── Illuminated fraction (Meeus ch. 48) ────────────────────────────

/// Illuminated fraction of the Moon's disk and the signed elongation
/// (moon − sun apparent longitude, degrees in [0, 360)); the elongation
/// determines waxing (< 180) vs waning (> 180).
fn moon_illumination(jde: f64) -> (f64, f64) {
  let (lambda_m, beta_m, delta) = moon_coordinates(jde);
  let (lambda_s, _, r_au) = sun_coordinates(jde);
  let r = r_au * 149597870.7; // km

  // Geocentric elongation (Meeus 48.2)
  let cos_psi = cos_d(beta_m) * cos_d(lambda_m - lambda_s);
  let psi = cos_psi.clamp(-1.0, 1.0).acos();
  // Phase angle (48.3)
  let i = f64::atan2(r * psi.sin(), delta - r * psi.cos());
  let frac = (1.0 + i.cos()) / 2.0;
  (frac, norm360(lambda_m - lambda_s))
}

// ─── Eclipses (Meeus ch. 54) ────────────────────────────────────────

pub enum SolarEclipseKind {
  Partial,
  Annular,
  Total,
  Hybrid,
}

pub enum LunarEclipseKind {
  Penumbral,
  Partial,
  Total,
}

/// Eclipse test at lunation index k (integer → solar at new moon,
/// half-integer → lunar at full moon). Returns the JDE of greatest
/// eclipse plus gamma and u when an eclipse occurs.
fn eclipse_at_k(k: f64) -> Option<(f64, f64, f64)> {
  let t = k / 1236.85;
  let t2 = t * t;
  let t3 = t2 * t;
  let t4 = t3 * t;

  let f = norm360(
    160.7108 + 390.67050284 * k - 0.0016118 * t2 - 0.00000227 * t3
      + 0.000000011 * t4,
  );
  // No eclipse when the Moon is too far from the node.
  if sin_d(f).abs() > 0.36 {
    return None;
  }

  let e = 1.0 - 0.002516 * t - 0.0000074 * t2;
  let m = norm360(2.5534 + 29.10535670 * k - 0.0000014 * t2 - 0.00000011 * t3);
  let mp = norm360(
    201.5643 + 385.81693528 * k + 0.0107582 * t2 + 0.00001238 * t3
      - 0.000000058 * t4,
  );
  let om =
    norm360(124.7746 - 1.56375588 * k + 0.0020672 * t2 + 0.00000215 * t3);
  let f1 = f - 0.02665 * sin_d(om);
  let a1 = norm360(299.77 + 0.107408 * k - 0.009173 * t2);

  let mut jde = 2451550.09766 + 29.530588861 * k + 0.00015437 * t2
    - 0.000000150 * t3
    + 0.00000000073 * t4;

  let is_solar = k.fract().abs() < 0.25; // integer k → new moon
  jde += if is_solar {
    -0.4075 * sin_d(mp) + 0.1721 * e * sin_d(m)
  } else {
    -0.4065 * sin_d(mp) + 0.1727 * e * sin_d(m)
  };
  jde += 0.0161 * sin_d(2.0 * mp) - 0.0097 * sin_d(2.0 * f1)
    + 0.0073 * e * sin_d(mp - m)
    - 0.0050 * e * sin_d(mp + m)
    - 0.0023 * sin_d(mp - 2.0 * f1)
    + 0.0021 * e * sin_d(2.0 * m)
    + 0.0012 * sin_d(mp + 2.0 * f1)
    + 0.0006 * e * sin_d(2.0 * mp + m)
    - 0.0004 * sin_d(3.0 * mp)
    - 0.0003 * e * sin_d(m + 2.0 * f1)
    + 0.0003 * sin_d(a1)
    - 0.0002 * e * sin_d(m - 2.0 * f1)
    - 0.0002 * e * sin_d(2.0 * mp - m)
    - 0.0002 * sin_d(om);

  let p = 0.2070 * e * sin_d(m) + 0.0024 * e * sin_d(2.0 * m)
    - 0.0392 * sin_d(mp)
    + 0.0116 * sin_d(2.0 * mp)
    - 0.0073 * e * sin_d(mp + m)
    + 0.0067 * e * sin_d(mp - m)
    + 0.0118 * sin_d(2.0 * f1);
  let q = 5.2207 - 0.0048 * e * cos_d(m) + 0.0020 * e * cos_d(2.0 * m)
    - 0.3299 * cos_d(mp)
    - 0.0060 * e * cos_d(mp + m)
    + 0.0041 * e * cos_d(mp - m);
  let w = cos_d(f1).abs();
  let gamma = (p * cos_d(f1) + q * sin_d(f1)) * (1.0 - 0.0048 * w);
  let u = 0.0059 + 0.0046 * e * cos_d(m) - 0.0182 * cos_d(mp)
    + 0.0004 * cos_d(2.0 * mp)
    - 0.0005 * cos_d(m + mp);

  Some((jde, gamma, u))
}

/// Next solar eclipse strictly after the UTC Julian date `jd`:
/// (UTC JD of greatest eclipse, kind).
fn next_solar_eclipse(jd: f64) -> (f64, SolarEclipseKind) {
  let year = jd_to_decimal_year(jd);
  let mut k = ((year - 2000.0) * 12.3685).floor() - 2.0;
  loop {
    if let Some((jde, gamma, u)) = eclipse_at_k(k) {
      let jd_event = jde_to_jd_utc(jde);
      let ag = gamma.abs();
      if jd_event > jd && ag <= 1.5433 + u {
        let kind = if ag < 0.9972 {
          if u < 0.0 {
            SolarEclipseKind::Total
          } else if u > 0.0047 {
            SolarEclipseKind::Annular
          } else {
            let omega = 0.00464 * (1.0 - gamma * gamma).max(0.0).sqrt();
            if u < omega {
              SolarEclipseKind::Hybrid
            } else {
              SolarEclipseKind::Annular
            }
          }
        } else {
          SolarEclipseKind::Partial
        };
        return (jd_event, kind);
      }
    }
    k += 1.0;
  }
}

/// Next lunar eclipse strictly after the UTC Julian date `jd`.
fn next_lunar_eclipse(jd: f64) -> (f64, LunarEclipseKind) {
  let year = jd_to_decimal_year(jd);
  let mut k = ((year - 2000.0) * 12.3685).floor() - 2.0 + 0.5;
  loop {
    if let Some((jde, gamma, u)) = eclipse_at_k(k) {
      let jd_event = jde_to_jd_utc(jde);
      let ag = gamma.abs();
      let mag_umbral = (1.0128 - u - ag) / 0.5450;
      let mag_penumbral = (1.5573 + u - ag) / 0.5450;
      if jd_event > jd && mag_penumbral > 0.0 {
        let kind = if mag_umbral >= 1.0 {
          LunarEclipseKind::Total
        } else if mag_umbral > 0.0 {
          LunarEclipseKind::Partial
        } else {
          LunarEclipseKind::Penumbral
        };
        return (jd_event, kind);
      }
    }
    k += 1.0;
  }
}

// ─── Rise and set (Meeus ch. 15) ────────────────────────────────────

/// Standard altitude of the Sun's center at rise/set: refraction plus
/// semidiameter (−50′).
const SUN_RISE_SET_ALTITUDE: f64 = -0.8333;

/// UTC Julian dates of sunrise and sunset (and transit) for the UTC
/// calendar day containing `jd_day` (any instant of that day). Returns
/// None for polar day/night (the Sun never crosses the horizon).
fn sun_rise_set(jd_day: f64, lat: f64, lon: f64) -> Option<(f64, f64)> {
  // Midnight UT of the target day
  let jd0 = (jd_day - 0.5).floor() + 0.5;
  let theta0 = apparent_gst_deg(jd0);

  // Solar coordinates at midnight of day-1, day, day+1 for interpolation
  let coords = |offset: f64| sun_ra_dec(jd_utc_to_jde(jd0 + offset));
  let (ra_m, dec_m) = coords(-1.0);
  let (ra_0, dec_0) = coords(0.0);
  let (ra_p, dec_p) = coords(1.0);

  let h0 = SUN_RISE_SET_ALTITUDE;
  let cos_h =
    (sin_d(h0) - sin_d(lat) * sin_d(dec_0)) / (cos_d(lat) * cos_d(dec_0));
  if !(-1.0..=1.0).contains(&cos_h) {
    return None;
  }
  let big_h0 = cos_h.acos() / DEG;

  // Meeus uses west longitudes positive; ours are east-positive.
  let mut m0 = (ra_0 - lon - theta0) / 360.0;
  m0 -= m0.floor(); // transit fraction of day in [0,1)
  let mut m1 = m0 - big_h0 / 360.0; // rise
  let mut m2 = m0 + big_h0 / 360.0; // set

  // Interpolate α and δ at fraction m of the day (unwrap RA around 360°).
  let interp = |y1: f64, y2: f64, y3: f64, n: f64| {
    let a = y2 - y1;
    let b = y3 - y2;
    let c = b - a;
    y2 + n / 2.0 * (a + b + n * c)
  };
  let unwrap = |prev: f64, x: f64| {
    let mut x = x;
    while x - prev > 180.0 {
      x -= 360.0;
    }
    while x - prev < -180.0 {
      x += 360.0;
    }
    x
  };
  let ra_0u = unwrap(ra_m, ra_0);
  let ra_pu = unwrap(ra_0u, ra_p);

  for m in [&mut m1, &mut m2] {
    for _ in 0..2 {
      // The α/δ samples are taken at 0h UT of each day (ΔT is already
      // applied inside jd_utc_to_jde), so the interpolation factor is
      // the UT day fraction itself.
      let n = *m;
      let theta = norm360(theta0 + 360.985647 * *m);
      let ra = interp(ra_m, ra_0u, ra_pu, n);
      let dec = interp(dec_m, dec_0, dec_p, n);
      let h = norm360(theta + lon - ra); // local hour angle
      let h_signed = if h > 180.0 { h - 360.0 } else { h };
      let alt = (sin_d(lat) * sin_d(dec)
        + cos_d(lat) * cos_d(dec) * cos_d(h_signed))
      .asin()
        / DEG;
      let dm = (alt - h0) / (360.0 * cos_d(dec) * cos_d(lat) * sin_d(h_signed));
      *m += dm;
    }
  }

  Some((jd0 + m1, jd0 + m2))
}

// ─── Argument parsing ───────────────────────────────────────────────

/// Woxi's built-in `$GeoLocation` — like wolframscript without
/// geolocation access, this falls back to Wolfram's default location.
pub fn default_geo_location() -> Expr {
  Expr::FunctionCall {
    name: "GeoPosition".to_string(),
    args: vec![Expr::List(
      vec![Expr::Real(40.11), Expr::Real(-88.24)].into(),
    )]
    .into(),
  }
}

fn expr_to_f64(e: &Expr) -> Option<f64> {
  match e {
    Expr::Integer(n) => Some(*n as f64),
    Expr::Real(v) => Some(*v),
    Expr::FunctionCall { name, args }
      if name == "Rational" && args.len() == 2 =>
    {
      match (&args[0], &args[1]) {
        (Expr::Integer(p), Expr::Integer(q)) if *q != 0 => {
          Some(*p as f64 / *q as f64)
        }
        _ => None,
      }
    }
    Expr::UnaryOp {
      op: crate::syntax::UnaryOperator::Minus,
      operand,
    } => expr_to_f64(operand).map(|v| -v),
    _ => None,
  }
}

/// A location argument: `GeoPosition[{lat, lon(, h)}]` or a bare
/// `{lat, lon}` pair. Returns (lat, lon) in degrees, east positive.
fn parse_location(expr: &Expr) -> Option<(f64, f64)> {
  let coords = match expr {
    Expr::FunctionCall { name, args }
      if name == "GeoPosition" && args.len() == 1 =>
    {
      &args[0]
    }
    other => other,
  };
  if let Expr::List(items) = coords
    && (items.len() == 2 || items.len() == 3)
  {
    let lat = expr_to_f64(&items[0])?;
    let lon = expr_to_f64(&items[1])?;
    if (-90.0..=90.0).contains(&lat) && (-360.0..=360.0).contains(&lon) {
      return Some((lat, lon));
    }
  }
  None
}

/// Whether an argument looks like a location (GeoPosition wrapper or a
/// numeric pair in latitude range). Two-element numeric lists are read
/// as {lat, lon}; longer lists as date lists.
fn is_location_arg(expr: &Expr) -> bool {
  match expr {
    Expr::FunctionCall { name, .. } if name == "GeoPosition" => true,
    Expr::List(items) if items.len() == 2 => parse_location(expr).is_some(),
    _ => false,
  }
}

/// A date argument (date list, DateObject, date string, or `Now`) →
/// UTC Julian date.
fn parse_date_arg(expr: &Expr) -> Option<f64> {
  if let Expr::Identifier(name) = expr
    && name == "Now"
  {
    return Some(now_jd());
  }
  let list = resolve_date_to_list(expr)?;
  let components = extract_date_components(&list)?;
  let mut c = components;
  while c.len() < 3 {
    c.push(1.0);
  }
  while c.len() < 6 {
    c.push(0.0);
  }
  let abs = date_to_absolute_seconds(
    c[0] as i64,
    c[1] as i64,
    c[2] as i64,
    c[3] as i64,
    c[4] as i64,
    c[5],
  );
  Some(abs_seconds_to_jd(abs))
}

/// Split leading positional args into (location, date) with defaults:
/// [] → (default, now); [loc] / [date] → the other defaulted;
/// [loc, date]. Returns None when the args don't fit that shape.
fn parse_location_date(args: &[Expr]) -> Option<((f64, f64), f64)> {
  let default_loc = parse_location(&default_geo_location())?;
  match args {
    [] => Some((default_loc, now_jd())),
    [one] => {
      if is_location_arg(one) {
        Some((parse_location(one)?, now_jd()))
      } else {
        Some((default_loc, parse_date_arg(one)?))
      }
    }
    [loc, date] => Some((parse_location(loc)?, parse_date_arg(date)?)),
    _ => None,
  }
}

// ─── Result builders ────────────────────────────────────────────────

/// DateObject[{y, m, d, h, min}, "Minute", "Gregorian", 0.] for a UTC
/// Julian date, rounded to the nearest minute.
fn date_object_minute(jd: f64) -> Expr {
  let abs = jd_to_abs_seconds(jd);
  let rounded = (abs / 60.0).round() * 60.0;
  let (y, m, d, h, min, _) = absolute_seconds_to_date(rounded);
  Expr::FunctionCall {
    name: "DateObject".to_string(),
    args: vec![
      Expr::List(
        vec![
          Expr::Integer(y as i128),
          Expr::Integer(m as i128),
          Expr::Integer(d as i128),
          Expr::Integer(h as i128),
          Expr::Integer(min as i128),
        ]
        .into(),
      ),
      Expr::String("Minute".to_string()),
      Expr::String("Gregorian".to_string()),
      Expr::Real(0.0),
    ]
    .into(),
  }
}

/// DateObject[{y, m, d, h, min, s}, "Instant", "Gregorian", 0.] for a
/// UTC Julian date, with Real seconds.
fn date_object_instant(jd: f64) -> Expr {
  let abs = jd_to_abs_seconds(jd);
  let (y, m, d, h, min, s) = absolute_seconds_to_date(abs);
  // Sub-millisecond digits are ephemeris noise; keep the output stable.
  let s = (s * 1000.0).round() / 1000.0;
  Expr::FunctionCall {
    name: "DateObject".to_string(),
    args: vec![
      Expr::List(
        vec![
          Expr::Integer(y as i128),
          Expr::Integer(m as i128),
          Expr::Integer(d as i128),
          Expr::Integer(h as i128),
          Expr::Integer(min as i128),
          Expr::Real(s),
        ]
        .into(),
      ),
      Expr::String("Instant".to_string()),
      Expr::String("Gregorian".to_string()),
      Expr::Real(0.0),
    ]
    .into(),
  }
}

fn angle_quantity(deg: f64) -> Expr {
  // SunPosition/MoonPosition report hundredths of a degree.
  let rounded = (deg * 100.0).round() / 100.0;
  Expr::FunctionCall {
    name: "Quantity".to_string(),
    args: vec![
      Expr::Real(rounded),
      Expr::String("AngularDegrees".to_string()),
    ]
    .into(),
  }
}

fn unevaluated(name: &str, args: &[Expr]) -> Result<Expr, InterpreterError> {
  Ok(Expr::FunctionCall {
    name: name.to_string(),
    args: args.to_vec().into(),
  })
}

// ─── MoonPhase ──────────────────────────────────────────────────────

/// Phase name from the Sun→Moon elongation, using 45°-wide sectors
/// centered on the principal phases.
fn phase_entity_name(elongation: f64) -> &'static str {
  let sector = ((elongation + 22.5) / 45.0).floor() as i64 % 8;
  match sector {
    0 => "New",
    1 => "WaxingCrescent",
    2 => "FirstQuarter",
    3 => "WaxingGibbous",
    4 => "Full",
    5 => "WaningGibbous",
    6 => "LastQuarter",
    _ => "WaningCrescent",
  }
}

/// MoonPhase[] / MoonPhase[date] / MoonPhase[date, property] — the
/// illuminated fraction of the Moon's disk ("Fraction", the default)
/// or the phase name as a MoonPhase entity ("Name").
pub fn moon_phase_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  // A lone string is a property only when it names one; otherwise it is
  // read as a date string.
  let is_property = |s: &str| matches!(s, "Fraction" | "Name");
  let (date_args, property): (&[Expr], &str) = match args {
    [] => (&[], "Fraction"),
    [Expr::String(p)] if is_property(p) => (&[], p.as_str()),
    [date] => (std::slice::from_ref(date), "Fraction"),
    [date, Expr::String(p)] => (std::slice::from_ref(date), p.as_str()),
    _ => return unevaluated("MoonPhase", args),
  };
  let jd = match date_args {
    [] => now_jd(),
    [date] => match parse_date_arg(date) {
      Some(jd) => jd,
      None => return unevaluated("MoonPhase", args),
    },
    _ => unreachable!(),
  };
  let (fraction, elongation) = moon_illumination(jd_utc_to_jde(jd));
  match property {
    "Fraction" => Ok(Expr::Real(fraction)),
    "Name" => Ok(Expr::FunctionCall {
      name: "Entity".to_string(),
      args: vec![
        Expr::String("MoonPhase".to_string()),
        Expr::String(phase_entity_name(elongation).to_string()),
      ]
      .into(),
    }),
    _ => unevaluated("MoonPhase", args),
  }
}

/// NewMoon[] / NewMoon[date] — the DateObject of the first new moon
/// after the date (default: now).
pub fn new_moon_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  next_phase_function("NewMoon", Phase::New, args)
}

/// FullMoon[] / FullMoon[date] — the first full moon after the date.
pub fn full_moon_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  next_phase_function("FullMoon", Phase::Full, args)
}

fn next_phase_function(
  name: &str,
  phase: Phase,
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let jd = match args {
    [] => now_jd(),
    [date] => match parse_date_arg(date) {
      Some(jd) => jd,
      None => return unevaluated(name, args),
    },
    _ => return unevaluated(name, args),
  };
  Ok(date_object_instant(next_phase_jd(jd, phase)))
}

/// MoonPhaseDate[] / MoonPhaseDate[date] / MoonPhaseDate[date, phase] —
/// the date of the next principal lunar phase after `date` (default:
/// now); with a phase (string or MoonPhase entity), the next occurrence
/// of that phase.
pub fn moon_phase_date_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let (date_arg, phase_arg) = match args {
    [] => (None, None),
    [one] => {
      if phase_from_expr(one).is_some() {
        (None, Some(one))
      } else {
        (Some(one), None)
      }
    }
    [date, phase] => (Some(date), Some(phase)),
    _ => return unevaluated("MoonPhaseDate", args),
  };
  let jd = match date_arg {
    None => now_jd(),
    Some(date) => match parse_date_arg(date) {
      Some(jd) => jd,
      None => return unevaluated("MoonPhaseDate", args),
    },
  };
  let result = match phase_arg {
    None => {
      // Next principal phase of any kind
      [
        Phase::New,
        Phase::FirstQuarter,
        Phase::Full,
        Phase::LastQuarter,
      ]
      .into_iter()
      .map(|p| next_phase_jd(jd, p))
      .fold(f64::INFINITY, f64::min)
    }
    Some(p) => match phase_from_expr(p) {
      Some(phase) => next_phase_jd(jd, phase),
      None => return unevaluated("MoonPhaseDate", args),
    },
  };
  Ok(date_object_instant(result))
}

/// A phase spec: "New"/"NewMoon", "FirstQuarter", "Full"/"FullMoon",
/// "LastQuarter"/"ThirdQuarter", possibly wrapped in
/// Entity["MoonPhase", …].
fn phase_from_expr(expr: &Expr) -> Option<Phase> {
  let name = match expr {
    Expr::String(s) => s.as_str(),
    Expr::FunctionCall { name, args }
      if name == "Entity" && args.len() == 2 =>
    {
      match (&args[0], &args[1]) {
        (Expr::String(ty), Expr::String(n)) if ty == "MoonPhase" => n.as_str(),
        _ => return None,
      }
    }
    _ => return None,
  };
  match name {
    "New" | "NewMoon" => Some(Phase::New),
    "FirstQuarter" | "FirstQuarterMoon" => Some(Phase::FirstQuarter),
    "Full" | "FullMoon" => Some(Phase::Full),
    "LastQuarter" | "ThirdQuarter" | "LastQuarterMoon" => {
      Some(Phase::LastQuarter)
    }
    _ => None,
  }
}

// ─── SunPosition / MoonPosition ─────────────────────────────────────

/// The CelestialSystem option value from trailing Rule args, plus the
/// positional args with options stripped. Only "Horizon" (default) and
/// "Equatorial" are supported.
fn split_celestial_options(args: &[Expr]) -> (Vec<Expr>, String) {
  let mut positional = Vec::new();
  let mut system = "Horizon".to_string();
  for arg in args {
    if let Expr::Rule {
      pattern,
      replacement,
    } = arg
      && matches!(&**pattern, Expr::Identifier(n) if n == "CelestialSystem")
    {
      if let Expr::String(s) = &**replacement {
        system = s.clone();
      }
      continue;
    }
    positional.push(arg.clone());
  }
  (positional, system)
}

fn body_position_ast(
  name: &str,
  ra_dec: fn(f64) -> (f64, f64),
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let (positional, system) = split_celestial_options(args);
  let Some(((lat, lon), jd)) = parse_location_date(&positional) else {
    return unevaluated(name, args);
  };
  let (ra, dec) = ra_dec(jd_utc_to_jde(jd));
  let (a, b) = match system.as_str() {
    "Equatorial" => (ra, dec),
    "Horizon" => equatorial_to_horizontal(ra, dec, lat, lon, jd),
    _ => return unevaluated(name, args),
  };
  Ok(Expr::List(
    vec![angle_quantity(a), angle_quantity(b)].into(),
  ))
}

/// SunPosition[loc?, date?] — azimuth/altitude of the Sun (or
/// RA/declination with CelestialSystem -> "Equatorial").
pub fn sun_position_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  body_position_ast("SunPosition", sun_ra_dec, args)
}

/// MoonPosition[loc?, date?] — azimuth/altitude of the Moon.
pub fn moon_position_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  body_position_ast("MoonPosition", moon_ra_dec, args)
}

// ─── SiderealTime ───────────────────────────────────────────────────

/// SiderealTime[loc?, date?] — local apparent sidereal time as
/// Quantity[MixedMagnitude[{h, m, s}], MixedUnit[{…OfRightAscension}]].
pub fn sidereal_time_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let Some(((_, lon), jd)) = parse_location_date(args) else {
    return unevaluated("SiderealTime", args);
  };
  let lst_hours = norm360(apparent_gst_deg(jd) + lon) / 15.0;
  let h = lst_hours.floor();
  let m = ((lst_hours - h) * 60.0).floor();
  let s = ((lst_hours - h) * 60.0 - m) * 60.0;
  // Guard against 60.0000 seconds from rounding at display precision.
  let s = (s * 10000.0).round() / 10000.0;
  let (s, m, h) = if s >= 60.0 {
    (s - 60.0, m + 1.0, h)
  } else {
    (s, m, h)
  };
  let (m, h) = if m >= 60.0 {
    (m - 60.0, h + 1.0)
  } else {
    (m, h)
  };
  let h = if h >= 24.0 { h - 24.0 } else { h };
  Ok(Expr::FunctionCall {
    name: "Quantity".to_string(),
    args: vec![
      Expr::FunctionCall {
        name: "MixedMagnitude".to_string(),
        args: vec![Expr::List(
          vec![
            Expr::Integer(h as i128),
            Expr::Integer(m as i128),
            Expr::Real(s),
          ]
          .into(),
        )]
        .into(),
      },
      Expr::FunctionCall {
        name: "MixedUnit".to_string(),
        args: vec![Expr::List(
          vec![
            Expr::String("HoursOfRightAscension".to_string()),
            Expr::String("MinutesOfRightAscension".to_string()),
            Expr::String("SecondsOfRightAscension".to_string()),
          ]
          .into(),
        )]
        .into(),
      },
    ]
    .into(),
  })
}

// ─── Sunrise / Sunset / DaylightQ ───────────────────────────────────

fn sun_event_ast(
  name: &str,
  rise: bool,
  args: &[Expr],
) -> Result<Expr, InterpreterError> {
  let Some(((lat, lon), jd)) = parse_location_date(args) else {
    return unevaluated(name, args);
  };
  let explicit_date =
    matches!(args, [d] if !is_location_arg(d)) || args.len() == 2;

  if explicit_date {
    // The event on the given UTC calendar day
    match sun_rise_set(jd, lat, lon) {
      Some((r, s)) => Ok(date_object_minute(if rise { r } else { s })),
      None => Ok(Expr::FunctionCall {
        name: "Missing".to_string(),
        args: vec![Expr::String("NotApplicable".to_string())].into(),
      }),
    }
  } else {
    // No date given: the next event after now (skipping polar
    // day/night periods, bounded by a full year).
    for day in 0..366 {
      if let Some((r, s)) = sun_rise_set(jd + day as f64, lat, lon) {
        let event = if rise { r } else { s };
        if event > jd {
          return Ok(date_object_minute(event));
        }
        // Event already passed today; check the next day.
        continue;
      }
    }
    Ok(Expr::FunctionCall {
      name: "Missing".to_string(),
      args: vec![Expr::String("NotApplicable".to_string())].into(),
    })
  }
}

/// Sunrise[loc?, date?] — with a date, the sunrise of that day; without,
/// the next sunrise. Minute-granular DateObject in UTC.
pub fn sunrise_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  sun_event_ast("Sunrise", true, args)
}

/// Sunset[loc?, date?] — like Sunrise for the evening crossing.
pub fn sunset_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  sun_event_ast("Sunset", false, args)
}

/// DaylightQ[loc?, date?] — whether the Sun is up (above the standard
/// −50′ rise/set altitude) at the location and time.
pub fn daylight_q_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let Some(((lat, lon), jd)) = parse_location_date(args) else {
    return unevaluated("DaylightQ", args);
  };
  let (ra, dec) = sun_ra_dec(jd_utc_to_jde(jd));
  let (_, alt) = equatorial_to_horizontal(ra, dec, lat, lon, jd);
  Ok(Expr::Identifier(
    if alt > SUN_RISE_SET_ALTITUDE {
      "True"
    } else {
      "False"
    }
    .to_string(),
  ))
}

// ─── SolarEclipse / LunarEclipse ────────────────────────────────────

/// Positional args and the "Type"/"MaximumEclipseDate" property for the
/// eclipse functions.
fn eclipse_args(args: &[Expr]) -> Option<(Option<&Expr>, &str)> {
  // A lone string is a property only when it names one; otherwise it is
  // read as a date string.
  let is_property = |s: &str| matches!(s, "Type" | "MaximumEclipseDate");
  match args {
    [] => Some((None, "MaximumEclipseDate")),
    [Expr::String(p)] if is_property(p) => Some((None, p.as_str())),
    [date] => Some((Some(date), "MaximumEclipseDate")),
    [date, Expr::String(p)] => Some((Some(date), p.as_str())),
    _ => None,
  }
}

/// SolarEclipse[date?, property?] — the next solar eclipse after the
/// date (default now): its time of greatest eclipse, or its "Type"
/// ("Partial" | "Annular" | "Total" | "Hybrid").
pub fn solar_eclipse_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let Some((date_arg, property)) = eclipse_args(args) else {
    return unevaluated("SolarEclipse", args);
  };
  let jd = match date_arg {
    None => now_jd(),
    Some(date) => match parse_date_arg(date) {
      Some(jd) => jd,
      None => return unevaluated("SolarEclipse", args),
    },
  };
  let (jd_max, kind) = next_solar_eclipse(jd);
  match property {
    "MaximumEclipseDate" => Ok(date_object_instant(jd_max)),
    "Type" => Ok(Expr::String(
      match kind {
        SolarEclipseKind::Partial => "Partial",
        SolarEclipseKind::Annular => "Annular",
        SolarEclipseKind::Total => "Total",
        SolarEclipseKind::Hybrid => "Hybrid",
      }
      .to_string(),
    )),
    _ => unevaluated("SolarEclipse", args),
  }
}

/// LunarEclipse[date?, property?] — the next lunar eclipse after the
/// date: its time of greatest eclipse, or its "Type"
/// ("Penumbral" | "Partial" | "Total").
pub fn lunar_eclipse_ast(args: &[Expr]) -> Result<Expr, InterpreterError> {
  let Some((date_arg, property)) = eclipse_args(args) else {
    return unevaluated("LunarEclipse", args);
  };
  let jd = match date_arg {
    None => now_jd(),
    Some(date) => match parse_date_arg(date) {
      Some(jd) => jd,
      None => return unevaluated("LunarEclipse", args),
    },
  };
  let (jd_max, kind) = next_lunar_eclipse(jd);
  match property {
    "MaximumEclipseDate" => Ok(date_object_instant(jd_max)),
    "Type" => Ok(Expr::String(
      match kind {
        LunarEclipseKind::Penumbral => "Penumbral",
        LunarEclipseKind::Partial => "Partial",
        LunarEclipseKind::Total => "Total",
      }
      .to_string(),
    )),
    _ => unevaluated("LunarEclipse", args),
  }
}

// ─── Reference tests against Meeus's worked examples ────────────────

#[cfg(test)]
mod tests {
  use super::*;

  /// Meeus example 25.a: the Sun on 1992 October 13.0 TD.
  #[test]
  fn sun_position_meeus_25a() {
    let jde = 2448908.5;
    let (lambda, _, r) = sun_coordinates(jde);
    assert!((lambda - 199.90895).abs() < 0.0002, "λ = {lambda}");
    // The truncated series' R differs from the VSOP87 value (0.99760853)
    // by ~5e-5 AU; check against the low-accuracy method's own result.
    assert!((r - 0.9976620).abs() < 0.00001, "R = {r}");
    let (ra, dec) = sun_ra_dec(jde);
    assert!((ra - 198.38083).abs() < 0.001, "α = {ra}");
    assert!((dec - -7.78507).abs() < 0.001, "δ = {dec}");
  }

  /// Meeus example 47.a: the Moon on 1992 April 12.0 TD.
  #[test]
  fn moon_position_meeus_47a() {
    let jde = 2448724.5;
    let (lambda, beta, delta) = moon_coordinates(jde);
    // λ includes nutation (apparent longitude): 133.167265°
    assert!((lambda - 133.167265).abs() < 0.001, "λ = {lambda}");
    assert!((beta - -3.229126).abs() < 0.001, "β = {beta}");
    assert!((delta - 368409.7).abs() < 10.0, "Δ = {delta}");
  }

  /// Meeus example 49.a: the new moon of 1977 February 18.
  #[test]
  fn new_moon_meeus_49a() {
    let jde = phase_jde(-283.0, Phase::New);
    assert!((jde - 2443192.65118).abs() < 0.0005, "JDE = {jde}");
  }

  /// Meeus example 49.b: the last quarter of 2044 January 21.
  #[test]
  fn last_quarter_meeus_49b() {
    let jde = phase_jde(544.0, Phase::LastQuarter);
    assert!((jde - 2467636.49186).abs() < 0.0005, "JDE = {jde}");
  }

  /// Meeus example 54.a: the partial solar eclipse of 1993 May 21.
  #[test]
  fn solar_eclipse_meeus_54a() {
    let (jde, gamma, _) = eclipse_at_k(-82.0).expect("eclipse expected");
    // Greatest eclipse 1993 May 21 at 14:20:14 TD
    assert!((jde - 2449129.0974).abs() < 0.001, "JDE = {jde}");
    assert!((gamma - 1.1348).abs() < 0.001, "γ = {gamma}");
  }

  /// Meeus example 54.b: the total lunar eclipse of 1997 September 16.
  #[test]
  fn lunar_eclipse_meeus_54b() {
    let (jde, gamma, u) = eclipse_at_k(-28.5).expect("eclipse expected");
    // Greatest eclipse 1997 September 16 at 18:47 TD
    assert!((jde - 2450708.2831).abs() < 0.001, "JDE = {jde}");
    assert!((gamma - -0.3791).abs() < 0.001, "γ = {gamma}");
    let mag = (1.0128 - u - gamma.abs()) / 0.5450;
    assert!((mag - 1.1868).abs() < 0.01, "magnitude = {mag}");
  }

  /// Meeus example 12.b: apparent sidereal time on 1987 April 10.0 UT.
  #[test]
  fn sidereal_time_meeus_12a() {
    let jd = 2446895.5;
    let gmst = gmst_deg(jd);
    // 13h 10m 46.3668s = 197.693195°
    assert!((gmst - 197.693195).abs() < 0.0001, "GMST = {gmst}");
  }
}
