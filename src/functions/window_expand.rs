//! Closed forms of the window functions, as `FunctionExpand` writes them:
//! a `Piecewise` that is the window's defining expression on
//! `-1/2 <= x <= 1/2` (or on its halves) and 0 elsewhere.
//!
//!   FunctionExpand[HannWindow[x]]
//!     → Piecewise[{{1/2 + Cos[2 Pi x]/2, -1/2 <= x <= 1/2}}, 0]
//!
//! The two-argument forms whose evaluator only covers one argument
//! (`HannWindow[x, a]`, `CosineWindow[x, a]`, `ConnesWindow[x, a]`,
//! `WelchWindow[x, a]`) are evaluated through the same closed form.

use crate::helpers::call;
use crate::syntax::Expr;

/// The template of `name` with `wX` standing for the variable and `wA` for
/// the parameter, as a list of `{value, condition}` clauses (default 0).
fn template(name: &str, with_param: bool) -> Option<&'static str> {
  Some(match (name, with_param) {
    ("HammingWindow", false) => {
      "{{25/46 + (21/46) Cos[2 Pi wX], -1/2 <= wX <= 1/2}}"
    }
    ("HannWindow", false) => "{{1/2 + (1/2) Cos[2 Pi wX], -1/2 <= wX <= 1/2}}",
    ("HannWindow", true) => {
      "{{wA + Cos[2 Pi wX] - wA Cos[2 Pi wX], -1/2 <= wX <= 1/2}}"
    }
    ("KaiserWindow", false) => {
      "{{BesselI[0, 3 Sqrt[1 - 2 wX] Sqrt[1 + 2 wX]]/BesselI[0, 3], -1/2 <= wX <= 1/2}}"
    }
    ("KaiserWindow", true) => {
      "{{BesselI[0, Sqrt[1 - 2 wX] Sqrt[1 + 2 wX] wA]/BesselI[0, wA], -1/2 <= wX <= 1/2}}"
    }
    ("BlackmanWindow", false) => {
      "{{(1/50) (21 + 25 Cos[2 Pi wX] + 4 Cos[4 Pi wX]), -1/2 <= wX <= 1/2}}"
    }
    ("DirichletWindow", false) => "{{1, -1/2 <= wX <= 1/2}}",
    ("BlackmanHarrisWindow", false) => {
      "{{(1/100000) (35875 + 48829 Cos[2 Pi wX] + 14128 Cos[4 Pi wX] + 1168 Cos[6 Pi wX]), -1/2 <= wX <= 1/2}}"
    }
    ("CosineWindow", false) => "{{Cos[Pi wX], -1/2 <= wX <= 1/2}}",
    ("CosineWindow", true) => "{{Cos[Pi wX]^wA, -1/2 <= wX <= 1/2}}",
    ("BartlettWindow", false) => {
      "{{1 - 2 wX, 0 <= wX <= 1/2}, {1 + 2 wX, -1/2 <= wX < 0}}"
    }
    ("BlackmanNuttallWindow", false) => {
      "{{(1/10000000) (3635819 + 4891775 Cos[2 Pi wX] + 1365995 Cos[4 Pi wX] + 106411 Cos[6 Pi wX]), -1/2 <= wX <= 1/2}}"
    }
    ("GaussianWindow", false) => "{{E^(-50 wX^2/9), -1/2 <= wX <= 1/2}}",
    ("GaussianWindow", true) => "{{E^(-wX^2/(2 wA^2)), -1/2 <= wX <= 1/2}}",
    ("CauchyWindow", false) => "{{1/(1 + 36 wX^2), -1/2 <= wX <= 1/2}}",
    ("CauchyWindow", true) => "{{1/(1 + 4 wX^2 wA^2), -1/2 <= wX <= 1/2}}",
    ("ConnesWindow", false) => "{{(1 - 4 wX^2)^2, -1/2 <= wX <= 1/2}}",
    ("ConnesWindow", true) => "{{(-4 wX^2 + wA^2)^2/wA^4, -1/2 <= wX <= 1/2}}",
    ("PoissonWindow", false) => {
      "{{E^(-6 wX), 0 <= wX <= 1/2}, {E^(6 wX), -1/2 <= wX < 0}}"
    }
    ("PoissonWindow", true) => {
      "{{E^(-2 wX wA), 0 <= wX <= 1/2}, {E^(2 wX wA), -1/2 <= wX < 0}}"
    }
    ("WelchWindow", false) => "{{1 - 4 wX^2, -1/2 <= wX <= 1/2}}",
    ("WelchWindow", true) => "{{(-4 wX^2 + wA^2)/wA^2, -1/2 <= wX <= 1/2}}",
    ("NuttallWindow", false) => {
      "{{(1/250000) (88942 + 121849 Cos[2 Pi wX] + 36058 Cos[4 Pi wX] + 3151 Cos[6 Pi wX]), -1/2 <= wX <= 1/2}}"
    }
    ("BartlettHannWindow", false) => {
      "{{31/50 - 12 wX/25 + (19/50) Cos[2 Pi wX], 0 <= wX <= 1/2}, {31/50 + 12 wX/25 + (19/50) Cos[2 Pi wX], -1/2 <= wX < 0}}"
    }
    ("ParzenWindow", false) => {
      "{{-2 (-1 + 2 wX)^3, 1/4 < wX <= 1/2}, {2 (1 + 2 wX)^3, -1/2 <= wX < -1/4}, {1 - 24 wX^2 - 48 wX^3, -1/4 <= wX < 0}, {1 - 24 wX^2 + 48 wX^3, 0 <= wX <= 1/4}}"
    }
    ("BohmanWindow", false) => {
      "{{(Pi Cos[2 Pi wX] + 2 Pi wX Cos[2 Pi wX] - Sin[2 Pi wX])/Pi, -1/2 <= wX < 0}, {(Pi Cos[2 Pi wX] - 2 Pi wX Cos[2 Pi wX] + Sin[2 Pi wX])/Pi, 0 <= wX <= 1/2}}"
    }
    ("ExactBlackmanWindow", false) => {
      "{{(1/9304) (3969 + 4620 Cos[2 Pi wX] + 715 Cos[4 Pi wX]), -1/2 <= wX <= 1/2}}"
    }
    ("FlatTopWindow", false) => {
      "{{(1/1000000000) (215578947 + 416631580 Cos[2 Pi wX] + 277263158 Cos[4 Pi wX] + 83578947 Cos[6 Pi wX] + 6947368 Cos[8 Pi wX]), -1/2 <= wX <= 1/2}}"
    }
    ("KaiserBesselWindow", false) => {
      "{{(1/1000) (402 + 498 Cos[2 Pi wX] + 99 Cos[4 Pi wX] + Cos[6 Pi wX]), -1/2 <= wX <= 1/2}}"
    }
    ("LanczosWindow", false) => "{{Sin[2 Pi wX]/(2 Pi wX), -1/2 <= wX <= 1/2}}",
    _ => return None,
  })
}

/// `Piecewise[...]` for `name[args]` (not yet evaluated), or None when the
/// window has no closed form here.
pub fn window_piecewise(name: &str, args: &[Expr]) -> Option<Expr> {
  if args.is_empty() || args.len() > 2 {
    return None;
  }
  let src = template(name, args.len() == 2)?;
  let clauses = crate::syntax::string_to_expr(src).ok()?;
  let mut rules = vec![Expr::Rule {
    pattern: Box::new(Expr::Identifier("wX".into())),
    replacement: Box::new(args[0].clone()),
  }];
  if let Some(a) = args.get(1) {
    rules.push(Expr::Rule {
      pattern: Box::new(Expr::Identifier("wA".into())),
      replacement: Box::new(a.clone()),
    });
  }
  let substituted = call("ReplaceAll", vec![clauses, Expr::List(rules.into())]);
  Some(call("Piecewise", vec![substituted, Expr::Integer(0)]))
}
