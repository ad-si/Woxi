//! `CategoricalDistribution`: a distribution over finite, unordered classes.
//!
//! Every specification evaluates to the canonical form
//! `CategoricalDistribution[<|"Probabilities" -> p, "Index" -> idx|>]`, where
//! `idx` maps each class to its position (one association per variable for a
//! multivariate distribution) and `p` is a single probability shared by every
//! domain element (the uniform case), a dense array of probabilities, or a
//! `SparseArray` when the weights were given as rules.

use crate::InterpreterError;
use crate::functions::math_ast::try_eval_to_f64;
use crate::helpers::{call, call1};
use crate::syntax::{Expr, expr_to_string};

fn eval(e: &Expr) -> Result<Expr, InterpreterError> {
  crate::evaluator::evaluate_expr_to_expr(e)
}

fn rule(k: Expr, v: Expr) -> Expr {
  Expr::Rule {
    pattern: Box::new(k),
    replacement: Box::new(v),
  }
}

fn as_rule(e: &Expr) -> Option<(&Expr, &Expr)> {
  match e {
    Expr::Rule {
      pattern,
      replacement,
    }
    | Expr::RuleDelayed {
      pattern,
      replacement,
    } => Some((pattern, replacement)),
    _ => None,
  }
}

/// Distinct elements in first-occurrence order.
fn distinct(items: &[Expr]) -> Vec<Expr> {
  let mut seen: Vec<String> = Vec::new();
  let mut out = Vec::new();
  for it in items {
    let s = expr_to_string(it);
    if !seen.contains(&s) {
      seen.push(s);
      out.push(it.clone());
    }
  }
  out
}

fn position_of(classes: &[Expr], e: &Expr) -> Option<usize> {
  let s = expr_to_string(e);
  classes.iter().position(|c| expr_to_string(c) == s)
}

fn index_assoc(classes: &[Expr]) -> Expr {
  Expr::Association(
    classes
      .iter()
      .enumerate()
      .map(|(i, c)| (c.clone(), Expr::Integer(i as i128 + 1)))
      .collect(),
  )
}

fn canonical(probabilities: Expr, index: Expr) -> Expr {
  call1(
    "CategoricalDistribution",
    Expr::Association(vec![
      (Expr::String("Probabilities".into()), probabilities),
      (Expr::String("Index".into()), index),
    ]),
  )
}

/// `w / Total[w]` for an array of weights.
fn normalize(weights: &Expr) -> Result<Expr, InterpreterError> {
  let total = call1("Total", call("Flatten", vec![weights.clone()]));
  eval(&call("Divide", vec![weights.clone(), total]))
}

/// The weights of rules `{elem -> w, …, _ -> default}` over a domain of
/// classes per variable, as a normalized `SparseArray`.
fn sparse_from_rules(
  rules: &[(Expr, Expr)],
  classes: &[Vec<Expr>],
) -> Result<Option<Expr>, InterpreterError> {
  let multivariate = classes.len() > 1;
  let mut default = Expr::Integer(0);
  let mut entries: Vec<(Vec<usize>, Expr)> = Vec::new();
  for (k, w) in rules {
    if is_blank(k) {
      default = w.clone();
      continue;
    }
    let comps: Vec<Expr> = match (multivariate, k) {
      (true, Expr::List(c)) => c.to_vec(),
      (true, _) => return Ok(None),
      (false, _) => vec![k.clone()],
    };
    if comps.len() != classes.len() {
      return Ok(None);
    }
    let Some(pos) = comps
      .iter()
      .zip(classes)
      .map(|(c, cl)| position_of(cl, c))
      .collect::<Option<Vec<_>>>()
    else {
      return Ok(None);
    };
    // A domain element given twice has no single weight.
    if entries.iter().any(|(p, _)| *p == pos) {
      return Ok(None);
    }
    entries.push((pos, w.clone()));
  }
  let domain_size: usize = classes.iter().map(Vec::len).product();
  // Total weight: the explicit weights plus the default for every element
  // not listed.
  let explicit = eval(&call(
    "Plus",
    entries.iter().map(|(_, w)| w.clone()).collect(),
  ))?;
  let omitted = (domain_size - entries.len()) as i128;
  let total = eval(&call(
    "Plus",
    vec![
      explicit,
      call("Times", vec![Expr::Integer(omitted), default.clone()]),
    ],
  ))?;
  let scale = |w: Expr| eval(&call("Divide", vec![w, total.clone()]));
  let rules_expr: Vec<Expr> = entries
    .into_iter()
    .map(|(pos, w)| {
      let at = if multivariate {
        Expr::List(pos.iter().map(|p| Expr::Integer(*p as i128 + 1)).collect())
      } else {
        Expr::Integer(pos[0] as i128 + 1)
      };
      Ok(rule(at, scale(w)?))
    })
    .collect::<Result<_, InterpreterError>>()?;
  let dims = Expr::List(
    classes
      .iter()
      .map(|c| Expr::Integer(c.len() as i128))
      .collect(),
  );
  Ok(Some(eval(&call(
    "SparseArray",
    vec![Expr::List(rules_expr.into()), dims, scale(default)?],
  ))?))
}

fn rules_of(e: &Expr) -> Option<Vec<(Expr, Expr)>> {
  match e {
    Expr::Association(pairs) => Some(pairs.clone()),
    Expr::List(items) if !items.is_empty() => items
      .iter()
      .map(|it| as_rule(it).map(|(k, v)| (k.clone(), v.clone())))
      .collect(),
    _ => None,
  }
}

fn is_blank(k: &Expr) -> bool {
  matches!(k, Expr::Pattern { .. })
    || matches!(k, Expr::FunctionCall { name, args } if name == "Blank" && args.is_empty())
}

/// The canonical form of `CategoricalDistribution[args]`, or `None` when the
/// arguments are already canonical or not a recognized specification.
pub(crate) fn categorical_distribution(
  args: &[Expr],
) -> Result<Option<Expr>, InterpreterError> {
  match args {
    // Already canonical.
    [Expr::Association(pairs)]
      if pairs
        .iter()
        .any(|(k, _)| matches!(k, Expr::String(s) if s == "Probabilities")) =>
    {
      Ok(None)
    }
    // Rules (or an association of counts) {elem -> w, …}.
    [spec] if rules_of(spec).is_some() => {
      let rules = rules_of(spec).unwrap_or_default();
      let elems: Vec<&Expr> = rules
        .iter()
        .map(|(k, _)| k)
        .filter(|k| !is_blank(k))
        .collect();
      let multivariate =
        !elems.is_empty() && elems.iter().all(|k| matches!(k, Expr::List(_)));
      // The classes of each variable, sorted.
      let classes: Vec<Vec<Expr>> = if multivariate {
        let Expr::List(first) = elems[0] else {
          return Ok(None);
        };
        let d = first.len();
        let mut cls = Vec::with_capacity(d);
        for i in 0..d {
          let mut col = Vec::with_capacity(elems.len());
          for k in &elems {
            match k {
              Expr::List(c) if c.len() == d => col.push(c[i].clone()),
              _ => return Ok(None),
            }
          }
          cls.push(sorted_distinct(col)?);
        }
        cls
      } else {
        vec![sorted_distinct(elems.into_iter().cloned().collect())?]
      };
      let Some(probs) = sparse_from_rules(&rules, &classes)? else {
        return Ok(None);
      };
      Ok(Some(canonical(probs, index_of(&classes))))
    }
    // A list of classes per variable (multivariate) or of classes.
    [Expr::List(items)] if !items.is_empty() => {
      if items.iter().all(|it| matches!(it, Expr::List(_))) {
        let classes: Vec<Vec<Expr>> = items
          .iter()
          .map(|it| match it {
            Expr::List(c) => distinct(c),
            _ => vec![],
          })
          .collect();
        let size: i128 = classes.iter().map(|c| c.len() as i128).product();
        let p = crate::functions::math_ast::make_rational(1, size);
        return Ok(Some(canonical(p, index_of(&classes))));
      }
      let classes = distinct(items);
      if classes.len() == items.len() {
        let p =
          crate::functions::math_ast::make_rational(1, classes.len() as i128);
        return Ok(Some(canonical(p, index_assoc(&classes))));
      }
      // Repeated classes: probabilities from their frequencies.
      let counts: Vec<Expr> = classes
        .iter()
        .map(|c| {
          let s = expr_to_string(c);
          Expr::Integer(
            items.iter().filter(|it| expr_to_string(it) == s).count() as i128,
          )
        })
        .collect();
      Ok(Some(canonical(
        normalize(&Expr::List(counts.into()))?,
        index_assoc(&classes),
      )))
    }
    // A domain with weights: an array, or rules for some domain elements.
    [Expr::List(domain), weights] if !domain.is_empty() => {
      let multivariate = domain.iter().all(|it| matches!(it, Expr::List(_)));
      let classes: Vec<Vec<Expr>> = if multivariate {
        domain
          .iter()
          .map(|it| match it {
            Expr::List(c) => c.to_vec(),
            _ => vec![],
          })
          .collect()
      } else {
        vec![domain.to_vec()]
      };
      if let Some(rules) = rules_of(weights)
        && !matches!(weights, Expr::Association(_))
      {
        let Some(probs) = sparse_from_rules(&rules, &classes)? else {
          return Ok(None);
        };
        return Ok(Some(canonical(probs, index_of(&classes))));
      }
      // The weight array must have the domain's dimensions.
      let dims = eval(&call1("Dimensions", weights.clone()))?;
      let want: Vec<Expr> = classes
        .iter()
        .map(|c| Expr::Integer(c.len() as i128))
        .collect();
      if expr_to_string(&dims) != expr_to_string(&Expr::List(want.into())) {
        return Ok(None);
      }
      let w = match weights {
        Expr::FunctionCall { name, .. } if name == "SparseArray" => {
          eval(&call(
            "Divide",
            vec![
              weights.clone(),
              call1(
                "Total",
                call1("Flatten", call1("Normal", weights.clone())),
              ),
            ],
          ))?
        }
        _ => normalize(weights)?,
      };
      Ok(Some(canonical(w, index_of(&classes))))
    }
    _ => Ok(None),
  }
}

fn sorted_distinct(items: Vec<Expr>) -> Result<Vec<Expr>, InterpreterError> {
  match &eval(&call1("Union", Expr::List(items.into())))? {
    Expr::List(v) => Ok(v.to_vec()),
    _ => Ok(vec![]),
  }
}

fn index_of(classes: &[Vec<Expr>]) -> Expr {
  if classes.len() == 1 {
    index_assoc(&classes[0])
  } else {
    Expr::List(classes.iter().map(|c| index_assoc(c)).collect())
  }
}

/// The parts of a canonical categorical distribution.
struct Categorical {
  probabilities: Expr,
  /// Classes of each variable, in index order.
  classes: Vec<Vec<Expr>>,
}

impl Categorical {
  fn from_expr(dist: &Expr) -> Option<Self> {
    let Expr::FunctionCall { name, args } = dist else {
      return None;
    };
    if name != "CategoricalDistribution" || args.len() != 1 {
      return None;
    }
    let Expr::Association(pairs) = &args[0] else {
      return None;
    };
    let get = |key: &str| {
      pairs
        .iter()
        .find(|(k, _)| matches!(k, Expr::String(s) if s == key))
        .map(|(_, v)| v.clone())
    };
    let probabilities = get("Probabilities")?;
    let keys = |e: &Expr| match e {
      Expr::Association(p) => Some(p.iter().map(|(k, _)| k.clone()).collect()),
      _ => None,
    };
    let index = get("Index")?;
    let classes = match &index {
      idx @ Expr::Association(_) => vec![keys(idx)?],
      Expr::List(idxs) => idxs.iter().map(keys).collect::<Option<Vec<_>>>()?,
      _ => return None,
    };
    Some(Self {
      probabilities,
      classes,
    })
  }

  fn dimension(&self) -> usize {
    self.classes.len()
  }

  fn is_uniform(&self) -> bool {
    !matches!(&self.probabilities, Expr::List(_))
      && !matches!(&self.probabilities, Expr::FunctionCall { name, .. } if name == "SparseArray")
  }

  /// The probability at 0-based `pos`.
  fn prob_at(&self, pos: &[usize]) -> Result<Expr, InterpreterError> {
    if self.is_uniform() {
      return Ok(self.probabilities.clone());
    }
    let mut part = vec![self.probabilities.clone()];
    part.extend(pos.iter().map(|p| Expr::Integer(*p as i128 + 1)));
    eval(&call("Part", part))
  }

  /// The 0-based position of a domain element.
  fn position(&self, elem: &Expr) -> Option<Vec<usize>> {
    if self.dimension() == 1 {
      return position_of(&self.classes[0], elem).map(|p| vec![p]);
    }
    let Expr::List(comps) = elem else {
      return None;
    };
    if comps.len() != self.dimension() {
      return None;
    }
    comps
      .iter()
      .zip(&self.classes)
      .map(|(c, cl)| position_of(cl, c))
      .collect()
  }

  /// All domain elements with their positions, in index order.
  fn domain(&self) -> Vec<(Expr, Vec<usize>)> {
    let mut out: Vec<Vec<usize>> = vec![vec![]];
    for cl in &self.classes {
      out = out
        .into_iter()
        .flat_map(|prefix| {
          (0..cl.len()).map(move |i| {
            let mut p = prefix.clone();
            p.push(i);
            p
          })
        })
        .collect();
    }
    out
      .into_iter()
      .map(|pos| {
        let elem = if self.dimension() == 1 {
          self.classes[0][pos[0]].clone()
        } else {
          Expr::List(
            pos
              .iter()
              .zip(&self.classes)
              .map(|(p, cl)| cl[*p].clone())
              .collect(),
          )
        };
        (elem, pos)
      })
      .collect()
  }

  fn probabilities_assoc(&self) -> Result<Vec<(Expr, Expr)>, InterpreterError> {
    self
      .domain()
      .into_iter()
      .map(|(e, pos)| Ok((e, self.prob_at(&pos)?)))
      .collect()
  }
}

/// `PDF[CategoricalDistribution[…], x]`; `None` when `dist` is not a
/// canonical categorical distribution.
pub(crate) fn categorical_pdf(
  dist: &Expr,
  x: &Expr,
) -> Option<Result<Expr, InterpreterError>> {
  let cd = Categorical::from_expr(dist)?;
  if let Some(pos) = cd.position(x) {
    return Some(cd.prob_at(&pos));
  }
  // A list of univariate classes gives the list of their probabilities.
  if cd.dimension() == 1
    && let Expr::List(xs) = x
    && let Some(ps) = xs
      .iter()
      .map(|e| cd.position(e))
      .collect::<Option<Vec<_>>>()
  {
    return Some(
      ps.iter()
        .map(|p| cd.prob_at(p))
        .collect::<Result<Vec<_>, _>>()
        .map(|v| Expr::List(v.into())),
    );
  }
  Some(Ok(call("PDF", vec![dist.clone(), x.clone()])))
}

/// The weighted domain of a categorical distribution with numeric
/// probabilities, for sampling: `weights -> elements`.
pub(crate) fn categorical_choice_spec(dist: &Expr) -> Option<Expr> {
  let cd = Categorical::from_expr(dist)?;
  let probs = cd.probabilities_assoc().ok()?;
  if probs.iter().any(|(_, p)| try_eval_to_f64(p).is_none()) {
    return None;
  }
  let (elems, weights): (Vec<Expr>, Vec<Expr>) = probs.into_iter().unzip();
  Some(rule(Expr::List(weights.into()), Expr::List(elems.into())))
}

/// Whether `dist` is a canonical categorical distribution.
pub(crate) fn is_categorical(dist: &Expr) -> bool {
  Categorical::from_expr(dist).is_some()
}

const PROPERTIES: [&str; 12] = [
  "Properties",
  "Entropy",
  "NEntropy",
  "Categories",
  "DomainSize",
  "DomainElements",
  "Probabilities",
  "ProbabilityTable",
  "ProbabilityArray",
  "ProbabilityPlot",
  "TopProbabilities",
  "Dimension",
];

/// `Information[CategoricalDistribution[…], property]` for the properties
/// that are plain expressions; `None` for anything else.
pub(crate) fn categorical_information(
  dist: &Expr,
  property: &Expr,
) -> Option<Result<Expr, InterpreterError>> {
  let cd = Categorical::from_expr(dist)?;
  let (name, top_n) = match property {
    Expr::String(s) => (s.as_str(), None),
    Expr::Rule {
      pattern,
      replacement,
    } => match (pattern.as_ref(), replacement.as_ref()) {
      (Expr::String(s), Expr::Integer(n))
        if s == "TopProbabilities" && *n >= 0 =>
      {
        (s.as_str(), Some(*n as usize))
      }
      _ => return None,
    },
    _ => return None,
  };
  let result = (|| -> Result<Option<Expr>, InterpreterError> {
    Ok(Some(match name {
      "Properties" => Expr::List(
        PROPERTIES
          .iter()
          .map(|p| Expr::String((*p).into()))
          .collect(),
      ),
      "Categories" => {
        let lists: Vec<Expr> = cd
          .classes
          .iter()
          .map(|c| Expr::List(c.clone().into()))
          .collect();
        if cd.dimension() == 1 {
          lists[0].clone()
        } else {
          Expr::List(lists.into())
        }
      }
      "Dimension" => Expr::Integer(cd.dimension() as i128),
      "DomainSize" => {
        Expr::Integer(cd.classes.iter().map(|c| c.len() as i128).product())
      }
      "DomainElements" => {
        Expr::List(cd.domain().into_iter().map(|(e, _)| e).collect())
      }
      "Probabilities" => Expr::Association(cd.probabilities_assoc()?),
      "ProbabilityArray" => {
        if cd.is_uniform() {
          let dims: Vec<Expr> = cd
            .classes
            .iter()
            .map(|c| Expr::Integer(c.len() as i128))
            .collect();
          eval(&call(
            "ConstantArray",
            vec![cd.probabilities.clone(), Expr::List(dims.into())],
          ))?
        } else {
          eval(&call1("Normal", cd.probabilities.clone()))?
        }
      }
      // Sum of p Log[1/p] over the elements of nonzero probability.
      "Entropy" => {
        let terms: Vec<Expr> = cd
          .probabilities_assoc()?
          .into_iter()
          .filter(|(_, p)| !matches!(p, Expr::Integer(0)))
          .map(|(_, p)| {
            call(
              "Times",
              vec![
                p.clone(),
                call1("Log", call("Power", vec![p, Expr::Integer(-1)])),
              ],
            )
          })
          .collect();
        eval(&call("Plus", terms))?
      }
      "TopProbabilities" => {
        let mut probs = cd.probabilities_assoc()?;
        // Stable sort by decreasing probability.
        let keyed: Option<Vec<f64>> =
          probs.iter().map(|(_, p)| try_eval_to_f64(p)).collect();
        let Some(keyed) = keyed else {
          return Ok(None);
        };
        let mut order: Vec<usize> = (0..probs.len()).collect();
        order.sort_by(|&a, &b| keyed[b].total_cmp(&keyed[a]));
        let mut sorted: Vec<(Expr, Expr)> =
          order.iter().map(|&i| probs[i].clone()).collect();
        probs.clear();
        if let Some(n) = top_n {
          sorted.truncate(n);
        }
        Expr::List(sorted.into_iter().map(|(e, p)| rule(e, p)).collect())
      }
      _ => return Ok(None),
    }))
  })();
  match result {
    Ok(Some(v)) => Some(Ok(v)),
    Ok(None) => None,
    Err(e) => Some(Err(e)),
  }
}

/// `MarginalDistribution[CategoricalDistribution[…], k]` for an array of
/// probabilities: the probabilities summed over the other variables.
pub(crate) fn categorical_marginal(
  dist: &Expr,
  k: &Expr,
) -> Option<Result<Expr, InterpreterError>> {
  let cd = Categorical::from_expr(dist)?;
  let Expr::Integer(k) = k else {
    return None;
  };
  let d = cd.dimension();
  if cd.is_uniform() || *k < 1 || *k as usize > d || d < 2 {
    return None;
  }
  let k = *k as usize - 1;
  let probs = cd.probabilities_assoc().ok()?;
  let n = cd.classes[k].len();
  let mut sums: Vec<Vec<Expr>> = vec![vec![]; n];
  for ((_, p), (_, pos)) in probs.into_iter().zip(cd.domain()) {
    sums[pos[k]].push(p);
  }
  Some(
    sums
      .into_iter()
      .map(|terms| eval(&call("Plus", terms)))
      .collect::<Result<Vec<_>, _>>()
      .map(|v| canonical(Expr::List(v.into()), index_assoc(&cd.classes[k]))),
  )
}
