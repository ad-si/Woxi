//! Conversion between Woxi's internal `syntax::Expr` and a FullForm-shaped
//! Python expression tree.
//!
//! The internal `Expr` is a parser AST: `a + b` is a `BinaryOp` in one place
//! and a `FunctionCall{"Plus"}` in another, and it grows variants as the
//! interpreter grows. Exposing it directly would leak that representation
//! into a public API. Instead the Python side sees the canonical FullForm
//! shape — atoms plus `Expr(head, args)` nodes — which is a stable contract
//! and matches what `wolframclient` users expect.
//!
//! The normalisation itself is *not* reimplemented here: everything that is
//! not an atom goes through `woxi::functions::expr_form::decompose_expr`,
//! the same decomposition `FullForm` and pattern matching use.

use num_bigint::BigInt;
use pyo3::exceptions::{PyRecursionError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::{
  PyBool, PyComplex, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple,
};

use woxi::functions::expr_form::{
  ComplexParts, ExprForm, complex_full_form_parts, decompose_expr,
};
use woxi::syntax::Expr as WExpr;

/// Deepest expression tree that converts in either direction. Deep enough
/// for anything a real evaluation produces, shallow enough that the native
/// frames cannot overflow the stack.
const MAX_DEPTH: usize = 5000;

fn too_deep() -> PyErr {
  PyRecursionError::new_err(format!(
    "expression nests deeper than {MAX_DEPTH} levels"
  ))
}

/// A Wolfram symbol, e.g. `Symbol("Pi")` or `Symbol("x")`.
///
/// Calling a symbol builds an expression: `Symbol("f")(1, 2)` is `f[1, 2]`.
#[pyclass(frozen, name = "Symbol", module = "woxi")]
pub struct Symbol {
  /// The symbol's name.
  #[pyo3(get)]
  pub name: String,
}

#[pymethods]
impl Symbol {
  #[new]
  fn new(name: String) -> Self {
    Self { name }
  }

  #[pyo3(signature = (*args))]
  fn __call__(
    slf: &Bound<'_, Self>,
    args: &Bound<'_, PyTuple>,
  ) -> PyResult<Expr> {
    Expr::new(slf.as_any(), args.as_any())
  }

  fn __repr__(&self) -> String {
    format!("Symbol({:?})", self.name)
  }

  fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
    match other.cast::<Self>() {
      Ok(o) => self.name == o.get().name,
      Err(_) => false,
    }
  }

  fn __hash__(&self) -> u64 {
    use std::hash::{DefaultHasher, Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    "woxi.Symbol".hash(&mut hasher);
    self.name.hash(&mut hasher);
    hasher.finish()
  }
}

/// A composite Wolfram expression: a head applied to arguments.
///
/// `head` is normally a [`Symbol`], but is itself an `Expr` for curried
/// calls — `f[a][b]` is `Expr(Expr(Symbol("f"), [Symbol("a")]), [Symbol("b")])`.
#[pyclass(frozen, name = "Expr", module = "woxi")]
pub struct Expr {
  /// The expression's head.
  #[pyo3(get)]
  pub head: Py<PyAny>,
  /// The arguments, as a list.
  #[pyo3(get)]
  pub args: Py<PyList>,
}

#[pymethods]
impl Expr {
  #[new]
  #[pyo3(signature = (head, args))]
  fn new(head: &Bound<'_, PyAny>, args: &Bound<'_, PyAny>) -> PyResult<Self> {
    let items = args.try_iter().map_err(|_| {
      PyTypeError::new_err("Expr args must be an iterable of expressions")
    })?;
    let args = PyList::new(
      args.py(),
      items.collect::<PyResult<Vec<Bound<'_, PyAny>>>>()?,
    )?;
    Ok(Self {
      head: head.clone().unbind(),
      args: args.unbind(),
    })
  }

  #[pyo3(signature = (*args))]
  fn __call__(
    slf: &Bound<'_, Self>,
    args: &Bound<'_, PyTuple>,
  ) -> PyResult<Self> {
    Self::new(slf.as_any(), args.as_any())
  }

  fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
    Ok(format!(
      "Expr({}, {})",
      self.head.bind(py).repr()?,
      self.args.bind(py).repr()?
    ))
  }

  fn __eq__(&self, other: &Bound<'_, PyAny>) -> PyResult<bool> {
    let py = other.py();
    let Ok(o) = other.cast::<Self>() else {
      return Ok(false);
    };
    let o = o.get();
    Ok(
      self.head.bind(py).eq(o.head.bind(py))?
        && self.args.bind(py).eq(o.args.bind(py))?,
    )
  }

  fn __hash__(&self, py: Python<'_>) -> PyResult<isize> {
    let args = PyTuple::new(py, self.args.bind(py).iter())?;
    PyTuple::new(py, [self.head.bind(py).clone(), args.into_any()])?.hash()
  }
}

/// An arbitrary-precision real, e.g. the result of `N[Pi, 30]`.
///
/// Kept distinct from `float` because the precision is part of the value:
/// a machine `float` cannot represent it.
#[pyclass(frozen, get_all, name = "BigReal", module = "woxi")]
pub struct BigReal {
  /// The digits, as Woxi stores them.
  pub digits: String,
  /// Precision in decimal digits.
  pub precision: f64,
}

#[pymethods]
impl BigReal {
  #[new]
  fn new(digits: String, precision: f64) -> Self {
    Self { digits, precision }
  }

  fn __repr__(&self) -> String {
    format!("BigReal({:?}, {})", self.digits, self.precision)
  }

  fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
    match other.cast::<Self>() {
      Ok(o) => {
        let o = o.get();
        self.digits == o.digits && self.precision == o.precision
      }
      Err(_) => false,
    }
  }
}

/// A rendered graphic. Output only — the SVG cannot be evaluated back, so
/// passing one to `evaluate_expr` raises `TypeError`. `EvaluationResult`
/// also exposes the same markup as its `graphics` field.
#[pyclass(frozen, get_all, name = "Graphics", module = "woxi")]
pub struct Graphics {
  /// The SVG markup.
  pub svg: String,
  /// Whether this came from `Graphics3D` rather than `Graphics`.
  pub is_3d: bool,
}

#[pymethods]
impl Graphics {
  fn __repr__(&self) -> String {
    format!(
      "Graphics(is_3d={}, svg=<{} bytes>)",
      if self.is_3d { "True" } else { "False" },
      self.svg.len()
    )
  }
}

/// A raster image. Output only, like [`Graphics`] — the pixel data is not
/// carried into Python.
#[pyclass(frozen, get_all, name = "Image", module = "woxi")]
pub struct Image {
  pub width: u32,
  pub height: u32,
  pub channels: u8,
}

#[pymethods]
impl Image {
  fn __repr__(&self) -> String {
    format!(
      "Image(width={}, height={}, channels={})",
      self.width, self.height, self.channels
    )
  }
}

/// Convert an interpreter expression to its Python tree.
pub fn expr_to_py(py: Python<'_>, expr: &WExpr) -> PyResult<Py<PyAny>> {
  expr_to_py_at(py, expr, 0)
}

fn symbol_obj(py: Python<'_>, name: &str) -> PyResult<Py<PyAny>> {
  Ok(
    Bound::new(
      py,
      Symbol {
        name: name.to_string(),
      },
    )?
    .into_any()
    .unbind(),
  )
}

/// A `(numerator, denominator)` pair as FullForm spells it: reduced to
/// lowest terms, a bare integer when the denominator is 1, otherwise
/// `Rational[n, d]`. Mirrors `render_rational_full_form`, which does the
/// same for the textual FullForm.
fn rational(parts: (i128, i128)) -> WExpr {
  let (numer, denom) = woxi::functions::math_ast::rat_reduce(parts.0, parts.1);
  if denom == 1 {
    WExpr::Integer(numer)
  } else {
    WExpr::FunctionCall {
      name: "Rational".to_string(),
      args: vec![WExpr::Integer(numer), WExpr::Integer(denom)].into(),
    }
  }
}

fn composite_obj(
  py: Python<'_>,
  head: Py<PyAny>,
  children: &[WExpr],
  depth: usize,
) -> PyResult<Py<PyAny>> {
  let args = children
    .iter()
    .map(|c| expr_to_py_at(py, c, depth + 1))
    .collect::<PyResult<Vec<_>>>()?;
  Ok(
    Bound::new(
      py,
      Expr {
        head,
        args: PyList::new(py, args)?.unbind(),
      },
    )?
    .into_any()
    .unbind(),
  )
}

fn expr_to_py_at(
  py: Python<'_>,
  expr: &WExpr,
  depth: usize,
) -> PyResult<Py<PyAny>> {
  if depth > MAX_DEPTH {
    return Err(too_deep());
  }
  // A complex number is stored as a Plus/Times tree but its head is
  // `Complex`, so it reports as `Complex[re, im]` — the same shape
  // FullForm shows.
  if let Some(parts) = complex_full_form_parts(expr) {
    let (re, im) = match parts {
      ComplexParts::Exact { re, im } => (rational(re), rational(im)),
      ComplexParts::Float { re, im } => (WExpr::Real(re), WExpr::Real(im)),
    };
    let head = symbol_obj(py, "Complex")?;
    return composite_obj(py, head, &[re, im], depth);
  }
  let obj = match expr {
    // Atoms that carry a value. `decompose_expr` renders these to display
    // text, which would throw the value away, so they are handled here.
    WExpr::Integer(n) => n.into_pyobject(py)?.into_any().unbind(),
    WExpr::BigInteger(n) => n.into_pyobject(py)?.into_any().unbind(),
    WExpr::Real(f) => f.into_pyobject(py)?.into_any().unbind(),
    WExpr::BigFloat(digits, precision) => Bound::new(
      py,
      BigReal {
        digits: digits.clone(),
        precision: *precision,
      },
    )?
    .into_any()
    .unbind(),
    WExpr::String(s) => s.into_pyobject(py)?.into_any().unbind(),
    // A Wolfram list is a Python list. `Expr(Symbol("List"), […])` is
    // accepted on the way back in, so this round-trips.
    WExpr::List(items) => {
      let items = items
        .iter()
        .map(|i| expr_to_py_at(py, i, depth + 1))
        .collect::<PyResult<Vec<_>>>()?;
      PyList::new(py, items)?.into_any().unbind()
    }
    // Heads that are themselves expressions. `decompose_expr` flattens
    // these to a rendered string, which loses the head's structure.
    WExpr::CurriedCall { func, args } => {
      let head = expr_to_py_at(py, func, depth + 1)?;
      composite_obj(py, head, args, depth)?
    }
    WExpr::PrefixApply { func, arg } => {
      let head = expr_to_py_at(py, func, depth + 1)?;
      composite_obj(py, head, std::slice::from_ref(arg.as_ref()), depth)?
    }
    WExpr::Postfix { expr, func } => {
      let head = expr_to_py_at(py, func, depth + 1)?;
      composite_obj(py, head, std::slice::from_ref(expr.as_ref()), depth)?
    }
    // Rendered output. A graphic that kept its symbolic form reports that
    // form; otherwise it is an opaque, output-only placeholder.
    WExpr::Graphics {
      structure: Some(structure),
      ..
    } => expr_to_py_at(py, structure, depth + 1)?,
    WExpr::Graphics { svg, is_3d, .. } => Bound::new(
      py,
      Graphics {
        svg: svg.clone(),
        is_3d: *is_3d,
      },
    )?
    .into_any()
    .unbind(),
    WExpr::Image {
      width,
      height,
      channels,
      ..
    } => Bound::new(
      py,
      Image {
        width: *width,
        height: *height,
        channels: *channels,
      },
    )?
    .into_any()
    .unbind(),
    // Everything else — operator forms, patterns, rules, associations,
    // comparison chains — goes through the canonical FullForm
    // decomposition rather than being re-derived here.
    other => match decompose_expr(other) {
      ExprForm::Atom(name) => symbol_obj(py, &name)?,
      ExprForm::Composite { head, children } => {
        let head = symbol_obj(py, &head)?;
        composite_obj(py, head, &children, depth)?
      }
    },
  };
  Ok(obj)
}

/// Convert a Python tree to an interpreter expression.
pub fn py_to_expr(obj: &Bound<'_, PyAny>) -> PyResult<WExpr> {
  py_to_expr_at(obj, 0)
}

fn py_to_expr_at(obj: &Bound<'_, PyAny>, depth: usize) -> PyResult<WExpr> {
  if depth > MAX_DEPTH {
    return Err(too_deep());
  }
  if obj.is_none() {
    return Ok(WExpr::Identifier("Null".to_string()));
  }
  if let Ok(sym) = obj.cast::<Symbol>() {
    return Ok(WExpr::Identifier(sym.get().name.clone()));
  }
  if let Ok(expr) = obj.cast::<Expr>() {
    return expr_node_to_expr(expr.get(), obj.py(), depth);
  }
  if let Ok(big) = obj.cast::<BigReal>() {
    let big = big.get();
    return Ok(WExpr::BigFloat(big.digits.clone(), big.precision));
  }
  // `bool` is a subclass of `int`, so it has to be tested first.
  if let Ok(b) = obj.cast::<PyBool>() {
    return Ok(WExpr::Identifier(
      if b.is_true() { "True" } else { "False" }.to_string(),
    ));
  }
  if let Ok(int) = obj.cast::<PyInt>() {
    return Ok(match int.extract::<i128>() {
      Ok(n) => WExpr::Integer(n),
      Err(_) => WExpr::BigInteger(int.extract::<BigInt>()?),
    });
  }
  if let Ok(f) = obj.cast::<PyFloat>() {
    return Ok(WExpr::Real(f.extract::<f64>()?));
  }
  if let Ok(s) = obj.cast::<PyString>() {
    return Ok(WExpr::String(s.extract::<String>()?));
  }
  if obj.cast::<PyList>().is_ok() || obj.cast::<PyTuple>().is_ok() {
    let items = obj
      .try_iter()?
      .map(|i| py_to_expr_at(&i?, depth + 1))
      .collect::<PyResult<Vec<_>>>()?;
    return Ok(WExpr::List(items.into()));
  }
  // The remaining cases are what `to_python()` produces, so its output can
  // be handed straight back to `evaluate_expr`.
  if let Ok(dict) = obj.cast::<PyDict>() {
    let mut pairs = Vec::with_capacity(dict.len());
    for (k, v) in dict.iter() {
      pairs
        .push((py_to_expr_at(&k, depth + 1)?, py_to_expr_at(&v, depth + 1)?));
    }
    return Ok(WExpr::Association(pairs));
  }
  if let Ok(c) = obj.cast::<PyComplex>() {
    return Ok(WExpr::FunctionCall {
      name: "Complex".to_string(),
      args: vec![WExpr::Real(c.real()), WExpr::Real(c.imag())].into(),
    });
  }
  if let (Ok(num), Ok(den)) =
    (obj.getattr("numerator"), obj.getattr("denominator"))
  {
    return Ok(WExpr::FunctionCall {
      name: "Rational".to_string(),
      args: vec![
        py_to_expr_at(&num, depth + 1)?,
        py_to_expr_at(&den, depth + 1)?,
      ]
      .into(),
    });
  }
  Err(PyTypeError::new_err(format!(
    "cannot convert {} to a Wolfram expression",
    obj.get_type().name()?
  )))
}

fn expr_node_to_expr(
  node: &Expr,
  py: Python<'_>,
  depth: usize,
) -> PyResult<WExpr> {
  let args = node
    .args
    .bind(py)
    .iter()
    .map(|a| py_to_expr_at(&a, depth + 1))
    .collect::<PyResult<Vec<_>>>()?;
  let head = node.head.bind(py);
  // A symbolic head is an ordinary function call; anything else — a curried
  // head like `f[a][b]` — keeps its structure.
  if let Ok(sym) = head.cast::<Symbol>() {
    let name = sym.get().name.clone();
    // `Expr(Symbol("List"), […])` is the FullForm spelling of a list.
    if name == "List" {
      return Ok(WExpr::List(args.into()));
    }
    return Ok(WExpr::FunctionCall {
      name,
      args: args.into(),
    });
  }
  Ok(WExpr::CurriedCall {
    func: Box::new(py_to_expr_at(head, depth + 1)?),
    args,
  })
}
