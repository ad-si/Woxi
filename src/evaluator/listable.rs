#[allow(unused_imports)]
use super::*;

/// Dispatch function call to built-in implementations (AST version).
/// This is the AST equivalent of the string-based function dispatch.
/// IMPORTANT: This function must NOT call interpret() to avoid infinite recursion.
/// Built-in Listable functions (thread automatically over list arguments)
pub fn is_builtin_listable(name: &str) -> bool {
  matches!(
    name,
    "Fibonacci"
      | "LucasL"
      | "Sin"
      | "SinDegrees"
      | "Cos"
      | "CosDegrees"
      | "Tan"
      | "TanDegrees"
      | "Sec"
      | "SecDegrees"
      | "Csc"
      | "CscDegrees"
      | "Cot"
      | "CotDegrees"
      | "Sinh"
      | "Cosh"
      | "Tanh"
      | "Coth"
      | "Sech"
      | "Csch"
      | "ArcSin"
      | "ArcSinDegrees"
      | "ArcCos"
      | "ArcCosDegrees"
      | "ArcTan"
      | "ArcTanDegrees"
      | "ArcCot"
      | "ArcCotDegrees"
      | "ArcSec"
      | "ArcSecDegrees"
      | "ArcCsc"
      | "ArcCscDegrees"
      | "ArcSinh"
      | "ArcCosh"
      | "ArcTanh"
      | "Gudermannian"
      | "InverseGudermannian"
      | "Exp"
      | "Log"
      | "Log2"
      | "Log10"
      | "LogisticSigmoid"
      | "Abs"
      | "Sign"
      | "Floor"
      | "Ceiling"
      | "Round"
      | "Sqrt"
      | "Surd"
      | "Factorial"
      | "Subfactorial"
      | "Gamma"
      | "Erf"
      | "Erfc"
      | "Erfi"
      | "InverseErf"
      | "InverseErfc"
      | "Prime"
      | "Power"
      | "Plus"
      | "Times"
      | "Mod"
      | "Quotient"
      | "GCD"
      | "LCM"
      | "Binomial"
      | "Multinomial"
      | "IntegerDigits"
      | "FactorInteger"
      | "IntegerLength"
      | "RealDigits"
      | "RomanNumeral"
      | "EulerPhi"
      | "CarmichaelLambda"
      | "MoebiusMu"
      | "Divisors"
      | "DivisorSigma"
      | "BernoulliB"
      | "BellB"
      | "PrimePowerQ"
      | "CatalanNumber"
      | "StirlingS1"
      | "StirlingS2"
      | "HarmonicNumber"
      | "CoefficientList"
      | "ContinuedFraction"
      | "Boole"
      | "BitLength"
      | "EvenQ"
      | "OddQ"
      | "PrimeQ"
      | "Positive"
      | "Negative"
      | "NonPositive"
      | "NonNegative"
      | "StringLength"
      | "MixedFractionParts"
      | "SinhIntegral"
      | "CoshIntegral"
      | "FresnelS"
      | "FresnelC"
      | "BetaRegularized"
      | "GammaRegularized"
      | "Hypergeometric1F1Regularized"
      | "Unitize"
      | "Hyperfactorial"
      | "FiniteGroupCount"
      | "FiniteAbelianGroupCount"
      | "UnitStep"
      | "RealSign"
      | "RealAbs"
      | "Re"
      | "Im"
      | "Arg"
      | "Conjugate"
  )
}

pub fn is_builtin_flat(name: &str) -> bool {
  matches!(name, "Plus" | "Times" | "Max" | "Min" | "And" | "Or")
}

pub fn is_builtin_orderless(name: &str) -> bool {
  matches!(name, "Plus" | "Times" | "Max" | "Min" | "GCD" | "LCM")
}

/// Thread a Listable function over list arguments.
/// Returns Some(result) if threading was applied, None otherwise.
pub fn thread_listable(
  name: &str,
  args: &[Expr],
) -> Result<Option<Expr>, InterpreterError> {
  // Check if any argument is a list
  let has_list = args.iter().any(|a| matches!(a, Expr::List(_)));
  if !has_list {
    return Ok(None);
  }

  // Find the list length (all lists must have the same length)
  let mut list_len = None;
  for arg in args {
    if let Expr::List(items) = arg {
      match list_len {
        None => list_len = Some(items.len()),
        Some(n) if n != items.len() => {
          // Mismatched list lengths — don't thread, let the function handle it
          return Ok(None);
        }
        _ => {}
      }
    }
  }

  let len = match list_len {
    Some(n) => n,
    None => return Ok(None),
  };

  // Thread element-wise
  let mut results = Vec::with_capacity(len);
  for i in 0..len {
    let threaded_args: Vec<Expr> = args
      .iter()
      .map(|arg| {
        if let Expr::List(items) = arg {
          items[i].clone()
        } else {
          arg.clone()
        }
      })
      .collect();
    results.push(evaluate_function_call_ast(name, &threaded_args)?);
  }
  Ok(Some(Expr::List(results)))
}

/// Flatten Sequence[...] arguments into the parent function's argument list.
/// In Wolfram Language, Sequence[a, b] appearing as an argument to f produces f[..., a, b, ...].
/// Functions with the SequenceHold attribute suppress this.
/// Look up system $ variables
pub fn get_system_variable(name: &str) -> Option<Expr> {
  match name {
    "$RecursionLimit" => Some(Expr::Integer(256)),
    "$IterationLimit" => Some(Expr::Integer(4096)),
    "$MachinePrecision" => Some(Expr::Real(15.954589770191003)),
    "$MachineEpsilon" => Some(Expr::Real(2.220446049250313e-16)),
    "$MaxMachineNumber" => Some(Expr::Real(1.7976931348623157e308)),
    "$MinMachineNumber" => Some(Expr::Real(5e-324)),
    "$MaxPrecision" => Some(Expr::Identifier("Infinity".to_string())),
    "$MinPrecision" => Some(Expr::Integer(0)),
    "$SystemWordLength" => Some(Expr::Integer(usize::BITS as i128)),
    "$SessionID" => Some(Expr::Integer(std::process::id() as i128)),
    "$ProcessID" => Some(Expr::Integer(std::process::id() as i128)),
    #[cfg(unix)]
    "$ParentProcessID" => {
      Some(Expr::Integer(unsafe { libc::getppid() } as i128))
    }
    #[cfg(unix)]
    "$MachineName" => {
      let mut buf = [0u8; 256];
      let ret = unsafe {
        libc::gethostname(buf.as_mut_ptr() as *mut libc::c_char, buf.len())
      };
      if ret == 0 {
        let len = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
        // Strip trailing .local, .lan etc to match wolframscript which returns
        // the short name (e.g. "Mac" rather than "Mac.local").
        let host = std::str::from_utf8(&buf[..len]).unwrap_or("");
        let short = host.split('.').next().unwrap_or(host);
        Some(Expr::String(short.to_string()))
      } else {
        None
      }
    }
    "$UserName" => std::env::var("USER")
      .or_else(|_| std::env::var("USERNAME"))
      .ok()
      .map(Expr::String),
    "$VersionNumber" => {
      Some(Expr::String(env!("WOXI_GIT_VERSION").to_string()))
    }
    "$CommandLine" => {
      Some(Expr::List(std::env::args().map(Expr::String).collect()))
    }
    "$ScriptCommandLine" => {
      Some(Expr::List(std::env::args().map(Expr::String).collect()))
    }
    "$SystemID" => {
      // Match wolframscript's SystemID format: e.g. "MacOSX-ARM64", "Linux-x86-64"
      let os = if cfg!(target_os = "macos") {
        "MacOSX"
      } else if cfg!(target_os = "linux") {
        "Linux"
      } else if cfg!(target_os = "windows") {
        "Windows"
      } else {
        "Unknown"
      };
      let arch = if cfg!(target_arch = "aarch64") {
        "ARM64"
      } else if cfg!(target_arch = "x86_64") {
        "x86-64"
      } else if cfg!(target_arch = "x86") {
        "x86"
      } else {
        std::env::consts::ARCH
      };
      Some(Expr::String(format!("{}-{}", os, arch)))
    }
    "$OperatingSystem" => {
      // wolframscript returns "MacOSX", "Unix", or "Windows".
      let os = if cfg!(target_os = "macos") {
        "MacOSX"
      } else if cfg!(target_os = "linux") {
        "Unix"
      } else if cfg!(target_os = "windows") {
        "Windows"
      } else {
        "Unknown"
      };
      Some(Expr::String(os.to_string()))
    }
    "$PathnameSeparator" => {
      Some(Expr::String(std::path::MAIN_SEPARATOR.to_string()))
    }
    #[cfg(not(target_arch = "wasm32"))]
    "$HomeDirectory" => std::env::var("HOME")
      .or_else(|_| std::env::var("USERPROFILE"))
      .ok()
      .map(Expr::String),
    #[cfg(not(target_arch = "wasm32"))]
    "$TemporaryDirectory" => {
      // Canonicalize to match wolframscript's output on macOS
      // (/var/folders/... -> /private/var/folders/...) and strip trailing slash.
      let tmp = std::env::temp_dir();
      let canon = std::fs::canonicalize(&tmp).unwrap_or(tmp);
      let mut s = canon.to_string_lossy().into_owned();
      while s.len() > 1 && s.ends_with('/') {
        s.pop();
      }
      Some(Expr::String(s))
    }
    #[cfg(not(target_arch = "wasm32"))]
    "$InitialDirectory" => std::env::current_dir()
      .ok()
      .map(|p| Expr::String(p.to_string_lossy().into_owned())),
    "$RootDirectory" => {
      // Filesystem root: "/" on Unix/Mac, "C:\" (or similar) on Windows.
      let root = if cfg!(target_os = "windows") {
        "C:\\"
      } else {
        "/"
      };
      Some(Expr::String(root.to_string()))
    }
    "$ProcessorType" => {
      // wolframscript returns just the arch string, e.g. "ARM64" or "x86-64"
      let arch = if cfg!(target_arch = "aarch64") {
        "ARM64"
      } else if cfg!(target_arch = "x86_64") {
        "x86-64"
      } else if cfg!(target_arch = "x86") {
        "x86"
      } else {
        std::env::consts::ARCH
      };
      Some(Expr::String(arch.to_string()))
    }
    #[cfg(target_os = "macos")]
    "$SystemMemory" => {
      // sysctlbyname("hw.memsize") returns total physical memory in bytes
      let mut size: u64 = 0;
      let mut len = std::mem::size_of::<u64>();
      let name = std::ffi::CString::new("hw.memsize").ok()?;
      let ret = unsafe {
        libc::sysctlbyname(
          name.as_ptr(),
          &mut size as *mut u64 as *mut libc::c_void,
          &mut len,
          std::ptr::null_mut(),
          0,
        )
      };
      if ret == 0 {
        Some(Expr::Integer(size as i128))
      } else {
        None
      }
    }
    #[cfg(target_os = "linux")]
    "$SystemMemory" => {
      // Parse /proc/meminfo for MemTotal (kB) and convert to bytes
      let contents = std::fs::read_to_string("/proc/meminfo").ok()?;
      for line in contents.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
          let kb: u64 = rest.split_whitespace().next()?.parse().ok()?;
          return Some(Expr::Integer((kb * 1024) as i128));
        }
      }
      None
    }
    "$Assumptions" => Some(Expr::Identifier("True".to_string())),
    "$Context" => Some(Expr::String("Global`".to_string())),
    "$ContextPath" => Some(Expr::List(vec![
      Expr::String("System`".to_string()),
      Expr::String("Global`".to_string()),
    ])),
    "$ImportFormats" => Some(Expr::List(
      ["BMP", "CSV", "GIF", "JPEG", "PNG", "TIFF"]
        .iter()
        .map(|s| Expr::String((*s).to_string()))
        .collect(),
    )),
    "$ExportFormats" => Some(Expr::List(
      ["BMP", "GIF", "JPEG", "PDF", "PNG", "SVG", "TIFF", "XLSX"]
        .iter()
        .map(|s| Expr::String((*s).to_string()))
        .collect(),
    )),
    _ => None,
  }
}

pub fn flatten_sequences(name: &str, args: &[Expr]) -> Vec<Expr> {
  // Check for SequenceHold attribute
  let has_sequence_hold = matches!(
    name,
    "Set"
      | "SetDelayed"
      | "Rule"
      | "RuleDelayed"
      | "HoldComplete"
      | "MakeBoxes"
  ) || crate::FUNC_ATTRS.with(|m| {
    m.borrow()
      .get(name)
      .is_some_and(|attrs| attrs.contains(&"SequenceHold".to_string()))
  });

  if has_sequence_hold {
    return args.to_vec();
  }

  let mut result = Vec::with_capacity(args.len());
  let mut had_sequence = false;
  for arg in args {
    if let Expr::FunctionCall {
      name: seq_name,
      args: seq_args,
    } = arg
      && seq_name == "Sequence"
    {
      result.extend(seq_args.iter().cloned());
      had_sequence = true;
      continue;
    }
    // Splice[list, head] — splice when the enclosing function matches head
    if let Expr::FunctionCall {
      name: splice_name,
      args: splice_args,
    } = arg
      && splice_name == "Splice"
      && splice_args.len() == 2
      && matches!(&splice_args[1], Expr::Identifier(h) if h == name)
      && let Expr::List(items) = &splice_args[0]
    {
      result.extend(items.iter().cloned());
      had_sequence = true;
      continue;
    }
    result.push(arg.clone());
  }
  if had_sequence { result } else { args.to_vec() }
}
