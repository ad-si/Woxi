use clap::{Parser, Subcommand};
mod jupyter;
mod repl;
use std::env;
use std::fs;
use std::path::PathBuf;
use woxi::notebook::{CellStyle, parse_notebook};
use woxi::{interpret, set_script_command_line, without_shebang};

/// Execute a script file referenced by `absolute_path`.
///
/// `.nb` notebook files are parsed and their Input/Code cells are
/// evaluated in order (printing each cell's result, like notebook
/// evaluation).  Every other file is treated as a plain Wolfram
/// Language script: the shebang is stripped and the whole content is
/// interpreted with the final expression's value suppressed.
fn run_script_file(absolute_path: &std::path::Path) {
  let content = match fs::read_to_string(absolute_path) {
    Ok(content) => content,
    Err(e) => {
      eprintln!("Error reading file: {}", e);
      return;
    }
  };

  let is_notebook = absolute_path
    .extension()
    .is_some_and(|ext| ext.eq_ignore_ascii_case("nb"));

  if is_notebook {
    // Running a `.nb` file: its directory is known, so make
    // `NotebookDirectory[]` resolve to it (e.g. for Export paths).
    if let Some(parent) = absolute_path.parent() {
      woxi::set_notebook_directory(Some(parent.to_string_lossy().into_owned()));
    }
    run_notebook(&content);
  } else {
    let code = without_shebang(&content);
    match interpret(&code) {
      Ok(_result) => {
        // Suppress automatic output of the final expression value when
        // running a script file.  Side-effects (Print[…]) have already
        // been written by the interpreter itself.
      }
      Err(e) => {
        eprintln!("Error interpreting file: {}", e);
        if let Some(trace) = woxi::take_error_trace() {
          eprintln!("{}", trace);
        }
      }
    }
  }
}

/// Parse a `.nb` notebook and evaluate its Input/Code cells in order,
/// printing the result of each cell that produces a value.
fn run_notebook(content: &str) {
  let notebook = match parse_notebook(content) {
    Ok(nb) => nb,
    Err(e) => {
      eprintln!("Error parsing notebook: {}", e);
      return;
    }
  };

  for (_group, cell) in notebook.flat_cells() {
    if !matches!(cell.style, CellStyle::Input | CellStyle::Code) {
      continue;
    }
    if cell.content.trim().is_empty() {
      continue;
    }
    match interpret(&cell.content) {
      Ok(result) => {
        // "\0" marks suppressed output (Null, trailing semicolon, …).
        if result != "\0" && !result.is_empty() {
          println!("{result}");
        }
      }
      Err(woxi::InterpreterError::EmptyInput) => {}
      Err(e) => {
        eprintln!("Error interpreting cell: {}", e);
        if let Some(trace) = woxi::take_error_trace() {
          eprintln!("{}", trace);
        }
      }
    }
  }
}

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
  #[command(subcommand)]
  command: Commands,
}

#[derive(Subcommand)]
enum Commands {
  /// Evaluate a Wolfram Language expression
  Eval {
    /// The Wolfram Language expression to evaluate. Pass `-` to read
    /// the expression from stdin instead — useful for inputs that
    /// exceed the shell's ARG_MAX (e.g. huge image NumericArrays).
    #[arg(allow_hyphen_values = true)]
    expression: String,
    /// Suppress Print output to stdout (Print still captured internally)
    #[arg(long)]
    quiet_print: bool,
  },
  /// Start an interactive REPL session
  Repl,
  /// Run a Wolfram Language file
  Run {
    /// The path to the Wolfram Language file to execute
    #[arg(value_name = "FILE")]
    file: PathBuf,
    /// Additional arguments passed to the script (accessible via $ScriptCommandLine)
    #[arg(trailing_var_arg = true)]
    args: Vec<String>,
  },
  /// Start a simple Jupyter kernel that always returns "hello world"
  Jupyter {
    /// Path to the Jupyter connection file (JSON) provided by the
    /// notebook frontend. Can be omitted.
    #[arg(value_name = "CONN_FILE")]
    connection_file: Option<PathBuf>,
  },
  /// Install Woxi as a Jupyter kernel
  InstallKernel {
    /// Install for the current user only (default)
    #[arg(long)]
    user: bool,
    /// Install system-wide (requires admin privileges)
    #[arg(long)]
    system: bool,
  },
  #[command(external_subcommand)]
  Script(Vec<String>), // invoked by a shebang:  woxi <file> [...]
}

fn install_kernel(user: bool, system: bool) -> std::io::Result<()> {
  let user_flag = if user {
    "--user"
  } else if system {
    "--system"
  } else {
    "--user"
  };

  // Get the path to the kernelspec directory
  let kernelspec_dir = env::current_dir()?.join("kernelspec/woxi");

  // Use jupyter kernelspec to install the kernel
  let status = std::process::Command::new("jupyter")
    .args([
      "kernelspec",
      "install",
      "--replace",
      user_flag,
      kernelspec_dir.to_str().unwrap(),
    ])
    .status()?;

  if status.success() {
    println!("Woxi kernel installed successfully!");
    println!(
      "You can now use it in Jupyter Lab or Notebook by selecting 'Woxi' from the kernel list."
    );
    Ok(())
  } else {
    Err(std::io::Error::other(format!(
      "Failed to install kernel. Exit code: {}",
      status
    )))
  }
}

fn main() {
  let cli = Cli::parse();
  // Run all work on a worker thread with a large stack. The interpreter and
  // its output formatters recurse over the expression tree; `stacker` grows the
  // stack on demand but cannot reliably measure the main thread's stack on some
  // platforms (macOS), so deeply nested inputs/results (e.g. rendering
  // Nest[List, {1}, 3000], which wolframscript handles) could overflow the base
  // 8 MB stack. A 512 MB reserved stack is virtual (paged in lazily) and lifts
  // the practical nesting depth well past anything wolframscript renders.
  let worker = std::thread::Builder::new()
    .stack_size(512 * 1024 * 1024)
    .spawn(move || run(cli))
    .expect("failed to spawn worker thread");
  match worker.join() {
    Ok(()) => {}
    // A panic on the worker thread already printed its message; exit non-zero.
    Err(_) => std::process::exit(101),
  }
}

fn run(cli: Cli) {
  match cli.command {
    Commands::Eval {
      expression,
      quiet_print,
    } => {
      if quiet_print {
        woxi::set_quiet_print(true);
      }
      woxi::set_messages_to_stdout(true);
      // Read from stdin when expression is `-`. Lets callers pass
      // huge inputs that would otherwise hit the shell's ARG_MAX
      // (the `Argument list too long` errors seen for some
      // image-heavy audit cases).
      let expression: String = if expression == "-" {
        use std::io::Read;
        let mut buf = String::new();
        if let Err(e) = std::io::stdin().read_to_string(&mut buf) {
          eprintln!("Error: failed to read stdin: {}", e);
          std::process::exit(1);
        }
        buf
      } else {
        expression
      };
      match interpret(&expression) {
        Ok(result) => {
          // "\0" is a sentinel for suppressed output (Null symbol, trailing semicolon,
          // or output already printed e.g. Part error).
          // In CLI mode, display "Null" to match wolframscript behavior.
          if result == "\0" {
            println!("Null");
          } else {
            println!("{result}");
          }
        }
        Err(woxi::InterpreterError::EmptyInput) => {
          // Comment-only or whitespace-only input: wolframscript
          // prints `Null` (the value of "nothing"). Truly empty
          // input ("") still suppresses output — only print Null
          // when the original `expression` is non-empty.
          if !expression.is_empty() {
            println!("Null");
          }
        }
        Err(e) => {
          eprintln!("Error: {}", e);
          if let Some(trace) = woxi::take_error_trace() {
            eprintln!("{}", trace);
          }
        }
      }
    }
    Commands::Repl => {
      repl::run();
    }
    Commands::Run { file, args } => {
      // `wolframscript -file` writes messages (e.g. `Get::noopen`,
      // `Power::infy`) to stdout, so match it: route diagnostics there too.
      woxi::set_messages_to_stdout(true);

      let absolute_path = if file.is_absolute() {
        file.clone()
      } else {
        env::current_dir()
          .unwrap_or_else(|_| PathBuf::from("."))
          .join(&file)
      };

      // Set $InputFileName and $ScriptCommandLine
      let abs_str = absolute_path.to_string_lossy().to_string();
      woxi::set_system_variable("$InputFileName", &format!("\"{}\"", abs_str));
      let mut cmd_line = vec![abs_str];
      cmd_line.extend(args);
      set_script_command_line(&cmd_line);

      run_script_file(&absolute_path);
    }
    Commands::Jupyter { connection_file } => {
      if let Err(e) = jupyter::run(connection_file.as_deref()) {
        eprintln!("Error starting Jupyter kernel: {e}");
      }
    }
    Commands::InstallKernel { user, system } => {
      if let Err(e) = install_kernel(user, system) {
        eprintln!("Error installing kernel: {e}");
      }
    }
    //  shebang / direct script execution  ---------------------------------
    Commands::Script(args) => {
      // Shebang scripts run like `wolframscript -file`; send messages to
      // stdout to match it (see the `Run` arm above).
      woxi::set_messages_to_stdout(true);

      if args.is_empty() {
        eprintln!("Error: no script file supplied");
        return;
      }
      let file = PathBuf::from(&args[0]);
      let absolute_path = if file.is_absolute() {
        file.clone()
      } else {
        env::current_dir()
          .unwrap_or_else(|_| PathBuf::from("."))
          .join(&file)
      };

      // Set $InputFileName and $ScriptCommandLine
      let abs_str = absolute_path.to_string_lossy().to_string();
      woxi::set_system_variable("$InputFileName", &format!("\"{}\"", abs_str));
      let mut cmd_line = vec![abs_str];
      cmd_line.extend(args.into_iter().skip(1));
      set_script_command_line(&cmd_line);

      run_script_file(&absolute_path);
    }
  }
}
