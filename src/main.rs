use clap::{Parser, Subcommand};
mod jupyter;
use std::env;
use std::fs;
use std::path::PathBuf;
use woxi::{interpret, set_script_command_line, without_shebang};

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
    /// The Wolfram Language expression to evaluate
    expression: String,
    /// Suppress Print output to stdout (Print still captured internally)
    #[arg(long)]
    quiet_print: bool,
  },
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

  match cli.command {
    Commands::Eval {
      expression,
      quiet_print,
    } => {
      if quiet_print {
        woxi::set_quiet_print(true);
      }
      woxi::set_messages_to_stdout(true);
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
          // No output for empty/comment-only input
        }
        Err(e) => {
          eprintln!("Error: {}", e);
          if let Some(trace) = woxi::take_error_trace() {
            eprintln!("{}", trace);
          }
        }
      }
    }
    Commands::Run { file, args } => {
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

      match fs::read_to_string(&absolute_path) {
        Ok(content) => {
          let code = without_shebang(&content);
          match interpret(&code) {
            Ok(_result) => {
              // Suppress automatic output of the final expression value
              // when running a script file.  Side-effects (Print[…]) have
              // already been written by the interpreter itself.
            }
            Err(e) => {
              eprintln!("Error interpreting file: {}", e);
              if let Some(trace) = woxi::take_error_trace() {
                eprintln!("{}", trace);
              }
            }
          }
        }
        Err(e) => eprintln!("Error reading file: {}", e),
      }
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

      match fs::read_to_string(&absolute_path) {
        Ok(content) => {
          let code = without_shebang(&content);
          match interpret(&code) {
            Ok(_result) => { /* suppress final value for shebang scripts */ }
            Err(e) => {
              eprintln!("Error interpreting file: {}", e);
              if let Some(trace) = woxi::take_error_trace() {
                eprintln!("{}", trace);
              }
            }
          }
        }
        Err(e) => eprintln!("Error reading file: {}", e),
      }
    }
  }
}
