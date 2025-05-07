use clap::{Parser, Subcommand};
use std::env;
use std::fs;
use std::path::PathBuf;
use woxi::interpret;

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
  },
  /// Run a Wolfram Language file
  Run {
    /// The path to the Wolfram Language file to execute
    #[arg(value_name = "FILE")]
    file: PathBuf,
  },
  #[command(external_subcommand)]
  Script(Vec<String>), // invoked by a shebang:  woxi <file> [...]
}

/// Remove a first line that starts with "#!" (shebang);
/// returns the remainder as a new `String`.
fn without_shebang(src: &str) -> String {
  if src.starts_with("#!") {
    src.lines().skip(1).collect::<Vec<_>>().join("\n")
  } else {
    src.to_owned()
  }
}

fn main() {
  let cli = Cli::parse();

  match cli.command {
    Commands::Eval { expression } => match interpret(&expression) {
      Ok(result) => println!("{result}"),
      Err(e) => eprintln!("Error: {}", e),
    },
    Commands::Run { file } => {
      let absolute_path = if file.is_absolute() {
        file.clone()
      } else {
        env::current_dir()
          .unwrap_or_else(|_| PathBuf::from("."))
          .join(&file)
      };

      match fs::read_to_string(&absolute_path) {
        Ok(content) => {
          let code = without_shebang(&content);
          match interpret(&code) {
            Ok(_result) => {
              // Suppress automatic output of the final expression value
              // when running a script file.  Side-effects (Print[…]) have
              // already been written by the interpreter itself.
            }
            Err(e) => eprintln!("Error interpreting file: {}", e),
          }
        }
        Err(e) => eprintln!("Error reading file: {}", e),
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

      match fs::read_to_string(&absolute_path) {
        Ok(content) => {
          let code = without_shebang(&content);
          match interpret(&code) {
            Ok(_result) => { /* suppress final value for shebang scripts */ }
            Err(e) => eprintln!("Error interpreting file: {}", e),
          }
        }
        Err(e) => eprintln!("Error reading file: {}", e),
      }
    }
  }
}
