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
        Ok(content) => match interpret(&content) {
          Ok(result) => println!("{result}"),
          Err(e) => eprintln!("Error interpreting file: {}", e),
        },
        Err(e) => eprintln!("Error reading file: {}", e),
      }
    }
  }
}
