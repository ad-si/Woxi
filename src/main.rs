use clap::{Parser, Subcommand};
use wolfram_parser::interpret;

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
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Eval { expression } => {
            match interpret(&expression) {
                Ok(result) => println!("{result}"),
                Err(e) => eprintln!("Error: {}", e),
            }
        }
    }
}
