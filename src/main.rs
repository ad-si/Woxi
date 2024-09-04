use clap::Parser;
use wolfram_parser::interpret;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
  /// The Wolfram Language expression to evaluate
  expression: String,
}

fn main() {
  let cli = Cli::parse();

  match interpret(&cli.expression) {
    Ok(result) => println!("{result}"),
    Err(e) => eprintln!("Error: {}", e),
  }
}
