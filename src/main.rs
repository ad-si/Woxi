use pest::Parser;
use pest_derive::Parser;

#[derive(Parser)]
#[grammar = "wolfram.pest"]
pub struct WolframParser;

pub fn parse(input: &str) -> Result<pest::iterators::Pairs<Rule>, pest::error::Error<Rule>> {
    WolframParser::parse(Rule::Program, input)
}

fn main() {
    let input = r#"
        f[x_] := x^2 + 2*x + 1
        Plot[f[x], {x, -2, 2}]
    "#;

    match parse(input) {
        Ok(pairs) => {
            for pair in pairs {
                print_pair(pair, 0);
            }
        }
        Err(e) => {
            eprintln!("Error: {}", e);
            eprintln!("Input:\n{}", input);
        },
    }
}

fn print_pair(pair: pest::iterators::Pair<Rule>, indent: usize) {
    let indent_str = " ".repeat(indent * 2);
    println!("{}Rule: {:?}", indent_str, pair.as_rule());
    println!("{}Span: {:?}", indent_str, pair.as_span());
    println!("{}Text: {}", indent_str, pair.as_str().trim());

    for inner_pair in pair.into_inner() {
        print_pair(inner_pair, indent + 1);
    }
}
