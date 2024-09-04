use pest::Parser;
use pest_derive::Parser;

#[derive(Parser)]
#[grammar = "wolfram.pest"]
pub struct WolframParser;

impl WolframParser {
    pub fn parse_wolfram(
        input: &str,
    ) -> Result<pest::iterators::Pairs<Rule>, pest::error::Error<Rule>> {
        Self::parse(Rule::Program, input)
    }
}

pub fn parse(input: &str) -> Result<pest::iterators::Pairs<Rule>, pest::error::Error<Rule>> {
    WolframParser::parse_wolfram(input)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculation() {
        let input = "1 + 2";
        let pair = parse(input).unwrap().next().unwrap();
        assert_eq!(pair.as_rule(), Rule::Program);
    }

    #[test]
    fn test_symbolic_calculation() {
        let input = "x + 2";
        let pair = parse(input).unwrap().next().unwrap();
        assert_eq!(pair.as_rule(), Rule::Program);
    }

    #[test]
    fn test_constant() {
        let input = "Sin[Pi]";
        let pair = parse(input).unwrap().next().unwrap();
        assert_eq!(pair.as_rule(), Rule::Program);
    }

    #[test]
    fn test_function_definition() {
        let input = "f[x_] := x^2 + 2*x + 1";
        let pair = parse(input).unwrap().next().unwrap();
        assert_eq!(pair.as_rule(), Rule::Program);
    }

    #[test]
    fn test_function_call() {
        let input = "Plot[f[x], {x, -2, 2}]";
        let pair = parse(input).unwrap().next().unwrap();
        assert_eq!(pair.as_rule(), Rule::Program);
    }

    #[test]
    fn test_complex_expression() {
        let input = "3*x^2 + 2*x + 1";
        let pair = parse(input).unwrap().next().unwrap();
        assert_eq!(pair.as_rule(), Rule::Program);
    }

    #[test]
    fn test_list() {
        let input = "{1, 2, 3, 4, 5}";
        let pair = parse(input).unwrap().next().unwrap();
        assert_eq!(pair.as_rule(), Rule::Program);
    }

    #[test]
    fn test_nested_function_calls() {
        let input = "Cos[Sin[x]]";
        let pair = parse(input).unwrap().next().unwrap();
        assert_eq!(pair.as_rule(), Rule::Program);
    }

    #[test]
    fn test_complex_nested_function_calls() {
        let input = "Plot[Sin[x], {x, -Pi, Pi}]";
        let pair = parse(input).unwrap().next().unwrap();
        assert_eq!(pair.as_rule(), Rule::Program);
    }
}
