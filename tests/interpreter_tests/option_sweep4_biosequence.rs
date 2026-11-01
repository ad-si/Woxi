// String functions applied to `BioSequence` objects, found missing by the
// reference-page parameter sweep (round 4).

use super::*;

mod string_join {
  use super::*;

  #[test]
  fn works_on_biosequence() {
    assert_eq!(
      interpret(
        "StringJoin[BioSequence[\"DNA\", \"A\"], BioSequence[\"DNA\", \"C\"]]"
      )
      .unwrap(),
      "BioSequence[DNA, AC]"
    );
  }
}

mod string_replace {
  use super::*;

  #[test]
  fn works_on_biosequence() {
    assert_eq!(
      interpret(
        "StringReplace[BioSequence[\"DNA\", \"ATGGCGGT\"], \"GG\" -> \"TC\"]"
      )
      .unwrap(),
      "BioSequence[DNA, ATTCCTCT]"
    );
    assert_eq!(
      interpret(
        "StringReplace[BioSequence[\"DNA\", \"GTTGGCTTG\"], \"GGN\" -> \"A\"]"
      )
      .unwrap(),
      "BioSequence[DNA, GTTATTG]"
    );
    assert_eq!(
      interpret("StringReplace[BioSequence[\"DNA\", \"GTTGGCTTG\"], \"GG\" ~~ Verbatim[\"N\"] -> \"A\"]").unwrap(),
      "BioSequence[DNA, GTTGGCTTG]"
    );
  }
}

mod string_length {
  use super::*;

  #[test]
  fn works_on_biosequence() {
    assert_eq!(
      interpret("StringLength[BioSequence[\"DNA\", \"CTTCGACTACCAGCGGGTCGAACGGCCCATGGCATC\"]]").unwrap(),
      "36"
    );
    assert_eq!(
      interpret("StringLength[BioSequence[\"CircularDNA\", \"CGGTTA\"]]")
        .unwrap(),
      "6"
    );
  }
}

mod string_take_drop {
  use super::*;

  #[test]
  fn works_on_biosequence() {
    assert_eq!(
      interpret("StringTake[BioSequence[\"DNA\", \"GTCTCGCGTAGC\"], {4, 6}]")
        .unwrap(),
      "BioSequence[DNA, TCG]"
    );
    assert_eq!(
      interpret("StringDrop[BioSequence[\"DNA\", \"GTCTCGCGTAGC\"], {4, 6}]")
        .unwrap(),
      "BioSequence[DNA, GTCCGTAGC]"
    );
    assert_eq!(
      interpret(
        "StringTakeDrop[BioSequence[\"DNA\", \"GTCTCGCGTAGC\"], {4, 6}]"
      )
      .unwrap(),
      "{BioSequence[DNA, TCG], BioSequence[DNA, GTCCGTAGC]}"
    );
  }
}

mod string_match_q {
  use super::*;

  #[test]
  fn works_on_biosequence() {
    assert_eq!(
      interpret("StringMatchQ[BioSequence[\"DNA\", \"ATGGC\"], \"AN\" ~~ ___]")
        .unwrap(),
      "True"
    );
    assert_eq!(
      interpret("StringMatchQ[BioSequence[\"DNA\", \"ATGGC\"], ___ ~~ Verbatim[\"N\"] ~~ ___]").unwrap(),
      "False"
    );
    assert_eq!(
      interpret("StringMatchQ[BioSequence[\"DNA\", \"ATGNC\"], ___ ~~ Verbatim[\"N\"] ~~ ___]").unwrap(),
      "True"
    );
  }
}

mod string_split {
  use super::*;

  #[test]
  fn works_on_biosequence() {
    assert_eq!(
      interpret(
        "StringSplit[BioSequence[\"DNA\", \"GATGGCAAGGCAG\"], \"GGC\"]"
      )
      .unwrap(),
      "{BioSequence[DNA, GAT], BioSequence[DNA, AA], BioSequence[DNA, AG]}"
    );
    assert_eq!(
      interpret(
        "StringSplit[BioSequence[\"DNA\", \"GATGGCAAGGNAG\"], \"GGN\"]"
      )
      .unwrap(),
      "{BioSequence[DNA, GAT], BioSequence[DNA, AA], BioSequence[DNA, AG]}"
    );
    assert_eq!(
      interpret("StringSplit[BioSequence[\"DNA\", \"GATGGCAAGGNAG\"], \"GG\" ~~ Verbatim[\"N\"]]").unwrap(),
      "{BioSequence[DNA, GATGGCAA], BioSequence[DNA, AG]}"
    );
  }
}

mod string_free_q {
  use super::*;

  #[test]
  fn works_on_biosequence() {
    assert_eq!(
      interpret(
        "StringFreeQ[BioSequence[\"DNA\", \"TCCCATCAATACCAC\"], \"CTT\"]"
      )
      .unwrap(),
      "True"
    );
    assert_eq!(
      interpret("StringFreeQ[BioSequence[\"DNA\", \"ATGGC\"], \"AN\"]")
        .unwrap(),
      "False"
    );
    assert_eq!(
      interpret(
        "StringFreeQ[BioSequence[\"DNA\", \"ATGGC\"], \"A\" ~~ Verbatim[\"N\"]]"
      )
      .unwrap(),
      "True"
    );
    assert_eq!(
      interpret(
        "StringFreeQ[BioSequence[\"DNA\", \"ANGGC\"], \"A\" ~~ Verbatim[\"N\"]]"
      )
      .unwrap(),
      "False"
    );
  }
}

mod string_position {
  use super::*;

  #[test]
  fn works_on_biosequence() {
    assert_eq!(
      interpret(
        "StringPosition[BioSequence[\"DNA\", \"GTCTCGCGTAGC\"], \"TCG\"]"
      )
      .unwrap(),
      "{{4, 6}}"
    );
    assert_eq!(
      interpret("StringPosition[BioSequence[\"DNA\", \"ATGGCGG\"], \"GGN\"]")
        .unwrap(),
      "{{3, 5}}"
    );
    assert_eq!(
      interpret("StringPosition[BioSequence[\"DNA\", \"ATGGCGGN\"], \"GG\" ~~ Verbatim[\"N\"]]").unwrap(),
      "{{6, 8}}"
    );
  }
}

mod string_reverse {
  use super::*;

  #[test]
  fn works_on_biosequence() {
    assert_eq!(
      interpret("StringReverse[BioSequence[\"DNA\", \"TCCGAA\"]]").unwrap(),
      "BioSequence[DNA, AAGCCT]"
    );
  }
}

mod string_contains_q {
  use super::*;

  #[test]
  fn works_on_biosequence() {
    assert_eq!(
      interpret(
        "StringContainsQ[BioSequence[\"DNA\", \"CGGTGGAAC\"], \"TGG\"]"
      )
      .unwrap(),
      "True"
    );
    assert_eq!(
      interpret("StringContainsQ[BioSequence[\"DNA\", \"ATGGC\"], \"GNC\"]")
        .unwrap(),
      "True"
    );
    assert_eq!(
      interpret("StringContainsQ[BioSequence[\"DNA\", \"ATGGC\"], \"G\" ~~ Verbatim[\"N\"] ~~ \"C\"]").unwrap(),
      "False"
    );
    assert_eq!(
      interpret("StringContainsQ[BioSequence[\"DNA\", \"ATGNC\"], \"G\" ~~ Verbatim[\"N\"] ~~ \"C\"]").unwrap(),
      "True"
    );
  }
}

mod string_insert {
  use super::*;

  #[test]
  fn works_on_biosequence() {
    assert_eq!(
      interpret(
        "StringInsert[BioSequence[\"DNA\", \"GCCCGAGAC\"], \"TTT\", 2]"
      )
      .unwrap(),
      "BioSequence[DNA, GTTTCCCGAGAC]"
    );
  }
}

mod string_delete {
  use super::*;

  #[test]
  fn works_on_biosequence() {
    assert_eq!(
      interpret(
        "StringDelete[BioSequence[\"DNA\", \"CACACGTCGACT\"], \"CGA\"]"
      )
      .unwrap(),
      "BioSequence[DNA, CACACGTCT]"
    );
  }
}

mod string_replace_part {
  use super::*;

  #[test]
  fn works_on_biosequence() {
    assert_eq!(
      interpret("StringReplacePart[BioSequence[\"DNA\", \"GCCTGGAGG\"], \"GGT\", {4, 6}]").unwrap(),
      "BioSequence[DNA, GCCGGTAGG]"
    );
  }
}

mod string_starts_q {
  use super::*;

  #[test]
  fn works_on_biosequence() {
    assert_eq!(
      interpret(
        "StringStartsQ[BioSequence[\"DNA\", \"TGAGCTGTGTTCCCT\"], \"TGA\"]"
      )
      .unwrap(),
      "True"
    );
    assert_eq!(
      interpret("StringStartsQ[BioSequence[\"DNA\", \"ATGGC\"], \"AN\"]")
        .unwrap(),
      "True"
    );
    assert_eq!(
      interpret("StringStartsQ[BioSequence[\"DNA\", \"ATGGC\"], \"A\" ~~ Verbatim[\"N\"]]").unwrap(),
      "False"
    );
    assert_eq!(
      interpret("StringStartsQ[BioSequence[\"DNA\", \"ANGGC\"], \"A\" ~~ Verbatim[\"N\"]]").unwrap(),
      "True"
    );
  }
}

mod string_ends_q {
  use super::*;

  #[test]
  fn works_on_biosequence() {
    assert_eq!(
      interpret(
        "StringEndsQ[BioSequence[\"DNA\", \"TGAGCTGTGTTCCCT\"], \"CCT\"]"
      )
      .unwrap(),
      "True"
    );
    assert_eq!(
      interpret("StringEndsQ[BioSequence[\"DNA\", \"ATGGC\"], \"GS\"]")
        .unwrap(),
      "True"
    );
    assert_eq!(
      interpret(
        "StringEndsQ[BioSequence[\"DNA\", \"ATGGC\"], \"G\" ~~ Verbatim[\"S\"]]"
      )
      .unwrap(),
      "False"
    );
    assert_eq!(
      interpret(
        "StringEndsQ[BioSequence[\"DNA\", \"ATGGS\"], \"G\" ~~ Verbatim[\"S\"]]"
      )
      .unwrap(),
      "True"
    );
  }
}

mod string_repeat {
  use super::*;

  #[test]
  fn works_on_biosequence() {
    assert_eq!(
      interpret("StringRepeat[BioSequence[\"DNA\", \"GTT\"], 3]").unwrap(),
      "BioSequence[DNA, GTTGTTGTT]"
    );
  }
}

mod string_pad {
  use super::*;

  #[test]
  fn works_on_biosequence() {
    assert_eq!(
      interpret("StringPadLeft[BioSequence[\"DNA\", \"CGGAGT\"], 9, \"A\"]")
        .unwrap(),
      "BioSequence[DNA, AAACGGAGT]"
    );
    assert_eq!(
      interpret("StringPadRight[BioSequence[\"DNA\", \"CGGAGT\"], 9, \"A\"]")
        .unwrap(),
      "BioSequence[DNA, CGGAGTAAA]"
    );
  }
}

mod string_rotate {
  use super::*;

  #[test]
  fn works_on_biosequence() {
    assert_eq!(
      interpret("StringRotateRight[BioSequence[\"DNA\", \"CGGTCTNNN\"], 3]")
        .unwrap(),
      "BioSequence[DNA, NNNCGGTCT]"
    );
    assert_eq!(
      interpret("StringRotateLeft[BioSequence[\"DNA\", \"NNNCGGTCT\"], 3]")
        .unwrap(),
      "BioSequence[DNA, CGGTCTNNN]"
    );
  }
}

mod rna_degenerate_letters {
  use super::*;

  #[test]
  fn works_on_biosequence() {
    assert_eq!(
      interpret("StringCount[BioSequence[\"RNA\", \"AUGGCGGU\"], \"GGY\"]")
        .unwrap(),
      "2"
    );
    assert_eq!(
      interpret("StringCount[BioSequence[\"RNA\", \"AUGGCGGU\"], \"GGR\"]")
        .unwrap(),
      "0"
    );
  }
}
