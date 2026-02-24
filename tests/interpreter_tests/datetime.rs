use super::*;

mod absolute_time {
  use super::*;

  #[test]
  fn absolute_time_year() {
    assert_eq!(interpret("AbsoluteTime[{2000}]").unwrap(), "3155673600");
  }

  #[test]
  fn absolute_time_date_string() {
    assert_eq!(
      interpret("AbsoluteTime[\"6 June 1991\"]").unwrap(),
      "2885155200"
    );
  }

  #[test]
  fn absolute_time_format_spec() {
    // String format spec returns Real (Wolfram behavior)
    assert_eq!(
      interpret(
        "AbsoluteTime[{\"01/02/03\", {\"Day\", \"Month\", \"YearShort\"}}]"
      )
      .unwrap(),
      "3.2530464*^9"
    );
  }

  #[test]
  fn absolute_time_format_spec_dashes() {
    // String format spec returns Real (Wolfram behavior)
    assert_eq!(
      interpret(
        "AbsoluteTime[{\"6-6-91\", {\"Day\", \"Month\", \"YearShort\"}}]"
      )
      .unwrap(),
      "2.8851552*^9"
    );
  }

  #[test]
  fn absolute_time_full_date() {
    assert_eq!(
      interpret("AbsoluteTime[{1991, 6, 6, 12, 0, 0}]").unwrap(),
      "2885198400"
    );
  }
}

mod date_list {
  use super::*;

  #[test]
  fn date_list_epoch() {
    assert_eq!(interpret("DateList[0]").unwrap(), "{1900, 1, 1, 0, 0, 0.}");
  }

  #[test]
  fn date_list_year_2000() {
    assert_eq!(
      interpret("DateList[3155673600]").unwrap(),
      "{2000, 1, 1, 0, 0, 0.}"
    );
  }

  #[test]
  fn date_list_overflow_days() {
    assert_eq!(
      interpret("DateList[{2012, 1, 300., 10}]").unwrap(),
      "{2012, 10, 26, 10, 0, 0.}"
    );
  }

  #[test]
  fn date_list_date_string() {
    assert_eq!(
      interpret("DateList[\"31/10/1991\"]").unwrap(),
      "{1991, 10, 31, 0, 0, 0.}"
    );
  }

  #[test]
  fn date_list_format_spec() {
    assert_eq!(
      interpret(
        "DateList[{\"31/10/91\", {\"Day\", \"Month\", \"YearShort\"}}]"
      )
      .unwrap(),
      "{1991, 10, 31, 0, 0, 0.}"
    );
  }

  #[test]
  fn date_list_format_with_separators() {
    assert_eq!(
      interpret("DateList[{\"31 10/91\", {\"Day\", \" \", \"Month\", \"/\", \"YearShort\"}}]")
        .unwrap(),
      "{1991, 10, 31, 0, 0, 0.}"
    );
  }
}

mod date_plus {
  use super::*;

  #[test]
  fn date_plus_days() {
    assert_eq!(
      interpret("DatePlus[{2010, 2, 5}, 73]").unwrap(),
      "{2010, 4, 19}"
    );
  }

  #[test]
  fn date_plus_multi_unit() {
    assert_eq!(
      interpret("DatePlus[{2010, 2, 5}, {{8, \"Week\"}, {1, \"Day\"}}]")
        .unwrap(),
      "{2010, 4, 3}"
    );
  }
}

mod date_difference {
  use super::*;

  #[test]
  fn date_difference_days() {
    assert_eq!(
      interpret("DateDifference[{2042, 1, 4}, {2057, 1, 1}]").unwrap(),
      "Quantity[5476, Days]"
    );
  }

  #[test]
  fn date_difference_year() {
    assert_eq!(
      interpret("DateDifference[{1936, 8, 14}, {2000, 12, 1}, \"Year\"]")
        .unwrap(),
      "Quantity[64.2986301369863, Years]"
    );
  }

  #[test]
  fn date_difference_hour() {
    assert_eq!(
      interpret("DateDifference[{2010, 6, 1}, {2015, 1, 1}, \"Hour\"]")
        .unwrap(),
      "Quantity[40200, Hours]"
    );
  }
}

mod date_string {
  use super::*;

  #[test]
  fn date_string_custom_format() {
    assert_eq!(
      interpret(
        "DateString[{1991, 10, 31, 0, 0}, {\"Day\", \" \", \"MonthName\", \" \", \"Year\"}]"
      )
      .unwrap(),
      "31 October 1991"
    );
  }

  #[test]
  fn date_string_default_format() {
    assert_eq!(
      interpret("DateString[{2007, 4, 15, 0}]").unwrap(),
      "Sun 15 Apr 2007 00:00:00"
    );
  }

  #[test]
  fn date_string_day_name_format() {
    assert_eq!(
      interpret(
        "DateString[{1979, 3, 14}, {\"DayName\", \"  \", \"Month\", \"-\", \"YearShort\"}]"
      )
      .unwrap(),
      "Wednesday  03-79"
    );
  }

  #[test]
  fn date_string_fractional_day() {
    assert_eq!(
      interpret("DateString[{1991, 6, 6.5}]").unwrap(),
      "Thu 6 Jun 1991 12:00:00"
    );
  }
}
