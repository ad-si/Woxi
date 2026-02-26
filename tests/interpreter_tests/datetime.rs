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

/// Helper: extract {y, m, d} from "DateObject[{y, m, d}, Day]tz]"
fn parse_date_object_ymd(s: &str) -> Vec<i32> {
  let list_start = s.find('{').unwrap();
  let list_end = s.find('}').unwrap();
  let inner = &s[list_start + 1..list_end];
  inner.split(", ").map(|p| p.parse().unwrap()).collect()
}

mod today {
  use super::*;

  #[test]
  fn today_returns_date_object_with_day_granularity() {
    let result = interpret("Today").unwrap();
    assert!(
      result.starts_with("DateObject[{"),
      "Today should return a DateObject, got: {}",
      result
    );
    assert!(
      result.contains("}, Day]"),
      "Today should include Day granularity, got: {}",
      result
    );
  }

  #[test]
  fn today_has_three_date_components() {
    let result = interpret("Today").unwrap();
    let parts = parse_date_object_ymd(&result);
    assert_eq!(
      parts.len(),
      3,
      "Today should have 3 date components (y, m, d), got: {:?}",
      parts
    );
  }

  #[test]
  fn today_returns_valid_date() {
    let result = interpret("Today").unwrap();
    let parts = parse_date_object_ymd(&result);
    assert!(
      parts[0] >= 2020 && parts[0] <= 2100,
      "Year should be reasonable"
    );
    assert!(parts[1] >= 1 && parts[1] <= 12, "Month should be 1-12");
    assert!(parts[2] >= 1 && parts[2] <= 31, "Day should be 1-31");
  }
}

mod tomorrow {
  use super::*;

  #[test]
  fn tomorrow_returns_date_object_with_day_granularity() {
    let result = interpret("Tomorrow").unwrap();
    assert!(
      result.starts_with("DateObject[{"),
      "Tomorrow should return a DateObject, got: {}",
      result
    );
    assert!(
      result.contains("}, Day]"),
      "Tomorrow should include Day granularity, got: {}",
      result
    );
  }

  #[test]
  fn tomorrow_has_three_date_components() {
    let result = interpret("Tomorrow").unwrap();
    let parts = parse_date_object_ymd(&result);
    assert_eq!(
      parts.len(),
      3,
      "Tomorrow should have 3 date components (y, m, d), got: {:?}",
      parts
    );
  }

  #[test]
  fn tomorrow_is_after_today() {
    let today_parts = parse_date_object_ymd(&interpret("Today").unwrap());
    let tomorrow_parts = parse_date_object_ymd(&interpret("Tomorrow").unwrap());

    let today_days =
      today_parts[0] * 366 + today_parts[1] * 31 + today_parts[2];
    let tomorrow_days =
      tomorrow_parts[0] * 366 + tomorrow_parts[1] * 31 + tomorrow_parts[2];
    assert!(tomorrow_days > today_days, "Tomorrow should be after Today");
  }
}

mod yesterday {
  use super::*;

  #[test]
  fn yesterday_returns_date_object_with_day_granularity() {
    let result = interpret("Yesterday").unwrap();
    assert!(
      result.starts_with("DateObject[{"),
      "Yesterday should return a DateObject, got: {}",
      result
    );
    assert!(
      result.contains("}, Day]"),
      "Yesterday should include Day granularity, got: {}",
      result
    );
  }

  #[test]
  fn yesterday_has_three_date_components() {
    let result = interpret("Yesterday").unwrap();
    let parts = parse_date_object_ymd(&result);
    assert_eq!(
      parts.len(),
      3,
      "Yesterday should have 3 date components (y, m, d), got: {:?}",
      parts
    );
  }

  #[test]
  fn yesterday_is_before_today() {
    let today_parts = parse_date_object_ymd(&interpret("Today").unwrap());
    let yesterday_parts =
      parse_date_object_ymd(&interpret("Yesterday").unwrap());

    let today_days =
      today_parts[0] * 366 + today_parts[1] * 31 + today_parts[2];
    let yesterday_days =
      yesterday_parts[0] * 366 + yesterday_parts[1] * 31 + yesterday_parts[2];
    assert!(
      yesterday_days < today_days,
      "Yesterday should be before Today"
    );
  }
}
