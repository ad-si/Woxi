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
  fn date_list_date_object() {
    assert_eq!(
      interpret("DateList[DateObject[{2026, 3, 15}]]").unwrap(),
      "{2026, 3, 15, 0, 0, 0.}"
    );
  }

  #[test]
  fn date_list_date_object_with_time() {
    assert_eq!(
      interpret("DateList[DateObject[{2026, 3, 15, 10, 30, 45}]]").unwrap(),
      "{2026, 3, 15, 10, 30, 45.}"
    );
  }

  #[test]
  fn date_list_date_object_partial() {
    assert_eq!(
      interpret("DateList[DateObject[{2026, 3}]]").unwrap(),
      "{2026, 3, 1, 0, 0, 0.}"
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

  #[test]
  fn date_object_subtraction() {
    assert_eq!(
      interpret("DateObject[{2026, 3, 30}] - DateObject[{2016, 7, 23}]")
        .unwrap(),
      "Quantity[3537, Days]"
    );
  }

  #[test]
  fn date_difference_with_date_objects() {
    assert_eq!(
      interpret(
        "DateDifference[DateObject[{2020, 1, 1}], DateObject[{2020, 1, 11}]]"
      )
      .unwrap(),
      "Quantity[10, Days]"
    );
  }

  #[test]
  fn date_object_normalizes_granularity() {
    assert_eq!(
      interpret("DateObject[{2016, 7, 23}]").unwrap(),
      "DateObject[{2016, 7, 23}, Day]"
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

  #[test]
  fn date_string_iso_datetime() {
    assert_eq!(
      interpret("DateString[{2026, 2, 27, 19, 54, 40}, \"ISODateTime\"]")
        .unwrap(),
      "2026-02-27T19:54:40"
    );
  }

  #[test]
  fn date_string_iso_date() {
    assert_eq!(
      interpret("DateString[{2026, 2, 27, 19, 54, 40}, \"ISODate\"]").unwrap(),
      "2026-02-27"
    );
  }

  #[test]
  fn date_string_datetime() {
    assert_eq!(
      interpret("DateString[{2026, 2, 27, 20, 5, 43}, \"DateTime\"]").unwrap(),
      "Friday 27 February 2026 20:05:43"
    );
  }

  #[test]
  fn date_string_datetime_short() {
    assert_eq!(
      interpret("DateString[{2026, 2, 27, 20, 5, 43}, \"DateTimeShort\"]")
        .unwrap(),
      "Fri 27 Feb 2026 20:05:43"
    );
  }

  #[test]
  fn date_string_date() {
    assert_eq!(
      interpret("DateString[{2026, 2, 27}, \"Date\"]").unwrap(),
      "Friday 27 February 2026"
    );
  }

  #[test]
  fn date_string_date_short() {
    assert_eq!(
      interpret("DateString[{2026, 2, 27}, \"DateShort\"]").unwrap(),
      "Fri 27 Feb 2026"
    );
  }

  #[test]
  fn date_string_time() {
    assert_eq!(
      interpret("DateString[{2026, 2, 27, 14, 30, 15}, \"Time\"]").unwrap(),
      "14:30:15"
    );
  }

  #[test]
  fn date_string_single_element_year() {
    assert_eq!(
      interpret("DateString[{2026, 2, 27}, \"Year\"]").unwrap(),
      "2026"
    );
  }

  #[test]
  fn date_string_single_element_month_name() {
    assert_eq!(
      interpret("DateString[{2026, 2, 27}, \"MonthName\"]").unwrap(),
      "February"
    );
  }

  #[test]
  fn date_string_with_date_object() {
    assert_eq!(
      interpret("DateString[DateObject[{2026, 2, 27, 19, 54, 40}, \"Instant\", \"Gregorian\", 0.], \"ISODateTime\"]").unwrap(),
      "2026-02-27T19:54:40"
    );
  }

  #[test]
  fn date_string_now_iso_datetime() {
    let result = interpret("DateString[Now, \"ISODateTime\"]").unwrap();
    // Should match YYYY-MM-DDTHH:MM:SS pattern
    assert!(
      result.len() == 19
        && result.contains("T")
        && result.contains("-")
        && result.contains(":"),
      "DateString[Now, \"ISODateTime\"] should return ISO format, got: {}",
      result
    );
  }

  #[test]
  fn date_string_from_string_date() {
    assert_eq!(
      interpret("DateString[\"2025-09-24\", {\"Year\", \"-\", \"Month\"}]")
        .unwrap(),
      "2025-09"
    );
  }

  #[test]
  fn date_string_from_string_date_returns_as_is() {
    // DateString["string"] with no format spec returns the string unchanged
    assert_eq!(
      interpret("DateString[\"2025-09-24\"]").unwrap(),
      "2025-09-24"
    );
    assert_eq!(
      interpret("DateString[\"6 June 1991\"]").unwrap(),
      "6 June 1991"
    );
    assert_eq!(
      interpret("DateString[\"March 5, 2025\"]").unwrap(),
      "March 5, 2025"
    );
  }

  #[test]
  fn date_string_from_string_date_iso() {
    assert_eq!(
      interpret("DateString[\"2025-09-24\", \"ISODate\"]").unwrap(),
      "2025-09-24"
    );
  }

  #[test]
  fn date_string_from_natural_language_date() {
    // "Day" gives zero-padded day; "Day2" is not a Wolfram element (treated as literal)
    assert_eq!(
      interpret("DateString[\"6 June 1991\", {\"Year\", \"-\", \"Month\", \"-\", \"Day\"}]")
        .unwrap(),
      "1991-06-06"
    );
  }

  #[test]
  fn date_string_day2_is_literal() {
    // "Day2" is not a recognized format element — treated as literal text
    assert_eq!(
      interpret("DateString[\"6 June 1991\", {\"Year\", \"-\", \"Month\", \"-\", \"Day2\"}]")
        .unwrap(),
      "1991-06-Day2"
    );
  }
}

/// Helper: extract {y, m, d} from "DateObject[{y, m, d}, \"Day\"]"
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

mod day_name {
  use super::*;

  #[test]
  fn monday() {
    assert_eq!(interpret("DayName[{2024, 1, 1}]").unwrap(), "Monday");
  }

  #[test]
  fn thursday() {
    assert_eq!(interpret("DayName[{2026, 3, 19}]").unwrap(), "Thursday");
  }

  #[test]
  fn saturday() {
    assert_eq!(interpret("DayName[{2000, 1, 1}]").unwrap(), "Saturday");
  }

  #[test]
  fn unix_epoch() {
    assert_eq!(interpret("DayName[{1970, 1, 1}]").unwrap(), "Thursday");
  }

  #[test]
  fn with_date_object() {
    assert_eq!(
      interpret("DayName[DateObject[{2024, 6, 15}]]").unwrap(),
      "Saturday"
    );
  }

  #[test]
  fn sunday() {
    assert_eq!(interpret("DayName[{2024, 3, 3}]").unwrap(), "Sunday");
  }
}

mod day_plus {
  use super::*;

  #[test]
  fn add_days() {
    assert_eq!(
      interpret("DayPlus[{2024, 1, 15}, 10]").unwrap(),
      "DateObject[{2024, 1, 25}, Day]"
    );
  }

  #[test]
  fn subtract_days() {
    assert_eq!(
      interpret("DayPlus[{2024, 1, 15}, -5]").unwrap(),
      "DateObject[{2024, 1, 10}, Day]"
    );
  }

  #[test]
  fn leap_year() {
    assert_eq!(
      interpret("DayPlus[{2024, 2, 28}, 1]").unwrap(),
      "DateObject[{2024, 2, 29}, Day]"
    );
  }

  #[test]
  fn cross_month() {
    assert_eq!(
      interpret("DayPlus[{2024, 1, 31}, 1]").unwrap(),
      "DateObject[{2024, 2, 1}, Day]"
    );
  }

  #[test]
  fn business_days() {
    assert_eq!(
      interpret(r#"DayPlus[{2024, 1, 15}, 10, "BusinessDay"]"#).unwrap(),
      "DateObject[{2024, 1, 29}, Day]"
    );
  }
}

mod dated {
  use super::*;

  #[test]
  fn dated_basic() {
    assert_eq!(
      interpret("Dated[100, {2015, 3, 1}]").unwrap(),
      "Dated[100, {2015, 3, 1}]"
    );
  }

  #[test]
  fn dated_head() {
    assert_eq!(
      interpret("Head[Dated[100, {2015, 3, 1}]]").unwrap(),
      "Dated"
    );
  }

  #[test]
  fn dated_part_extraction() {
    assert_eq!(interpret("Dated[100, {2015, 3, 1}][[1]]").unwrap(), "100");
    assert_eq!(
      interpret("Dated[100, {2015, 3, 1}][[2]]").unwrap(),
      "{2015, 3, 1}"
    );
  }

  #[test]
  fn dated_string_value() {
    assert_eq!(
      interpret("Dated[\"hello\", {2020}]").unwrap(),
      "Dated[hello, {2020}]"
    );
  }
}
