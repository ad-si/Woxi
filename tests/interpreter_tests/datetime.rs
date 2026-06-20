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
  fn date_plus_date_object_returns_date_object() {
    assert_eq!(
      interpret("FullForm[DatePlus[DateObject[{2026, 4, 1}, \"Day\"], 7]]")
        .unwrap(),
      "FullForm[DateObject[{2026, 4, 8}, \"Day\"]]"
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
  #[test]
  fn date_plus_multi_unit_applies_every_increment() {
    // Regression: a multi-increment spec must apply ALL pairs in order, not
    // just the first. {+1 Month, -1 Day} from the 1st is the month's last day.
    assert_eq!(
      interpret("DatePlus[{2013, 8}, {{1, \"Month\"}, {-1, \"Day\"}}]")
        .unwrap(),
      "{2013, 8, 31}"
    );
    assert_eq!(
      interpret("DatePlus[{2013, 2}, {{1, \"Month\"}, {-1, \"Day\"}}]")
        .unwrap(),
      "{2013, 2, 28}"
    );
  }
  #[test]
  fn date_plus_single_month_pair() {
    // Regression: a single {n, "unit"} pair (not wrapped in a list) must be
    // applied; Jan 31 + 1 month clamps to Feb 29 in a leap year.
    assert_eq!(
      interpret("DatePlus[{2020, 1, 31}, {1, \"Month\"}]").unwrap(),
      "{2020, 2, 29}"
    );
    assert_eq!(
      interpret("DatePlus[{2013, 12, 15}, {1, \"Year\"}]").unwrap(),
      "{2014, 12, 15}"
    );
  }
  #[test]
  fn date_plus_plural_units() {
    // A {n, unit} pair accepts the plural unit spelling, like Quantity does.
    assert_eq!(
      interpret("DatePlus[{2024, 1, 1}, {2, \"Months\"}]").unwrap(),
      "{2024, 3, 1}"
    );
    assert_eq!(
      interpret("DatePlus[{2024, 1, 15}, {3, \"Days\"}]").unwrap(),
      "{2024, 1, 18}"
    );
    // Month overflow rolls into the next year.
    assert_eq!(
      interpret("DatePlus[{2024, 11, 1}, {3, \"Months\"}]").unwrap(),
      "{2025, 2, 1}"
    );
    assert_eq!(
      interpret("DatePlus[{2024, 1, 1}, {1, \"Years\"}]").unwrap(),
      "{2025, 1, 1}"
    );
    assert_eq!(
      interpret("DatePlus[{2024, 1, 1}, {2, \"Weeks\"}]").unwrap(),
      "{2024, 1, 15}"
    );
  }
  #[test]
  fn date_plus_quantity_days() {
    assert_eq!(
      interpret("FullForm[DatePlus[DateObject[{2026, 4, 1}, \"Day\"], Quantity[7, \"Days\"]]]")
        .unwrap(),
      "FullForm[DateObject[{2026, 4, 8}, \"Day\"]]"
    );
  }

  #[test]
  fn date_plus_quantity_weeks() {
    assert_eq!(
      interpret("FullForm[DatePlus[DateObject[{2026, 4, 1}, \"Day\"], Quantity[4, \"Weeks\"]]]")
        .unwrap(),
      "FullForm[DateObject[{2026, 4, 29}, \"Day\"]]"
    );
  }

  #[test]
  fn date_plus_quantity_months() {
    assert_eq!(
      interpret("FullForm[DatePlus[DateObject[{2026, 4, 1}, \"Day\"], Quantity[5, \"Months\"]]]")
        .unwrap(),
      "FullForm[DateObject[{2026, 9, 1}, \"Day\"]]"
    );
  }

  #[test]
  fn date_plus_quantity_months_with_day_clamping() {
    assert_eq!(
      interpret("DatePlus[{2026, 1, 31}, Quantity[1, \"Months\"]]").unwrap(),
      "{2026, 2, 28}"
    );
  }

  #[test]
  fn date_plus_quantity_years() {
    assert_eq!(
      interpret("DatePlus[{2026, 4, 1}, Quantity[2, \"Years\"]]").unwrap(),
      "{2028, 4, 1}"
    );
  }

  #[test]
  fn day_name_date_plus_quantity() {
    assert_eq!(
      interpret("DayName[DatePlus[DateObject[{2026, 4, 1}, \"Day\"], Quantity[5, \"Months\"]]]")
        .unwrap(),
      "Tuesday"
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

  // An ISO date/time string is parsed into a component list with the implied
  // granularity (Y-M-D → Day, Y-M → Month, Y → Year, with a time → Instant).
  #[test]
  fn date_object_parses_iso_string() {
    assert_eq!(
      interpret("DateObject[\"2024-03-15\"]").unwrap(),
      "DateObject[{2024, 3, 15}, Day]"
    );
    assert_eq!(
      interpret("DateObject[\"2024-3-15\"]").unwrap(),
      "DateObject[{2024, 3, 15}, Day]"
    );
    assert_eq!(
      interpret("DateObject[\"2024-03\"]").unwrap(),
      "DateObject[{2024, 3}, Month]"
    );
    assert_eq!(
      interpret("DateObject[\"2024\"]").unwrap(),
      "DateObject[{2024}, Year]"
    );
  }

  #[test]
  fn date_object_parses_iso_datetime() {
    assert_eq!(
      interpret("DateObject[\"2024-03-15 14:30:00\"]").unwrap(),
      "DateObject[{2024, 3, 15, 14, 30, 0}, Instant, Gregorian, 0.]"
    );
    assert_eq!(
      interpret("DateObject[\"2024-03-15T14:30:00\"]").unwrap(),
      "DateObject[{2024, 3, 15, 14, 30, 0}, Instant, Gregorian, 0.]"
    );
  }

  // A string that is not an ISO date stays unparsed.
  #[test]
  fn date_object_non_iso_string_unparsed() {
    assert_eq!(
      interpret("DateObject[\"hello\"]").unwrap(),
      "DateObject[hello]"
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

  // DateString[fmt] with a recognized format name formats the CURRENT date,
  // i.e. DateString[fmt] == DateString[Now, fmt]. Tested structurally (the
  // actual date is non-deterministic) plus a consistency check.
  #[test]
  fn date_string_single_format_uses_current_date() {
    assert_eq!(
      interpret("StringLength[DateString[\"ISODate\"]]").unwrap(),
      "10"
    );
    assert_eq!(
      interpret("StringLength[DateString[\"ISODateTime\"]]").unwrap(),
      "19"
    );
    assert_eq!(
      interpret(
        "StringMatchQ[DateString[\"ISODate\"], RegularExpression[\"\\\\d{4}-\\\\d{2}-\\\\d{2}\"]]"
      )
      .unwrap(),
      "True"
    );
    // Consistent with the explicit current-date form.
    assert_eq!(
      interpret(
        "DateString[\"ISODate\"] == DateString[DateList[], \"ISODate\"]"
      )
      .unwrap(),
      "True"
    );
    assert_eq!(
      interpret("DateString[\"Year\"] == DateString[DateList[], \"Year\"]")
        .unwrap(),
      "True"
    );
    // A non-format string is still returned unchanged.
    assert_eq!(interpret("DateString[\"hello\"]").unwrap(), "hello");
  }

  #[test]
  fn date_string_date_object_day_granularity_omits_time() {
    // DateObject with "Day" granularity should omit time in default format
    assert_eq!(
      interpret("DateString[DateObject[{2024, 1, 15}, \"Day\"]]").unwrap(),
      "Mon 15 Jan 2024"
    );
  }

  #[test]
  fn date_string_date_object_implicit_day_granularity() {
    // DateObject[{y,m,d}] creates Day granularity
    assert_eq!(
      interpret("DateString[DateObject[{2024, 1, 15}]]").unwrap(),
      "Mon 15 Jan 2024"
    );
  }

  #[test]
  fn date_string_plain_list_includes_time() {
    // Plain list always includes time (even when zero)
    assert_eq!(
      interpret("DateString[{2024, 6, 15}]").unwrap(),
      "Sat 15 Jun 2024 00:00:00"
    );
  }

  #[test]
  fn date_string_date_object_with_time_includes_time() {
    // DateObject with time components should include time
    assert_eq!(
      interpret("DateString[DateObject[{2024, 1, 15, 10, 30, 0}]]").unwrap(),
      "Mon 15 Jan 2024 10:30:00"
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

mod leap_year_q {
  use woxi::interpret;

  #[test]
  fn leap_year_bare_integer() {
    // Wolfram's LeapYearQ does not accept bare integers, returns False
    assert_eq!(interpret("LeapYearQ[2024]").unwrap(), "False");
    assert_eq!(interpret("LeapYearQ[2023]").unwrap(), "False");
    assert_eq!(interpret("LeapYearQ[2000]").unwrap(), "False");
    assert_eq!(interpret("LeapYearQ[1900]").unwrap(), "False");
  }

  #[test]
  fn leap_year_list_format() {
    assert_eq!(interpret("LeapYearQ[{2024}]").unwrap(), "True");
    assert_eq!(interpret("LeapYearQ[{2023}]").unwrap(), "False");
  }

  #[test]
  fn leap_year_century_rules() {
    // Divisible by 400 -> leap
    assert_eq!(interpret("LeapYearQ[{2000}]").unwrap(), "True");
    // Divisible by 100 but not 400 -> not leap
    assert_eq!(interpret("LeapYearQ[{1900}]").unwrap(), "False");
    // Divisible by 4 but not 100 -> leap
    assert_eq!(interpret("LeapYearQ[{2004}]").unwrap(), "True");
  }

  #[test]
  fn leap_year_string_format() {
    // Wolfram parses a string date spec and tests its year.
    // Year-only strings:
    assert_eq!(interpret(r#"LeapYearQ["2024"]"#).unwrap(), "True");
    assert_eq!(interpret(r#"LeapYearQ["2023"]"#).unwrap(), "False");
    assert_eq!(interpret(r#"LeapYearQ["1996"]"#).unwrap(), "True");
    // Century rules also apply to string years:
    assert_eq!(interpret(r#"LeapYearQ["2000"]"#).unwrap(), "True");
    assert_eq!(interpret(r#"LeapYearQ["1900"]"#).unwrap(), "False");
    // ISO date string -> uses its year:
    assert_eq!(interpret(r#"LeapYearQ["2024-03-01"]"#).unwrap(), "True");
  }
}

mod cases {
  use super::super::case_helpers::assert_case;

  #[test]
  fn with() {
    // The mathics original (`>> AbsoluteTime[] = ...`) accepts any
    // output — wolframscript returns seconds since 1900-01-01, which
    // changes every second. Verify the documented contract: a Real
    // greater than the year-2000 epoch second count (3155673600).
    assert_case(
      r#"With[{t = AbsoluteTime[]}, Head[t] === Real && t > 3155673600]"#,
      r#"True"#,
    );
  }
  #[test]
  fn absolute_time_1() {
    assert_case(r#"AbsoluteTime[]; AbsoluteTime[{2000}]"#, r#"3155673600"#);
  }
  #[test]
  fn absolute_time_2() {
    assert_case(
      r#"AbsoluteTime[]; AbsoluteTime[{2000}]; AbsoluteTime[{"01/02/03", {"Day", "Month", "YearShort"}}]"#,
      r#"3.2530464*^9"#,
    );
  }
  #[test]
  fn absolute_time_3() {
    assert_case(
      r#"AbsoluteTime[]; AbsoluteTime[{2000}]; AbsoluteTime[{"01/02/03", {"Day", "Month", "YearShort"}}]; AbsoluteTime["6 June 1991"]"#,
      r#"2885155200"#,
    );
  }
  #[test]
  fn absolute_time_4() {
    assert_case(
      r#"AbsoluteTime[]; AbsoluteTime[{2000}]; AbsoluteTime[{"01/02/03", {"Day", "Month", "YearShort"}}]; AbsoluteTime["6 June 1991"]; AbsoluteTime[{"6-6-91", {"Day", "Month", "YearShort"}}]"#,
      r#"2.8851552*^9"#,
    );
  }
  #[test]
  fn date_difference_1() {
    assert_case(
      r#"DateDifference[{2042, 1, 4}, {2057, 1, 1}]"#,
      r#"Quantity[5476, "Days"]"#,
    );
  }
  #[test]
  fn date_difference_2() {
    assert_case(
      r#"DateDifference[{2042, 1, 4}, {2057, 1, 1}]; DateDifference[{1936, 8, 14}, {2000, 12, 1}, "Year"]"#,
      r#"Quantity[64.2986301369863, "Years"]"#,
    );
  }
  #[test]
  fn date_difference_3() {
    assert_case(
      r#"DateDifference[{2042, 1, 4}, {2057, 1, 1}]; DateDifference[{1936, 8, 14}, {2000, 12, 1}, "Year"]; DateDifference[{2010, 6, 1}, {2015, 1, 1}, "Hour"]"#,
      r#"Quantity[40200, "Hours"]"#,
    );
  }
  #[test]
  fn date_difference_4() {
    assert_case(
      r#"DateDifference[{2042, 1, 4}, {2057, 1, 1}]; DateDifference[{1936, 8, 14}, {2000, 12, 1}, "Year"]; DateDifference[{2010, 6, 1}, {2015, 1, 1}, "Hour"]; DateDifference[{2003, 8, 11}, {2003, 10, 19}, {"Week", "Day"}]"#,
      r#"Quantity[MixedMagnitude[{9, 6}], MixedUnit[{"Weeks", "Days"}]]"#,
    );
  }
  #[test]
  fn date_object() {
    assert_case(
      r#"DateObject[{2020, 4, 15}]"#,
      r#"DateObject[{2020, 4, 15}, "Day"]"#,
    );
  }
  #[test]
  fn date_plus_1() {
    assert_case(r#"DatePlus[{2010, 2, 5}, 73]"#, r#"{2010, 4, 19}"#);
  }
  #[test]
  fn date_plus_2() {
    assert_case(
      r#"DatePlus[{2010, 2, 5}, 73]; DatePlus[{2010, 2, 5}, {{8, "Week"}, {1, "Day"}}]"#,
      r#"{2010, 4, 3}"#,
    );
  }
  #[test]
  fn date_list_1() {
    assert_case(r#"DateList[0]"#, r#"{1900, 1, 1, 0, 0, 0.}"#);
  }
  #[test]
  fn date_list_2() {
    assert_case(
      r#"DateList[0]; DateList[3155673600]"#,
      r#"{2000, 1, 1, 0, 0, 0.}"#,
    );
  }
  #[test]
  fn date_list_3() {
    assert_case(
      r#"DateList[0]; DateList[3155673600]; DateList[{2003, 5, 0.5, 0.1, 0.767}]"#,
      r#"{2003, 4, 30, 12, 6, 46.019999980926514}"#,
    );
  }
  #[test]
  fn date_list_4() {
    assert_case(
      r#"DateList[0]; DateList[3155673600]; DateList[{2003, 5, 0.5, 0.1, 0.767}]; DateList[{2012, 1, 300., 10}]"#,
      r#"{2012, 10, 26, 10, 0, 0.}"#,
    );
  }
  #[test]
  fn date_list_5() {
    assert_case(
      r#"DateList[0]; DateList[3155673600]; DateList[{2003, 5, 0.5, 0.1, 0.767}]; DateList[{2012, 1, 300., 10}]; DateList["31/10/1991"]"#,
      r#"{1991, 10, 31, 0, 0, 0.}"#,
    );
  }
  #[test]
  fn date_string_1() {
    assert_case(
      r#"DateString[]; DateString[{1991, 10, 31, 0, 0}, {"Day", " ", "MonthName", " ", "Year"}]"#,
      r#""31 October 1991""#,
    );
  }
  #[test]
  fn date_string_2() {
    assert_case(
      r#"DateString[]; DateString[{1991, 10, 31, 0, 0}, {"Day", " ", "MonthName", " ", "Year"}]; DateString[{2007, 4, 15, 0}]"#,
      r#""Sun 15 Apr 2007 00:00:00""#,
    );
  }
  #[test]
  fn date_string_3() {
    assert_case(
      r#"DateString[]; DateString[{1991, 10, 31, 0, 0}, {"Day", " ", "MonthName", " ", "Year"}]; DateString[{2007, 4, 15, 0}]; DateString[{1979, 3, 14}, {"DayName", "  ", "Month", "-", "YearShort"}]"#,
      r#""Wednesday  03-79""#,
    );
  }
  #[test]
  fn date_string_4() {
    assert_case(
      r#"DateString[]; DateString[{1991, 10, 31, 0, 0}, {"Day", " ", "MonthName", " ", "Year"}]; DateString[{2007, 4, 15, 0}]; DateString[{1979, 3, 14}, {"DayName", "  ", "Month", "-", "YearShort"}]; DateString[{1991, 6, 6.5}]"#,
      r#""Thu 6 Jun 1991 12:00:00""#,
    );
  }
  #[test]
  fn easter_sunday_1() {
    assert_case(r#"EasterSunday[2000]"#, r#"EasterSunday[2000]"#);
  }
  #[test]
  fn easter_sunday_2() {
    assert_case(
      r#"EasterSunday[2000]; EasterSunday[2030]"#,
      r#"EasterSunday[2030]"#,
    );
  }
  #[test]
  fn absolute_time_5() {
    assert_case(r#"AbsoluteTime[1000]"#, r#"1000"#);
  }

  #[test]
  fn day_range_basic() {
    assert_case(
      r#"DayRange[{2020, 1, 1}, {2020, 1, 5}]"#,
      r#"{DateObject[{2020, 1, 1}, Day], DateObject[{2020, 1, 2}, Day], DateObject[{2020, 1, 3}, Day], DateObject[{2020, 1, 4}, Day], DateObject[{2020, 1, 5}, Day]}"#,
    );
  }

  #[test]
  fn day_range_single() {
    assert_case(
      r#"DayRange[{2020, 1, 1}, {2020, 1, 1}]"#,
      r#"{DateObject[{2020, 1, 1}, Day]}"#,
    );
  }

  #[test]
  fn day_range_weekday_filter() {
    // 3-argument form keeps only the given weekday; `{2013, 1}` defaults the
    // day to 1.
    assert_case(
      r#"DayRange[{2013, 1}, {2013, 1, 31}, Sunday]"#,
      r#"{DateObject[{2013, 1, 6}, Day], DateObject[{2013, 1, 13}, Day], DateObject[{2013, 1, 20}, Day], DateObject[{2013, 1, 27}, Day]}"#,
    );
  }

  #[test]
  fn day_range_month_boundary() {
    assert_case(
      r#"DayRange[{2020, 1, 30}, {2020, 2, 2}]"#,
      r#"{DateObject[{2020, 1, 30}, Day], DateObject[{2020, 1, 31}, Day], DateObject[{2020, 2, 1}, Day], DateObject[{2020, 2, 2}, Day]}"#,
    );
  }

  #[test]
  fn day_range_year_boundary() {
    assert_case(
      r#"DayRange[{2019, 12, 30}, {2020, 1, 2}]"#,
      r#"{DateObject[{2019, 12, 30}, Day], DateObject[{2019, 12, 31}, Day], DateObject[{2020, 1, 1}, Day], DateObject[{2020, 1, 2}, Day]}"#,
    );
  }

  // Reversed range is normalized to ascending order (matches wolframscript).
  #[test]
  fn day_range_reversed() {
    assert_case(
      r#"DayRange[{2020, 1, 3}, {2020, 1, 1}]"#,
      r#"{DateObject[{2020, 1, 1}, Day], DateObject[{2020, 1, 2}, Day], DateObject[{2020, 1, 3}, Day]}"#,
    );
  }

  #[test]
  fn day_range_date_object_inputs() {
    assert_case(
      r#"DayRange[DateObject[{2020, 1, 1}], DateObject[{2020, 1, 3}]]"#,
      r#"{DateObject[{2020, 1, 1}, Day], DateObject[{2020, 1, 2}, Day], DateObject[{2020, 1, 3}, Day]}"#,
    );
  }
}

mod date_range {
  use super::super::case_helpers::assert_case;

  #[test]
  fn date_range_basic_daily() {
    // Default increment is one day; each element is a six-field date list with
    // a Real seconds component.
    assert_case(
      r#"DateRange[{2024, 1, 1}, {2024, 1, 3}]"#,
      r#"{{2024, 1, 1, 0, 0, 0.}, {2024, 1, 2, 0, 0, 0.}, {2024, 1, 3, 0, 0, 0.}}"#,
    );
  }

  #[test]
  fn date_range_single_element() {
    assert_case(
      r#"DateRange[{2024, 1, 1}, {2024, 1, 1}]"#,
      r#"{{2024, 1, 1, 0, 0, 0.}}"#,
    );
  }

  #[test]
  fn date_range_integer_increment_is_days() {
    assert_case(
      r#"DateRange[{2024, 1, 1}, {2024, 1, 10}, 2]"#,
      r#"{{2024, 1, 1, 0, 0, 0.}, {2024, 1, 3, 0, 0, 0.}, {2024, 1, 5, 0, 0, 0.}, {2024, 1, 7, 0, 0, 0.}, {2024, 1, 9, 0, 0, 0.}}"#,
    );
  }

  #[test]
  fn date_range_quantity_days() {
    assert_case(
      r#"DateRange[{2024, 1, 1}, {2024, 1, 10}, Quantity[3, "Days"]]"#,
      r#"{{2024, 1, 1, 0, 0, 0.}, {2024, 1, 4, 0, 0, 0.}, {2024, 1, 7, 0, 0, 0.}, {2024, 1, 10, 0, 0, 0.}}"#,
    );
  }

  #[test]
  fn date_range_quantity_weeks() {
    assert_case(
      r#"DateRange[{2024, 1, 1}, {2024, 1, 15}, Quantity[1, "Weeks"]]"#,
      r#"{{2024, 1, 1, 0, 0, 0.}, {2024, 1, 8, 0, 0, 0.}, {2024, 1, 15, 0, 0, 0.}}"#,
    );
  }

  #[test]
  fn date_range_quantity_hours_preserves_time() {
    assert_case(
      r#"DateRange[{2024, 1, 1, 12, 0, 0}, {2024, 1, 1, 15, 0, 0}, Quantity[1, "Hours"]]"#,
      r#"{{2024, 1, 1, 12, 0, 0.}, {2024, 1, 1, 13, 0, 0.}, {2024, 1, 1, 14, 0, 0.}, {2024, 1, 1, 15, 0, 0.}}"#,
    );
  }

  #[test]
  fn date_range_calendar_months() {
    assert_case(
      r#"DateRange[{2024, 1, 15}, {2024, 4, 15}, Quantity[1, "Months"]]"#,
      r#"{{2024, 1, 15, 0, 0, 0.}, {2024, 2, 15, 0, 0, 0.}, {2024, 3, 15, 0, 0, 0.}, {2024, 4, 15, 0, 0, 0.}}"#,
    );
  }

  #[test]
  fn date_range_calendar_years() {
    assert_case(
      r#"DateRange[{2020, 6, 1}, {2023, 6, 1}, Quantity[1, "Years"]]"#,
      r#"{{2020, 6, 1, 0, 0, 0.}, {2021, 6, 1, 0, 0, 0.}, {2022, 6, 1, 0, 0, 0.}, {2023, 6, 1, 0, 0, 0.}}"#,
    );
  }

  #[test]
  fn date_range_string_unit() {
    assert_case(
      r#"DateRange[{2024, 1, 1}, {2024, 3, 1}, "Month"]"#,
      r#"{{2024, 1, 1, 0, 0, 0.}, {2024, 2, 1, 0, 0, 0.}, {2024, 3, 1, 0, 0, 0.}}"#,
    );
  }

  #[test]
  fn date_range_reversed_is_empty() {
    // Unlike DayRange, DateRange does not normalize a descending range.
    assert_case(r#"DateRange[{2024, 1, 5}, {2024, 1, 1}]"#, r#"{}"#);
  }
}

mod time_object {
  use super::super::case_helpers::assert_case;

  #[test]
  fn time_object_minute_granularity() {
    assert_case(r#"TimeObject[{14, 30}]"#, r#"TimeObject[{14, 30}, Minute]"#);
  }

  #[test]
  fn time_object_hour_granularity() {
    assert_case(r#"TimeObject[{14}]"#, r#"TimeObject[{14}, Hour]"#);
  }

  #[test]
  fn time_object_instant_granularity() {
    assert_case(
      r#"TimeObject[{14, 30, 15}]"#,
      r#"TimeObject[{14, 30, 15}, Instant]"#,
    );
  }

  #[test]
  fn time_object_keeps_fractional_seconds() {
    assert_case(
      r#"TimeObject[{14, 30, 15.5}]"#,
      r#"TimeObject[{14, 30, 15.5}, Instant]"#,
    );
  }

  #[test]
  fn time_object_hours_wrap_modulo_day() {
    assert_case(r#"TimeObject[{25, 30}]"#, r#"TimeObject[{1, 30}, Minute]"#);
  }

  #[test]
  fn time_object_minutes_carry() {
    assert_case(r#"TimeObject[{14, 75}]"#, r#"TimeObject[{15, 15}, Minute]"#);
  }

  #[test]
  fn time_object_seconds_carry() {
    assert_case(
      r#"TimeObject[{14, 30, 75}]"#,
      r#"TimeObject[{14, 31, 15}, Instant]"#,
    );
  }

  #[test]
  fn time_object_negative_wraps() {
    assert_case(r#"TimeObject[{-1, 30}]"#, r#"TimeObject[{23, 30}, Minute]"#);
  }

  #[test]
  fn time_object_fractional_hours_carry() {
    assert_case(
      r#"TimeObject[{14.5, 30}]"#,
      r#"TimeObject[{15, 0}, Minute]"#,
    );
  }

  #[test]
  fn time_object_full_day_wraps_to_zero() {
    assert_case(
      r#"TimeObject[{23, 59, 60}]"#,
      r#"TimeObject[{0, 0, 0}, Instant]"#,
    );
  }

  #[test]
  fn time_object_fractional_hour_floored_at_hour_granularity() {
    assert_case(r#"TimeObject[{14.5}]"#, r#"TimeObject[{14}, Hour]"#);
  }
}

mod julian_date {
  use super::*;

  #[test]
  fn epoch_and_common_dates() {
    // J2000.0
    assert_eq!(
      interpret("JulianDate[{2000, 1, 1, 12, 0, 0}]").unwrap(),
      "2.451545*^6"
    );
    // Midnight gives the half-day boundary
    assert_eq!(
      interpret("JulianDate[{2026, 6, 11}]").unwrap(),
      "2.4612025*^6"
    );
    // Short date lists default to January 1st, 00:00
    assert_eq!(interpret("JulianDate[{2000}]").unwrap(), "2.4515445*^6");
    assert_eq!(
      interpret("JulianDate[{2000, 1, 1, 12}]").unwrap(),
      "2.451545*^6"
    );
    // Second overflow rolls over (leap-second style input)
    assert_eq!(
      interpret("JulianDate[{1999, 12, 31, 23, 59, 60}]").unwrap(),
      "2.4515445*^6"
    );
    // Gregorian calendar adoption date
    assert_eq!(
      interpret("JulianDate[{1582, 10, 15}]").unwrap(),
      "2.2991605*^6"
    );
    // Fractional seconds stay at machine precision
    assert_eq!(
      interpret("JulianDate[{2000, 1, 1, 12, 30, 45.5}]").unwrap(),
      "2.4515450213599536*^6"
    );
  }

  #[test]
  fn no_input_year_zero() {
    // Wolfram treats both 0 and -1 as 1 BC (astronomical year 0)
    assert_eq!(interpret("JulianDate[{0, 1, 1}]").unwrap(), "1.7210595*^6");
    assert_eq!(interpret("JulianDate[{-1, 1, 1}]").unwrap(), "1.7210595*^6");
    assert_eq!(interpret("JulianDate[{1, 1, 1}]").unwrap(), "1.7214255*^6");
    assert_eq!(
      interpret("JulianDate[{-100, 1, 1}]").unwrap(),
      "1.6849005*^6"
    );
    // JD 0 sits in input year -4713 (proleptic Gregorian)
    assert_eq!(interpret("JulianDate[{-4713, 1, 1}]").unwrap(), "37.5");
    assert_eq!(
      interpret("JulianDate[{-4712, 1, 1, 12, 0, 0}]").unwrap(),
      "404."
    );
  }

  #[test]
  fn non_date_input_stays_unevaluated() {
    assert_eq!(interpret("JulianDate[x]").unwrap(), "JulianDate[x]");
  }
}

mod from_unix_time {
  use super::*;

  #[test]
  fn epoch_and_known_instants() {
    // The Unix epoch itself.
    assert_eq!(
      interpret("FromUnixTime[0, TimeZone -> 0]").unwrap(),
      "DateObject[{1970, 1, 1, 0, 0, 0}, Instant, Gregorian, 0.]"
    );
    // 2020-01-01T00:00:00Z.
    assert_eq!(
      interpret("FromUnixTime[1577836800, TimeZone -> 0]").unwrap(),
      "DateObject[{2020, 1, 1, 0, 0, 0}, Instant, Gregorian, 0.]"
    );
    // A billion seconds after the epoch.
    assert_eq!(
      interpret("FromUnixTime[1000000000, TimeZone -> 0]").unwrap(),
      "DateObject[{2001, 9, 9, 1, 46, 40}, Instant, Gregorian, 0.]"
    );
    // Sub-minute precision survives.
    assert_eq!(
      interpret("FromUnixTime[1577836845, TimeZone -> 0]").unwrap(),
      "DateObject[{2020, 1, 1, 0, 0, 45}, Instant, Gregorian, 0.]"
    );
  }

  #[test]
  fn time_zone_offsets_shift_the_displayed_time() {
    assert_eq!(
      interpret("FromUnixTime[0, TimeZone -> 1]").unwrap(),
      "DateObject[{1970, 1, 1, 1, 0, 0}, Instant, Gregorian, 1.]"
    );
    assert_eq!(
      interpret("FromUnixTime[0, TimeZone -> -5]").unwrap(),
      "DateObject[{1969, 12, 31, 19, 0, 0}, Instant, Gregorian, -5.]"
    );
    // Fractional offsets are supported.
    assert_eq!(
      interpret("FromUnixTime[0, TimeZone -> 5.5]").unwrap(),
      "DateObject[{1970, 1, 1, 5, 30, 0}, Instant, Gregorian, 5.5]"
    );
  }

  #[test]
  fn head_is_date_object() {
    assert_eq!(
      interpret("Head[FromUnixTime[0, TimeZone -> 0]]").unwrap(),
      "DateObject"
    );
  }
}

mod from_absolute_time {
  use super::*;

  #[test]
  fn epoch_and_known_instants() {
    // The Wolfram epoch itself (1900-01-01), with a zero time-zone offset.
    assert_eq!(
      interpret("FromAbsoluteTime[0]").unwrap(),
      "DateObject[{1900, 1, 1, 0, 0, 0}, Instant, Gregorian, 0.]"
    );
    // One day later.
    assert_eq!(
      interpret("FromAbsoluteTime[86400]").unwrap(),
      "DateObject[{1900, 1, 2, 0, 0, 0}, Instant, Gregorian, 0.]"
    );
    // The Unix epoch is 2208988800 seconds after the Wolfram epoch.
    assert_eq!(
      interpret("FromAbsoluteTime[2208988800]").unwrap(),
      "DateObject[{1970, 1, 1, 0, 0, 0}, Instant, Gregorian, 0.]"
    );
    // 2000-01-01.
    assert_eq!(
      interpret("FromAbsoluteTime[3155673600]").unwrap(),
      "DateObject[{2000, 1, 1, 0, 0, 0}, Instant, Gregorian, 0.]"
    );
  }

  // FromAbsoluteTime inverts AbsoluteTime.
  #[test]
  fn round_trips_with_absolute_time() {
    assert_eq!(
      interpret("FromAbsoluteTime[AbsoluteTime[{2024, 6, 14}]]").unwrap(),
      "DateObject[{2024, 6, 14, 0, 0, 0}, Instant, Gregorian, 0.]"
    );
  }

  #[test]
  fn head_is_date_object() {
    assert_eq!(
      interpret("Head[FromAbsoluteTime[0]]").unwrap(),
      "DateObject"
    );
  }
}

mod unix_time {
  use super::*;

  #[test]
  fn from_date_list() {
    // A full date list, interpreted as UTC.
    assert_eq!(
      interpret("UnixTime[{2020, 1, 1, 0, 0, 0}]").unwrap(),
      "1577836800"
    );
    // Short date lists default the missing components.
    assert_eq!(interpret("UnixTime[{2020, 1, 1}]").unwrap(), "1577836800");
  }

  #[test]
  fn round_trips_with_from_unix_time() {
    // UnixTime undoes FromUnixTime regardless of the display time zone.
    assert_eq!(
      interpret("UnixTime[FromUnixTime[1577836800, TimeZone -> 0]]").unwrap(),
      "1577836800"
    );
    assert_eq!(
      interpret("UnixTime[FromUnixTime[1577836800, TimeZone -> -5]]").unwrap(),
      "1577836800"
    );
    assert_eq!(
      interpret("UnixTime[FromUnixTime[1000000000, TimeZone -> 5.5]]").unwrap(),
      "1000000000"
    );
  }
}

mod date_value {
  use super::*;

  #[test]
  fn calendar_components() {
    assert_eq!(
      interpret(r#"DateValue[{2024, 6, 15}, "Year"]"#).unwrap(),
      "2024"
    );
    assert_eq!(
      interpret(r#"DateValue[{2024, 6, 15}, "Month"]"#).unwrap(),
      "6"
    );
    assert_eq!(
      interpret(r#"DateValue[{2024, 6, 15}, "Day"]"#).unwrap(),
      "15"
    );
    assert_eq!(
      interpret(r#"DateValue[{2024, 6, 15, 10, 30, 0}, "Hour"]"#).unwrap(),
      "10"
    );
  }

  #[test]
  fn named_components() {
    assert_eq!(
      interpret(r#"DateValue[{2024, 6, 15}, "DayName"]"#).unwrap(),
      "Saturday"
    );
    // DayName is a Symbol (no quotes in InputForm), matching wolframscript;
    // MonthName below is a String.
    assert_eq!(
      interpret(r#"ToString[DateValue[{2024, 6, 15}, "DayName"], InputForm]"#)
        .unwrap(),
      "Saturday"
    );
    assert_eq!(
      interpret(
        r#"ToString[DateValue[{2024, 6, 15}, "MonthName"], InputForm]"#
      )
      .unwrap(),
      "\"June\""
    );
    assert_eq!(
      interpret(r#"DateValue[{2024, 6, 15}, "MonthName"]"#).unwrap(),
      "June"
    );
    assert_eq!(
      interpret(r#"DateValue[{2024, 6, 15}, "Quarter"]"#).unwrap(),
      "2"
    );
  }

  // "DayNameShort" returns the short weekday name as a string.
  #[test]
  fn day_name_short() {
    assert_eq!(
      interpret(r#"DateValue[{2024, 3, 15}, "DayNameShort"]"#).unwrap(),
      "Fri"
    );
    assert_eq!(
      interpret(r#"DateValue[{2024, 12, 25}, "DayNameShort"]"#).unwrap(),
      "Wed"
    );
  }

  // "MonthNameShort" returns the abbreviated month name.
  #[test]
  fn month_name_short() {
    assert_eq!(
      interpret(r#"DateValue[{2024, 3, 15}, "MonthNameShort"]"#).unwrap(),
      "Mar"
    );
    assert_eq!(
      interpret(r#"DateValue[{2024, 1, 1}, "MonthNameShort"]"#).unwrap(),
      "Jan"
    );
    assert_eq!(
      interpret(r#"DateValue[{2024, 12, 25}, "MonthNameShort"]"#).unwrap(),
      "Dec"
    );
    // Works in a property list alongside other short names.
    assert_eq!(
      interpret(
        r#"DateValue[{2024, 3, 15}, {"MonthNameShort", "DayNameShort"}]"#
      )
      .unwrap(),
      "{Mar, Fri}"
    );
  }

  // "Hour12" is the 12-hour-clock hour: 0 and 12 both map to 12, 13..23 to
  // 1..11. "AMPM" is the meridiem string.
  #[test]
  fn hour12_and_ampm() {
    assert_eq!(
      interpret(r#"DateValue[{2024, 3, 15, 14, 30, 0}, "Hour12"]"#).unwrap(),
      "2"
    );
    assert_eq!(
      interpret(r#"DateValue[{2024, 3, 15, 0, 0, 0}, "Hour12"]"#).unwrap(),
      "12"
    );
    assert_eq!(
      interpret(r#"DateValue[{2024, 3, 15, 12, 0, 0}, "Hour12"]"#).unwrap(),
      "12"
    );
    assert_eq!(
      interpret(r#"DateValue[{2024, 3, 15, 14, 0, 0}, "AMPM"]"#).unwrap(),
      "PM"
    );
    assert_eq!(
      interpret(r#"DateValue[{2024, 3, 15, 0, 0, 0}, "AMPM"]"#).unwrap(),
      "AM"
    );
  }

  // "…Short" component forms return the plain integer field, except
  // "YearShort" which is the year modulo 100.
  #[test]
  fn short_component_forms() {
    assert_eq!(
      interpret(r#"DateValue[{2024, 3, 15}, "YearShort"]"#).unwrap(),
      "24"
    );
    assert_eq!(
      interpret(r#"DateValue[{2005, 3, 15}, "YearShort"]"#).unwrap(),
      "5"
    );
    assert_eq!(
      interpret(r#"DateValue[{2024, 3, 15}, "MonthShort"]"#).unwrap(),
      "3"
    );
    assert_eq!(
      interpret(r#"DateValue[{2024, 3, 15}, "DayShort"]"#).unwrap(),
      "15"
    );
    assert_eq!(
      interpret(r#"DateValue[{2024, 3, 15, 14, 30, 45}, "HourShort"]"#)
        .unwrap(),
      "14"
    );
    assert_eq!(
      interpret(r#"DateValue[{2024, 3, 15, 14, 30, 45}, "SecondShort"]"#)
        .unwrap(),
      "45"
    );
  }

  // ISO-8601 week date: "ISOWeek" and "ISOWeekYear", which differ from the
  // calendar values near year boundaries. Verified against wolframscript.
  #[test]
  fn iso_week_and_year() {
    assert_eq!(
      interpret(r#"DateValue[{2024, 3, 15}, "ISOWeek"]"#).unwrap(),
      "11"
    );
    assert_eq!(
      interpret(r#"DateValue[{2024, 3, 15}, "ISOWeekYear"]"#).unwrap(),
      "2024"
    );
  }

  #[test]
  fn iso_week_year_boundaries() {
    // 2023-01-01 is a Sunday → ISO week 52 of 2022.
    assert_eq!(
      interpret(r#"DateValue[{2023, 1, 1}, "ISOWeek"]"#).unwrap(),
      "52"
    );
    assert_eq!(
      interpret(r#"DateValue[{2023, 1, 1}, "ISOWeekYear"]"#).unwrap(),
      "2022"
    );
    // 2021-01-01 is a Friday → ISO week 53 of 2020.
    assert_eq!(
      interpret(r#"DateValue[{2021, 1, 1}, "ISOWeekYear"]"#).unwrap(),
      "2020"
    );
    // 2025-12-29 is a Monday → ISO week 1 of 2026.
    assert_eq!(
      interpret(r#"DateValue[{2025, 12, 29}, "ISOWeekYear"]"#).unwrap(),
      "2026"
    );
  }

  #[test]
  fn day_of_year_and_iso_weekday() {
    assert_eq!(
      interpret(r#"DateValue[{2024, 6, 15}, "DayOfYear"]"#).unwrap(),
      "167"
    );
    assert_eq!(
      interpret(r#"DateValue[{2024, 2, 29}, "DayOfYear"]"#).unwrap(),
      "60"
    );
    // Saturday is ISO weekday 6 (Monday = 1).
    assert_eq!(
      interpret(r#"DateValue[{2024, 6, 15}, "ISOWeekDay"]"#).unwrap(),
      "6"
    );
  }

  // ISO-8601 week number, including the year-boundary edge cases.
  #[test]
  fn iso_week_number() {
    assert_eq!(
      interpret(r#"DateValue[{2024, 6, 15}, "Week"]"#).unwrap(),
      "24"
    );
    assert_eq!(
      interpret(r#"DateValue[{2024, 1, 1}, "Week"]"#).unwrap(),
      "1"
    );
    // 2021-01-01 belongs to ISO week 53 of 2020.
    assert_eq!(
      interpret(r#"DateValue[{2021, 1, 1}, "Week"]"#).unwrap(),
      "53"
    );
    // 2024-12-30 belongs to ISO week 1 of 2025.
    assert_eq!(
      interpret(r#"DateValue[{2024, 12, 30}, "Week"]"#).unwrap(),
      "1"
    );
  }

  #[test]
  fn list_of_properties() {
    assert_eq!(
      interpret(r#"DateValue[{2024, 6, 15}, {"Year", "Month", "Day"}]"#)
        .unwrap(),
      "{2024, 6, 15}"
    );
  }

  // DateObject and date-string inputs are accepted too.
  #[test]
  fn accepts_date_object_and_string() {
    assert_eq!(
      interpret(r#"DateValue[DateObject[{2024, 6, 15}], "DayName"]"#).unwrap(),
      "Saturday"
    );
    assert_eq!(
      interpret(r#"DateValue["2024-06-15", "MonthName"]"#).unwrap(),
      "June"
    );
  }

  // Unrecognized element specs stay unevaluated.
  #[test]
  fn unknown_property_unevaluated() {
    assert_eq!(
      interpret(r#"DateValue[{2024, 6, 15}, "DayOfWeek"]"#).unwrap(),
      "DateValue[{2024, 6, 15}, DayOfWeek]"
    );
  }
}

mod day_match_q {
  use super::*;

  // 2024-06-15 is a Saturday; 2024-06-17 a Monday.
  #[test]
  fn weekday_name_symbol() {
    assert_eq!(
      interpret("DayMatchQ[{2024, 6, 15}, Saturday]").unwrap(),
      "True"
    );
    assert_eq!(
      interpret("DayMatchQ[{2024, 6, 15}, Sunday]").unwrap(),
      "False"
    );
    assert_eq!(
      interpret("DayMatchQ[{2024, 6, 17}, Monday]").unwrap(),
      "True"
    );
  }

  #[test]
  fn weekday_name_string() {
    assert_eq!(
      interpret(r#"DayMatchQ[{2024, 6, 15}, "Saturday"]"#).unwrap(),
      "True"
    );
  }

  #[test]
  fn weekend_and_weekday_categories() {
    assert_eq!(
      interpret(r#"DayMatchQ[{2024, 6, 15}, "Weekend"]"#).unwrap(),
      "True"
    );
    assert_eq!(
      interpret(r#"DayMatchQ[{2024, 6, 17}, "Weekend"]"#).unwrap(),
      "False"
    );
    assert_eq!(
      interpret(r#"DayMatchQ[{2024, 6, 17}, "Weekday"]"#).unwrap(),
      "True"
    );
    assert_eq!(
      interpret(r#"DayMatchQ[{2024, 6, 15}, "Weekday"]"#).unwrap(),
      "False"
    );
  }

  #[test]
  fn accepts_date_object() {
    assert_eq!(
      interpret("DayMatchQ[DateObject[{2024, 6, 15}], Saturday]").unwrap(),
      "True"
    );
  }

  // A bare category symbol (not a string) stays unevaluated, like Wolfram.
  #[test]
  fn bare_category_symbol_unevaluated() {
    assert_eq!(
      interpret("DayMatchQ[{2024, 6, 15}, Weekend]").unwrap(),
      "DayMatchQ[{2024, 6, 15}, Weekend]"
    );
  }
}

mod day_round {
  use super::*;

  // 2024-06-15 is a Saturday. DayRound advances to the next occurrence of the
  // weekday (the default next-day convention), staying put when it matches.
  #[test]
  fn rounds_forward_to_next_weekday() {
    assert_eq!(
      interpret("DayRound[{2024, 6, 15}, Saturday]").unwrap(),
      "DateObject[{2024, 6, 15}, Day]"
    );
    assert_eq!(
      interpret("DayRound[{2024, 6, 15}, Monday]").unwrap(),
      "DateObject[{2024, 6, 17}, Day]"
    );
    assert_eq!(
      interpret("DayRound[{2024, 6, 15}, Sunday]").unwrap(),
      "DateObject[{2024, 6, 16}, Day]"
    );
    // Friday is the previous day, but rounding still goes forward a week.
    assert_eq!(
      interpret("DayRound[{2024, 6, 15}, Friday]").unwrap(),
      "DateObject[{2024, 6, 21}, Day]"
    );
  }

  #[test]
  fn crosses_month_and_year_boundaries() {
    assert_eq!(
      interpret("DayRound[{2024, 6, 30}, Monday]").unwrap(),
      "DateObject[{2024, 7, 1}, Day]"
    );
    assert_eq!(
      interpret("DayRound[{2024, 12, 31}, Sunday]").unwrap(),
      "DateObject[{2025, 1, 5}, Day]"
    );
  }

  #[test]
  fn accepts_string_name_and_date_object() {
    assert_eq!(
      interpret(r#"DayRound[{2024, 6, 15}, "Monday"]"#).unwrap(),
      "DateObject[{2024, 6, 17}, Day]"
    );
    assert_eq!(
      interpret("DayRound[DateObject[{2024, 6, 15}], Monday]").unwrap(),
      "DateObject[{2024, 6, 17}, Day]"
    );
  }

  // DayRound[date] (no daytype) floors the date to its containing day.
  #[test]
  fn single_arg_floors_to_day() {
    assert_eq!(
      interpret("DayRound[{2024, 3, 15}]").unwrap(),
      "DateObject[{2024, 3, 15}, Day]"
    );
  }

  #[test]
  fn single_arg_drops_time_components() {
    // Any time of day truncates to the containing day (no half-up rounding).
    assert_eq!(
      interpret("DayRound[{2024, 3, 15, 8, 30, 0}]").unwrap(),
      "DateObject[{2024, 3, 15}, Day]"
    );
    assert_eq!(
      interpret("DayRound[{2024, 3, 15, 23, 59, 0}]").unwrap(),
      "DateObject[{2024, 3, 15}, Day]"
    );
  }

  #[test]
  fn single_arg_completes_partial_dates() {
    assert_eq!(
      interpret("DayRound[{2024}]").unwrap(),
      "DateObject[{2024, 1, 1}, Day]"
    );
    assert_eq!(
      interpret("DayRound[{2024, 2}]").unwrap(),
      "DateObject[{2024, 2, 1}, Day]"
    );
  }

  #[test]
  fn single_arg_accepts_date_object() {
    assert_eq!(
      interpret("DayRound[DateObject[{2024, 3, 15}]]").unwrap(),
      "DateObject[{2024, 3, 15}, Day]"
    );
  }
}
