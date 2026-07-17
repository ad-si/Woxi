use super::*;

// Astronomical computations are checked against dates with well-known
// ephemeris values (eclipses, phases, solstice positions). All results
// follow Woxi's datetime convention: UTC instants, TimeZone 0.

mod moon_phase {
  use super::*;

  #[test]
  fn moon_phase_fraction() {
    // Waxing gibbous halfway between first quarter (Jan 18) and full
    // moon (Jan 25)
    assert_eq!(
      interpret("MoonPhase[DateObject[{2024, 1, 20, 12, 0, 0}]]").unwrap(),
      "0.7431388863547352"
    );
  }

  #[test]
  fn moon_phase_fraction_near_full() {
    assert_eq!(
      interpret("MoonPhase[{2024, 1, 25}]").unwrap(),
      "0.9928330914826304"
    );
  }

  #[test]
  fn moon_phase_names() {
    // 2024 January: new moon 11th, first quarter 18th, full 25th;
    // last quarter Feb 2
    for (date, name) in [
      ("{2024, 1, 11, 12, 0, 0}", "New"),
      ("{2024, 1, 14, 12, 0, 0}", "WaxingCrescent"),
      ("{2024, 1, 18, 4, 0, 0}", "FirstQuarter"),
      ("{2024, 1, 20, 12, 0, 0}", "WaxingGibbous"),
      ("{2024, 1, 25, 18, 0, 0}", "Full"),
      ("{2024, 2, 2, 12, 0, 0}", "LastQuarter"),
    ] {
      assert_eq!(
        interpret(&format!("MoonPhase[DateObject[{date}], \"Name\"]")).unwrap(),
        format!("Entity[MoonPhase, {name}]"),
        "phase name at {date}"
      );
    }
  }

  #[test]
  fn moon_phase_wrong_args_unevaluated() {
    assert_eq!(
      interpret("MoonPhase[1, 2, 3]").unwrap(),
      "MoonPhase[1, 2, 3]"
    );
  }
}

mod phase_dates {
  use super::*;

  #[test]
  fn new_moon_after_date() {
    // The new moon of the 2024 April 8 total solar eclipse (18:21 UTC)
    assert_eq!(
      interpret("NewMoon[DateObject[{2024, 4, 1}]]").unwrap(),
      "DateObject[{2024, 4, 8, 18, 20, 48.057}, Instant, Gregorian, 0.]"
    );
  }

  #[test]
  fn full_moon_after_date() {
    // Full moon of 2024 January 25, 17:54 UTC
    assert_eq!(
      interpret("FullMoon[DateObject[{2024, 1, 1}]]").unwrap(),
      "DateObject[{2024, 1, 25, 17, 53, 56.243}, Instant, Gregorian, 0.]"
    );
  }

  #[test]
  fn moon_phase_date_with_phase_string() {
    // Full moon of 2024 April 23, 23:49 UTC
    assert_eq!(
      interpret("MoonPhaseDate[DateObject[{2024, 4, 1}], \"FullMoon\"]")
        .unwrap(),
      "DateObject[{2024, 4, 23, 23, 48, 58.794}, Instant, Gregorian, 0.]"
    );
  }

  #[test]
  fn moon_phase_date_with_entity() {
    assert_eq!(
      interpret(
        "MoonPhaseDate[DateObject[{2024, 4, 1}], Entity[\"MoonPhase\", \
         \"FirstQuarter\"]]"
      )
      .unwrap(),
      "DateObject[{2024, 4, 15, 19, 13, 2.637}, Instant, Gregorian, 0.]"
    );
  }

  #[test]
  fn moon_phase_date_next_principal_phase() {
    // After the 2024 Apr 8 new moon, the next principal phase is the
    // first quarter of Apr 15
    assert_eq!(
      interpret("MoonPhaseDate[DateObject[{2024, 4, 9}]]").unwrap(),
      "DateObject[{2024, 4, 15, 19, 13, 2.637}, Instant, Gregorian, 0.]"
    );
  }
}

mod positions {
  use super::*;

  #[test]
  fn sun_position_summer_solstice() {
    // Local solar noon at the default location on the June solstice:
    // the Sun stands almost due south near its maximum altitude
    assert_eq!(
      interpret(
        "SunPosition[GeoPosition[{40.11, -88.24}], \
         DateObject[{2024, 6, 21, 18, 0, 0}]]"
      )
      .unwrap(),
      "{Quantity[184.04, AngularDegrees], Quantity[73.29, AngularDegrees]}"
    );
  }

  #[test]
  fn sun_position_equatorial_at_equinox() {
    // Hours after the 2024 March equinox (03:06 UTC), RA and
    // declination are barely past zero
    assert_eq!(
      interpret(
        "SunPosition[GeoPosition[{40.11, -88.24}], \
         DateObject[{2024, 3, 20, 12, 0, 0}], \
         CelestialSystem -> \"Equatorial\"]"
      )
      .unwrap(),
      "{Quantity[0.34, AngularDegrees], Quantity[0.15, AngularDegrees]}"
    );
  }

  #[test]
  fn sun_position_bare_lat_lon_list() {
    assert_eq!(
      interpret(
        "SunPosition[{52.52, 13.405}, DateObject[{2024, 12, 21, 12, 0, 0}]]"
      )
      .unwrap(),
      "{Quantity[193.01, AngularDegrees], Quantity[13.09, AngularDegrees]}"
    );
  }

  #[test]
  fn moon_position_horizon() {
    assert_eq!(
      interpret(
        "MoonPosition[GeoPosition[{40.11, -88.24}], \
         DateObject[{2024, 6, 21, 6, 0, 0}]]"
      )
      .unwrap(),
      "{Quantity[191.86, AngularDegrees], Quantity[21.16, AngularDegrees]}"
    );
  }

  #[test]
  fn moon_position_equatorial() {
    assert_eq!(
      interpret(
        "MoonPosition[GeoPosition[{0, 0}], DateObject[{2024, 1, 1, 0, 0, \
         0}], CelestialSystem -> \"Equatorial\"]"
      )
      .unwrap(),
      "{Quantity[159.12, AngularDegrees], Quantity[12.63, AngularDegrees]}"
    );
  }
}

mod sidereal_time {
  use super::*;

  #[test]
  fn sidereal_time_meeus_example() {
    // Meeus example 12.b: apparent sidereal time at Greenwich on
    // 1987 April 10.0 UT is 13h 10m 46.1351s
    assert_eq!(
      interpret(
        "SiderealTime[GeoPosition[{0, 0}], DateObject[{1987, 4, \
                 10}]]"
      )
      .unwrap(),
      "Quantity[MixedMagnitude[{13, 10, 46.1306}], \
       MixedUnit[{\"HoursOfRightAscension\", \"MinutesOfRightAscension\", \
       \"SecondsOfRightAscension\"}]]"
    );
  }

  #[test]
  fn sidereal_time_with_longitude() {
    assert_eq!(
      interpret(
        "SiderealTime[GeoPosition[{40.11, -88.24}], \
         DateObject[{2024, 6, 21, 18, 0, 0}]]"
      )
      .unwrap(),
      "Quantity[MixedMagnitude[{6, 8, 43.7647}], \
       MixedUnit[{\"HoursOfRightAscension\", \"MinutesOfRightAscension\", \
       \"SecondsOfRightAscension\"}]]"
    );
  }
}

mod rise_set {
  use super::*;

  #[test]
  fn sunrise_berlin_june_solstice() {
    // Berlin 2024-06-21: sunrise 04:43 CEST = 02:43 UTC
    assert_eq!(
      interpret(
        "Sunrise[GeoPosition[{52.52, 13.405}], DateObject[{2024, 6, 21}]]"
      )
      .unwrap(),
      "DateObject[{2024, 6, 21, 2, 43}, Minute, Gregorian, 0.]"
    );
  }

  #[test]
  fn sunset_berlin_june_solstice() {
    // Berlin 2024-06-21: sunset 21:33 CEST = 19:33 UTC
    assert_eq!(
      interpret(
        "Sunset[GeoPosition[{52.52, 13.405}], DateObject[{2024, 6, 21}]]"
      )
      .unwrap(),
      "DateObject[{2024, 6, 21, 19, 33}, Minute, Gregorian, 0.]"
    );
  }

  #[test]
  fn sunrise_with_date_list() {
    // Berlin 2024-12-21: sunrise 08:15 CET = 07:15 UTC
    assert_eq!(
      interpret("Sunrise[GeoPosition[{52.52, 13.405}], {2024, 12, 21}]")
        .unwrap(),
      "DateObject[{2024, 12, 21, 7, 15}, Minute, Gregorian, 0.]"
    );
  }

  #[test]
  fn sunset_equator_equinox() {
    // On the equator at the equinox the Sun sets ~6h04m after local
    // solar noon (12:07 UTC at longitude 0)
    assert_eq!(
      interpret("Sunset[{0, 0}, {2024, 3, 20}]").unwrap(),
      "DateObject[{2024, 3, 20, 18, 11}, Minute, Gregorian, 0.]"
    );
  }

  #[test]
  fn sunrise_polar_night_missing() {
    // Svalbard around the December solstice: the Sun never rises
    assert_eq!(
      interpret(
        "Sunrise[GeoPosition[{78.22, 15.63}], DateObject[{2024, 12, 21}]]"
      )
      .unwrap(),
      "Missing[NotApplicable]"
    );
  }

  #[test]
  fn daylight_q() {
    assert_eq!(
      interpret(
        "DaylightQ[GeoPosition[{52.52, 13.405}], \
         DateObject[{2024, 6, 21, 12, 0, 0}]]"
      )
      .unwrap(),
      "True"
    );
    assert_eq!(
      interpret(
        "DaylightQ[GeoPosition[{52.52, 13.405}], \
         DateObject[{2024, 6, 21, 23, 0, 0}]]"
      )
      .unwrap(),
      "False"
    );
  }
}

mod eclipses {
  use super::*;

  #[test]
  fn solar_eclipse_2024_total() {
    // Total solar eclipse of 2024 April 8, greatest eclipse 18:17:21 UTC
    assert_eq!(
      interpret("SolarEclipse[DateObject[{2024, 4, 1}]]").unwrap(),
      "DateObject[{2024, 4, 8, 18, 17, 41.163}, Instant, Gregorian, 0.]"
    );
    assert_eq!(
      interpret("SolarEclipse[DateObject[{2024, 4, 1}], \"Type\"]").unwrap(),
      "Total"
    );
  }

  #[test]
  fn solar_eclipse_types() {
    // 2023 Oct 14: annular; 2025 Sep 21: partial; 2013 Nov 3: hybrid
    assert_eq!(
      interpret("SolarEclipse[DateObject[{2023, 10, 1}], \"Type\"]").unwrap(),
      "Annular"
    );
    assert_eq!(
      interpret("SolarEclipse[DateObject[{2025, 9, 1}], \"Type\"]").unwrap(),
      "Partial"
    );
    assert_eq!(
      interpret("SolarEclipse[DateObject[{2013, 10, 20}], \"Type\"]").unwrap(),
      "Hybrid"
    );
  }

  #[test]
  fn lunar_eclipse_2025_total() {
    // Total lunar eclipse of 2025 March 14, greatest eclipse 06:58 UTC
    assert_eq!(
      interpret("LunarEclipse[DateObject[{2025, 1, 1}]]").unwrap(),
      "DateObject[{2025, 3, 14, 6, 59, 22.768}, Instant, Gregorian, 0.]"
    );
    assert_eq!(
      interpret("LunarEclipse[DateObject[{2025, 1, 1}], \"Type\"]").unwrap(),
      "Total"
    );
  }

  #[test]
  fn lunar_eclipse_types() {
    // 2024 Mar 25: penumbral; 2024 Sep 18: partial
    assert_eq!(
      interpret("LunarEclipse[DateObject[{2024, 3, 1}], \"Type\"]").unwrap(),
      "Penumbral"
    );
    assert_eq!(
      interpret("LunarEclipse[DateObject[{2024, 9, 1}], \"Type\"]").unwrap(),
      "Partial"
    );
  }
}

mod geo_location {
  use super::*;

  #[test]
  fn default_geo_location() {
    // Wolfram's offline fallback location
    assert_eq!(
      interpret("$GeoLocation").unwrap(),
      "GeoPosition[{40.11, -88.24}]"
    );
  }
}
