---
icon: lucide/moon-star
---

# Astronomy

Woxi computes astronomical quantities from the algorithms in Meeus's
*Astronomical Algorithms*: lunar phases, Sun/Moon positions, sidereal
time, sunrise/sunset, and eclipse predictions. Locations must be given
explicitly as `GeoPosition[{lat, lon}]` (or a bare `{lat, lon}` pair):
determining `$GeoLocation` needs a GeoIP lookup, so — like wolframscript
without internet access — Woxi leaves the location unresolved and a call
that omits one stays unevaluated. All returned dates are UTC instants
(TimeZone `0.`), whereas wolframscript localizes them to the location's
time zone.

## MoonPhase

The illuminated fraction of the Moon, or the phase name as an entity:

```scrut
$ wo 'MoonPhase[DateObject[{2024, 1, 25}]]'
0.9928330914826304
```

```scrut
$ wo 'MoonPhase[DateObject[{2024, 1, 20, 12, 0, 0}], "Name"]'
Entity[MoonPhase, WaxingGibbous]
```

## NewMoon, FullMoon & MoonPhaseDate

The first new/full moon (or any principal phase) after a date:

```scrut
$ wo 'FullMoon[DateObject[{2024, 1, 1}]]'
DateObject[{2024, 1, 25, 17, 53, 56.243}, Instant, Gregorian, 0.]
```

```scrut
$ wo 'MoonPhaseDate[DateObject[{2024, 4, 1}], "FirstQuarter"]'
DateObject[{2024, 4, 15, 19, 13, 2.637}, Instant, Gregorian, 0.]
```

## SunPosition & MoonPosition

Azimuth (from north) and altitude, or right ascension and declination
with `CelestialSystem -> "Equatorial"`:

```scrut
$ wo 'SunPosition[GeoPosition[{40.11, -88.24}], DateObject[{2024, 6, 21, 18, 0, 0}]]'
{Quantity[184.04, AngularDegrees], Quantity[73.29, AngularDegrees]}
```

```scrut
$ wo 'MoonPosition[GeoPosition[{0, 0}], DateObject[{2024, 1, 1, 0, 0, 0}], CelestialSystem -> "Equatorial"]'
{Quantity[159.12, AngularDegrees], Quantity[12.63, AngularDegrees]}
```

## SiderealTime

Local apparent sidereal time:

```scrut
$ wo 'SiderealTime[GeoPosition[{0, 0}], DateObject[{1987, 4, 10}]]'
Quantity[MixedMagnitude[{13, 10, 46.1306}], MixedUnit[{"HoursOfRightAscension", "MinutesOfRightAscension", "SecondsOfRightAscension"}]]
```

## Sunrise, Sunset & DaylightQ

```scrut
$ wo 'Sunrise[GeoPosition[{52.52, 13.405}], DateObject[{2024, 6, 21}]]'
DateObject[{2024, 6, 21, 2, 43}, Minute, Gregorian, 0.]
```

```scrut
$ wo 'Sunset[GeoPosition[{52.52, 13.405}], DateObject[{2024, 6, 21}]]'
DateObject[{2024, 6, 21, 19, 33}, Minute, Gregorian, 0.]
```

```scrut
$ wo 'DaylightQ[GeoPosition[{52.52, 13.405}], DateObject[{2024, 6, 21, 12, 0, 0}]]'
True
```

## SolarEclipse & LunarEclipse

The next eclipse after a date — its time of greatest eclipse or its type:

```scrut
$ wo 'SolarEclipse[DateObject[{2024, 4, 1}]]'
DateObject[{2024, 4, 8, 18, 17, 41.163}, Instant, Gregorian, 0.]
```

```scrut
$ wo 'SolarEclipse[DateObject[{2023, 10, 1}], "Type"]'
Entity[EclipseType, Annular]
```

```scrut
$ wo 'LunarEclipse[DateObject[{2025, 1, 1}], "Type"]'
Entity[EclipseType, Total]
```
