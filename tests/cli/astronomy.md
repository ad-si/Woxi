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

Positions are topocentric — the observer sits on the Earth's surface, so
the Moon's roughly one-degree parallax and the atmosphere's refraction of
the altitude are both included. Truncated series cannot reach the accuracy
of a full ephemeris, so the expected outputs below are written as regular
expressions that pin the digits Woxi and wolframscript agree on: a few
arcseconds for a position, a few tens of seconds for a phase or eclipse
time, and about half a minute for a sunrise (whose horizon threshold the
two engines define slightly differently).

## MoonPhase

The illuminated fraction of the Moon, or the phase name as an entity:

```scrut
$ wo 'MoonPhase[DateObject[{2024, 1, 25}]]'
0\.99283\d+ (regex)
```

```scrut
$ wo 'MoonPhase[DateObject[{2024, 1, 20, 12, 0, 0}], "Name"]'
Entity[MoonPhase, WaxingGibbous]
```

## NewMoon, FullMoon & MoonPhaseDate

The first new/full moon (or any principal phase) after a date:

```scrut
$ wo 'FullMoon[DateObject[{2024, 1, 1}]]'
DateObject\[\{2024, 1, 25, 17, 5[34], \d+\.\d+\}, Instant, Gregorian, 0\.\] (regex)
```

```scrut
$ wo 'MoonPhaseDate[DateObject[{2024, 4, 1}], "FirstQuarter"]'
DateObject\[\{2024, 4, 15, 19, 13, \d+\.\d+\}, Instant, Gregorian, 0\.\] (regex)
```

## SunPosition & MoonPosition

Azimuth (from north) and altitude, or right ascension — in hours, as
right ascension is customarily written — and declination with
`CelestialSystem -> "Equatorial"`:

```scrut
$ wo 'SunPosition[GeoPosition[{40.11, -88.24}], DateObject[{2024, 6, 21, 18, 0, 0}]]'
\{Quantity\[184\.0\d+, AngularDegrees\], Quantity\[73\.29\d+, AngularDegrees\]\} (regex)
```

```scrut
$ wo 'MoonPosition[GeoPosition[{0, 0}], DateObject[{2024, 1, 1, 0, 0, 0}], CelestialSystem -> "Equatorial"]'
\{Quantity\[10\.6612\d+, HoursOfRightAscension\], Quantity\[12\.728\d+, AngularDegrees\]\} (regex)
```

## SiderealTime

Local apparent sidereal time:

```scrut
$ wo 'SiderealTime[GeoPosition[{0, 0}], DateObject[{1987, 4, 10}]]'
Quantity\[MixedMagnitude\[\{13, 10, 4[56]\.\d+\}\], MixedUnit\[\{HoursOfRightAscension, MinutesOfRightAscension, SecondsOfRightAscension\}\]\] (regex)
```

## Sunrise, Sunset & DaylightQ

```scrut
$ wo 'Sunrise[GeoPosition[{52.52, 13.405}], DateObject[{2024, 6, 21}]]'
DateObject\[\{2024, 6, 21, 2, 4[23], \d+\.\d+\}, Instant, Gregorian, 0\.\] (regex)
```

```scrut
$ wo 'Sunset[GeoPosition[{52.52, 13.405}], DateObject[{2024, 6, 21}]]'
DateObject\[\{2024, 6, 21, 19, 33, \d+\.\d+\}, Instant, Gregorian, 0\.\] (regex)
```

```scrut
$ wo 'DaylightQ[GeoPosition[{52.52, 13.405}], DateObject[{2024, 6, 21, 12, 0, 0}]]'
True
```

## SolarEclipse & LunarEclipse

The next eclipse after a date — its time of greatest eclipse or its type:

```scrut
$ wo 'SolarEclipse[DateObject[{2024, 4, 1}]]'
DateObject\[\{2024, 4, 8, 18, 17, \d+\.\d+\}, Instant, Gregorian, 0\.\] (regex)
```

```scrut
$ wo 'SolarEclipse[DateObject[{2023, 10, 1}], "Type"]'
Entity[EclipseType, Annular]
```

```scrut
$ wo 'LunarEclipse[DateObject[{2025, 1, 1}], "Type"]'
Entity[EclipseType, Total]
```
