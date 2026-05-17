events = {{"2009-12-25", "Christmas Day"}, {"2009-04-22",
    "Earth Day"}, {"2009-09-07", "Labor Day"}, {"2009-07-04",
    "Independence Day"}, {"2009-10-31", "Halloween"}, {"2009-05-25",
    "Memorial Day"}, {"2009-03-14", "PI Day"}, {"2009-01-01",
    "New Year's Day"}, {"2009-12-31",
    "New Year's Eve"}, {"2009-11-26", "Thanksgiving"}, {"2009-02-14",
    "St. Valentine's Day"}, {"2009-03-17",
    "St. Patrick's Day"}, {"2009-01-19",
    "Martin Luther King Day"}, {"2009-02-16", "President's Day"}};
date = 1;
name = 2;
SortBy[events, #[[name]] &] // Grid
SortBy[events, #[[date]] &] // Grid


(* ========== *)

2009-12-25 Christmas Day
2009-04-22 Earth Day
2009-10-31 Halloween
2009-07-04 Independence Day
2009-09-07 Labor Day
2009-01-19 Martin Luther King Day
2009-05-25 Memorial Day
2009-01-01 New Year's Day
2009-12-31 New Year's Eve
2009-03-14 PI Day
2009-02-16 President's Day
2009-03-17 St. Patrick's Day
2009-02-14 St. Valentine's Day
2009-11-26 Thanksgiving


2009-01-01 New Year's Day
2009-01-19 Martin Luther King Day
2009-02-14 St. Valentine's Day
2009-02-16 President's Day
2009-03-14 PI Day
2009-03-17 St. Patrick's Day
2009-04-22 Earth Day
2009-05-25 Memorial Day
2009-07-04 Independence Day
2009-09-07 Labor Day
2009-10-31 Halloween
2009-11-26 Thanksgiving
2009-12-25 Christmas Day
2009-12-31 New Year's Eve
