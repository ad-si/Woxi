years = {1900, 2100}; months = {1, 3, 5, 7, 8, 10, 12};
result = Select[Tuples[{Range @@ years, months}], (DateString[# ~ Join ~ {1}, "DayNameShort"] == "Fri") &];
Print[Length[result], " months with 5 weekends"]
