Needs["Calendar`"];


(* ========== *)

monthlyCalendar[y_, m_] :=
    Module[{
           days = {Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday},
           months = {"January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"},
           d1, shortDays, offset, daysInMonth},

     d1 = DayOfWeek[{y, m, 1}];

     daysInMonth[year_, month_] := DaysBetween[{year, month, 1}, {If[month == 12, year + 1, year],  If[month == 12, 1, month + 1], 1}];

     shortDays = (StringTake[ToString[#], 3] & /@ days);

     offset = d1 /. Thread[days -> Range[0, 6]];

     Grid[
             Prepend[
                     Prepend[
                             Partition[
                                   PadRight[PadLeft[Range[daysInMonth[y, m]], daysInMonth[y, m] + offset, ""],
                                                  36, ""  ],
                                           7],
                                   shortDays],
                                    {months[[m]], SpanFromLeft}]]]


(* ========== *)

yearlyCalendar[y_] := Grid[Partition[Table[monthlyCalendar[y, k], {k, 12}], 3], Spacings -> {4, 2}];
