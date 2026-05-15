daysarray = {"first", "second", "third", "fourth", "fifth", "sixth",
   "seventh", "eighth", "ninth", "tenth", "eleventh", "twelfth"};
giftsarray = {"And a partridge in a pear tree.", "Two turtle doves",
   "Three french hens", "Four calling birds", "FIVE GOLDEN RINGS",
   "Six geese a-laying", "Seven swans a-swimming,",
   "Eight maids a-milking", "Nine ladies dancing",
   "Ten lords a-leaping", "Eleven pipers piping",
   "Twelve drummers drumming"};
Do[Print[StringForm[
   "On the `1` day of Christmas, my true love gave to me: `2`",
   daysarray[[i]],
   If[i == 1, "A partridge in a pear tree.",
    Row[Reverse[Take[giftsarray, i]], ","]]]], {i, 1, 12}]
