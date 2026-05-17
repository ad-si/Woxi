Off[Power::indet];(*Supress 0^0 warnings*)
Select[Range[5000], Total[IntegerDigits[#]^IntegerDigits[#]] == # &]
