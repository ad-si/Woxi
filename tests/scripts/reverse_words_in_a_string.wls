poem = "---------- Ice and Fire ------------

     fire, in end will world the say Some
     ice. in say Some
     desire of tasted I've what From
     fire. favor who those with hold I

     ... elided paragraph last ...

     Frost Robert -----------------------";
lines = StringSplit[poem, "\n"];
wordArray = StringSplit[#] &   @ lines ;
reversedWordArray = Reverse[#] & /@ wordArray ;
linesWithReversedWords =
  StringJoin[Riffle[#, " "]] & /@ reversedWordArray;
finaloutput = StringJoin[Riffle[#, "\n"]] &  @ linesWithReversedWords
