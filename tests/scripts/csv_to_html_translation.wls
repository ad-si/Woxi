a="Character,Speech
The multitude,The messiah! Show us the messiah!
Brians mother,<angry>Now you listen here! He's not the messiah;he's a very naughty boy! Now go away!</angry>
The multitude,Who are you?
Brians mother,I'm his mother;that's who!
The multitude,Behold his mother! Behold his mother!";
(*Naive*)
StringJoin["<table>\n",Map[StringJoin["<tr><td>",#,"</td></tr>\n"]&,
StringSplit[StringReplace[a,{","->"</td><td>","<"->"&lt;",">"->"&gt;"}],"\n"]]
,"</table>"]
(*Extra*)
StringJoin["<table>\n",StringJoin["<tr><th>",#,"</th></tr>\n"]&[
StringSplit[StringReplace[a,{","->"</td><td>","<"->"&lt;",">"->"&gt;"}],"\n"]//First]
,Map[StringJoin["<tr><td>",#,"</td></tr>\n"]&,
StringSplit[StringReplace[a,{","->"</td><td>","<"->"&lt;",">"->"&gt;"}],"\n"]//Rest]
,"</table>"]
