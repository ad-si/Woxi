data = <|"account" -> "xyz", "password" -> "x", "UID" -> 1003,
   "GID" -> 1000, "fullname" -> "X Yz", "office" -> "Room 1003",
   "extension" -> "(234)555-8913", "homephone" -> "(234)555-0033",
   "email" -> "xyz@rosettacode.org", "directory" -> "/home/xyz",
   "shell" -> "/bin/bash"|>;
asString[data_] :=
  StringRiffle[
   ToString /@
    Insert[data /@ {"account", "password", "UID", "GID", "directory",
       "shell"},
     StringRiffle[
      data /@ {"fullname", "office", "extension", "homephone",
        "email"}, ","], 5], ":"];
fname = FileNameJoin[{$TemporaryDirectory, "testfile"}];
str = OpenWrite[fname]; (* Use OpenAppend if file exists *)
Close[str];
Print["Appended record: " <> asString[data]];
