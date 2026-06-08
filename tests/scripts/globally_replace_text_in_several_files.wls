listOfFiles = {"a.txt", "b.txt", "c.txt"};
Do[Export[f, "Goodbye London!", "Text"], {f, listOfFiles}];
Do[
  filetext = Import[filename, "Text"];
  filetext = StringReplace[filetext, "Goodbye London!" -> "Hello New York!"];
  Export[filename, filetext, "Text"];
  Print[filename, ": ", Import[filename, "Text"]],
  {filename, listOfFiles}]
