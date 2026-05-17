f[doc_String, start_Integer, n_Integer] := Module[{p, newdoc},
  p = Import[doc, "List"];
  If[start + n - 1 <= Length@p,
   p = Drop[p, {start, start + n - 1}];
   newdoc = FileNameJoin[{DirectoryName[doc], FileBaseName[doc] <> "_removed.txt"}];
   Export[newdoc, p, "List"];
   ,
   Print["Too few lines in document. Operation aborted."]
  ]
 ]
