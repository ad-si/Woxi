chooseLine[file_] := Block[{strm = OpenRead[file], n = 1, rec, selected},
  					     rec = selected = Read[strm];
  					     While[rec =!= EndOfFile,
                                                    rec=Read[strm];
                                                    n++;
                                                    If[RandomReal[] < 1/n, selected = rec]];
                                             Close[strm];
                                             selected]
