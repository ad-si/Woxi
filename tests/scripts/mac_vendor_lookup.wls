macLookup[mac_String] := Quiet[Check[Import["http://api.macvendors.com/" <> mac], "N/A"]]


