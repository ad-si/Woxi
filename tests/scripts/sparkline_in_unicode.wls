toSparkline[data_String] := FromCharacterCode[Round[7 Rescale@Flatten@ImportString[data, "Table", "FieldSeparators" -> {" ", ","}]] + 16^^2581];
