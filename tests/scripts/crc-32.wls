type="CRC32"; (*pick one out of 13 predefined hash types*)
StringForm[
"The "<>type<>" hash code of \"``\" is ``.",
s="The quick brown fox jumps over the lazy dog",
Hash[s,type,"HexString"]
]
