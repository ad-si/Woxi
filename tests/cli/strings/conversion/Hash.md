# `Hash`

Computes a hash of an expression.
The one-argument form uses an implementation-defined hash whose numeric
value differs between Woxi and Mathematica, so prefer an explicit algorithm:

```scrut
$ wo 'Hash["hello", "MD5", "HexString"]'
5d41402abc4b2a76b9719d911017c592
```

Beyond MD2, MD4, MD5, SHA, SHA224, SHA256, SHA384 and SHA512 there are the
SHA-3 family, the pre-standard Keccak padding Ethereum kept, RIPEMD160 and
the RIPEMD160-of-SHA256 a Bitcoin address is built from, plus CRC32 and
Adler32:

```scrut
$ wo 'Hash["abc", "SHA3-256", "HexString"]'
3a985da74fe225b2045c172d6bd390bd855f086e3e9d525b46bfe24511431532
```

```scrut
$ wo 'Hash["abc", "Keccak256", "HexString"]'
4e03657aea45a94fc7d47ba826c8d667c0d1e6e33a64a036ec44f58fa12d6c45
```

A `ByteArray` is hashed over its bytes, so it gives the same digest as the
string those bytes spell:

```scrut
$ wo 'Hash[ByteArray[{97, 98, 99}], "MD5", "HexString"]'
900150983cd24fb0d6963f7d28e17f72
```
