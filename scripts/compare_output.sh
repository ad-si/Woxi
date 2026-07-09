#! /usr/bin/env bash

check() {
  expected=$(wolframscript -code "$1" | sed -e 's/[[:space:]]*$//')
  actual=$(cargo run --quiet -- eval "$1")

  if test "$expected" = "$actual"
  then echo -e "\n✅ $1"
  else
    echo -e "\n❌ $1"
    echo "  wolframscript: $expected"
    echo "           woxi: $actual"
  fi
}

check 'ProductQ[_] = False; ProductQ[4]'
check 'HalfIntegerQ[u__] := False; HalfIntegerQ[1/2]' # See https://reference.wolfram.com/language/ref/BlankSequence.html
