#! /usr/bin/env bash

# Ensure jq is available (nix-profile or global)
if ! command -v jq &>/dev/null; then
  export PATH="$HOME/.nix-profile/bin:$PATH"
fi

if test -f /.dockerenv
then HOST=host.docker.internal
else HOST=localhost
fi

check() {
  payload=$(jq -n --arg code "$1" '{cmd: "wolframscript -code \($code | @sh)"}')
  expected=$(curl -s -X POST "http://$HOST:3456/exec" -d "$payload" | jq -r '.stdout | gsub("\\s+$"; "")')
  actual=$(cargo run --quiet -- eval "$1")

  if test "$expected" = "$actual"
  then echo -e "\n✅ $1"
  else
    echo -e "\n❌ $1"
    echo "  wolframscript: $expected"
    echo "           woxi: $actual"
  fi
}

check 'Integrate[x * Sin[x],x]'
