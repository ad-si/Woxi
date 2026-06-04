rpn[str_] :=
  StringRiffle[
   ToString /@
    Module[{in = StringSplit[str], stack = {}, out = {}, next},
     While[in != {}, next = in[[1]]; in = in[[2 ;;]];
      Which[DigitQ[next], AppendTo[out, next], LetterQ[next],
       AppendTo[stack, next], next == ",",
       While[stack[[-1]] != "(", AppendTo[out, stack[[-1]]];
        stack = stack[[;; -2]]], next == "^", AppendTo[stack, "^"],
       next == "*",
       While[stack != {} && MatchQ[stack[[-1]], "^" | "*" | "/"],
        AppendTo[out, stack[[-1]]]; stack = stack[[;; -2]]];
       AppendTo[stack, "*"], next == "/",
       While[stack != {} && MatchQ[stack[[-1]], "^" | "*" | "/"],
        AppendTo[out, stack[[-1]]]; stack = stack[[;; -2]]];
       AppendTo[stack, "/"], next == "+",
       While[stack != {} &&
         MatchQ[stack[[-1]], "^" | "*" | "/" | "+" | "-"],
        AppendTo[out, stack[[-1]]]; stack = stack[[;; -2]]];
       AppendTo[stack, "+"], next == "-",
       While[stack != {} &&
         MatchQ[stack[[-1]], "^" | "*" | "/" | "+" | "-"],
        AppendTo[out, stack[[-1]]]; stack = stack[[;; -2]]];
       AppendTo[stack, "-"], next == "(", AppendTo[stack, "("],
       next == ")",
       While[stack[[-1]] =!= "(", AppendTo[out, stack[[-1]]];
        stack = stack[[;; -2]]]; stack = stack[[;; -2]];
       If[StringQ[stack[[-1]]], AppendTo[out, stack[[-1]]];
        stack = stack[[;; -2]]]]];
     While[stack != {}, AppendTo[out, stack[[-1]]];
      stack = stack[[;; -2]]]; out]];
Print[rpn["3 + 4 * 2 / ( 1 - 5 ) ^ 2 ^ 3"]];
