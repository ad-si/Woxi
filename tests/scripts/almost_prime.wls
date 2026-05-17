kprimes[k_,n_] :=
  (* generates a list of the n smallest k-almost-primes *)
  Module[{firstnprimes, runningkprimes = {}},
  firstnprimes = Prime[Range[n]];
  runningkprimes = firstnprimes;
  Do[
   runningkprimes =
     Outer[Times, firstnprimes , runningkprimes ] // Flatten // Union  // Take[#, n] & ;
   (* only keep lowest n numbers in our running list *)
   , {i, 1, k - 1}];
  runningkprimes
  ]
(* now to create table with n=10 and k ranging from 1 to 5 *)
Table[Flatten[{"k = " <> ToString[i] <> ": ", kprimes[i, 10]}], {i,1,5}] // TableForm
