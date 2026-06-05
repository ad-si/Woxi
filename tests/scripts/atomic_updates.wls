(* `bucks = RandomInteger[10, 20]` normally seeds 20 random balances, and two
   parallel threads move random amounts between buckets forever (an "atomic"
   transfer that conserves the total). Hardcoded here to fixed balances and a
   fixed sequence of transfers, run sequentially, so the snapshot is
   deterministic; each transfer still conserves the total. *)
transfer[bucks_, src_, dest_, n_] :=
  ReplacePart[bucks, {src -> Max[bucks[[src]] - n, 0],
    dest -> bucks[[dest]] + Min[bucks[[src]], n]}];
bucks = {5, 3, 8, 1, 6, 2, 9, 4, 7, 0};
Print["Original sum: ", Total[bucks]];
bucks = transfer[bucks, 3, 7, 4];
bucks = transfer[bucks, 1, 10, 2];
bucks = transfer[bucks, 5, 2, 3];
Print["After transfers: ", bucks];
Print["Final sum: ", Total[bucks]]
