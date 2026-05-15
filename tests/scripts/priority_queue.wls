push = Function[{queue, priority, item},
   queue = SortBy[Append[queue, {priority, item}], First], HoldFirst];
pop = Function[queue,
   If[Length@queue == 0, Null,
    With[{item = queue[[-1, 2]]}, queue = Most@queue; item]],
   HoldFirst];
peek = Function[queue,
   If[Length@queue == 0, Null, Max[queue[[All, 1]]]], HoldFirst];
merge = Function[{queue1, queue2},
   SortBy[Join[queue1, queue2], First], HoldAll];


(* ========== *)

queue = {};
push[queue, 3, "Clear drains"];
push[queue, 4, "Feed cat"];
push[queue, 5, "Make tea"];
push[queue, 1, "Solve RC tasks"];
push[queue, 2, "Tax return"];
Print[peek[queue]];
Print[pop[queue]];
queue1 = {};
push[queue1, 6, "Drink tea"];
Print[merge[queue, queue1]];
