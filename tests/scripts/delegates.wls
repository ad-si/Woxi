delegator[del_]@operate :=
  If[StringQ[del@operate], del@operate, "default implementation"];
del1 = Null;
del2@banana = "phone";
del3@operate = "delegate implementation";
Print[delegator[#]@operate] & /@ {del1, del2, del3};
