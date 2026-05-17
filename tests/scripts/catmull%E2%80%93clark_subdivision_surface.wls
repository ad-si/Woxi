subSetQ[large_,small_] := MemberQ[large,small]
subSetQ[large_,small_List] := And@@(MemberQ[large,#]&/@small)

containing[groupList_,item_]:= Flatten[Position[groupList,group_/;subSetQ[group,item]]]

ReplaceFace[face_]:=Transpose[Prepend[Transpose[{#[[1]],face,#[[2]]}&/@Transpose[Partition[face,2,1,1]//{#,RotateRight[#]}&]],face]]


