preorder[a_Integer] := a;
preorder[a_[b__]] := Flatten@{a, preorder /@ {b}};
inorder[a_Integer] := a;
inorder[a_[b_, c_]] := Flatten@{inorder@b, a, inorder@c};
inorder[a_[b_]] := Flatten@{inorder@b, a}; postorder[a_Integer] := a;
postorder[a_[b__]] := Flatten@{postorder /@ {b}, a};
levelorder[a_] :=
 Flatten[Table[Level[a, {n}], {n, 0, Depth@a}]] /. {b_Integer[__] :>
    b};


