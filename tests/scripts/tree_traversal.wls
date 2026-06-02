preorder[a_Integer] := a;
preorder[a_[b__]] := Flatten@{a, preorder /@ {b}};
inorder[a_Integer] := a;
inorder[a_[b_, c_]] := Flatten@{inorder@b, a, inorder@c};
inorder[a_[b_]] := Flatten@{inorder@b, a};
postorder[a_Integer] := a;
postorder[a_[b__]] := Flatten@{postorder /@ {b}, a};
tree = 1[2[4[7], 5], 3[6[8, 9]]];
Print[preorder[tree]];
Print[inorder[tree]];
Print[postorder[tree]]
