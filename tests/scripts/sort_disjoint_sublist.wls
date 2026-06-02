vals = {7, 6, 5, 4, 3, 2, 1, 0}; indices = {7, 2, 8};
vals[[Sort[indices]]] = Sort[vals[[indices]]];
Print[vals]
