img = Image[ConstantArray[{1, 0, 0}, {1000, 1000}]];
img = ReplacePart[img, {1, 1, 1} -> {0, 0, 1}];
ImageValue[img, {1, 1}]


