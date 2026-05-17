pentaFlake[0] = RegularPolygon[5];
pentaFlake[n_] :=
 GeometricTransformation[pentaFlake[n - 1],
  TranslationTransform /@
   CirclePoints[{GoldenRatio^(2 n - 1), Pi/10}, 5]]

Graphics@pentaFlake[4]
