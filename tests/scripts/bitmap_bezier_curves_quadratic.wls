pts = {{0, 0}, {1, -1}, {2, 1}};
Graphics[{BSplineCurve[pts], Green, Line[pts], Red, Point[pts]}]


(* ========== *)

pts = {{0, 0}, {1, -1}, {2, 1}};
Graphics[{BezierCurve[pts], Green, Line[pts], Red, Point[pts]}]
