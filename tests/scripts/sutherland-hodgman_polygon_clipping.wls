p1 = Polygon[{{50, 150}, {200, 50}, {350, 150}, {350, 300}, {250, 300}, {200, 250}, {150, 350}, {100, 250}, {100, 200}}];
p2 = Polygon[{{100, 100}, {300, 100}, {300, 300}, {100, 300}}];

RegionIntersection[p1, p2]

Graphics[{Red, p1, Blue, p2, Green, RegionIntersection[p1, p2]}]
