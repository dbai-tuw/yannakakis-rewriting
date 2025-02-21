select count(*) from dblp p1, dblp p2, dblp p3 where p1.toNode = p2.fromNode AND p2.toNode = p3.fromNode
