select count(*) from wiki p1, wiki p2, wiki p3 where p1.toNode = p2.fromNode AND p2.toNode = p3.fromNode
