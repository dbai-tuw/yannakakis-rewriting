select count(*) from patents p1, patents p2, patents p3, patents p4 where p1.toNode = p2.fromNode AND p2.toNode = p3.fromNode AND p3.toNode = p4.fromNode
