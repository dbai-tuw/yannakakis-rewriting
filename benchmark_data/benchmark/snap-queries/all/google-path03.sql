select count(*) from google p1, google p2, google p3, google p4 where p1.toNode = p2.fromNode AND p2.toNode = p3.fromNode AND p3.toNode = p4.fromNode
