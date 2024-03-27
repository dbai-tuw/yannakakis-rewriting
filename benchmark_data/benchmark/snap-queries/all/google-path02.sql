select count(*) from google p1, google p2, google p3 where p1.toNode = p2.fromNode AND p2.toNode = p3.fromNode
