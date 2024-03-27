SELECT COUNT(*) FROM wiki p1, wiki p2, wiki p3, wiki p4a, wiki p4b
WHERE p1.toNode = p2.fromNode
AND p2.toNode = p3.fromNode
    AND p3.toNode = p4a.fromNode
    AND p3.toNode = p4b.fromNode
