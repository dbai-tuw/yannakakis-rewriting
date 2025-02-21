SELECT COUNT(*) FROM dblp p1, dblp p2, dblp p3, dblp p4a, dblp p4b
WHERE p1.toNode = p2.fromNode
AND p2.toNode = p3.fromNode
    AND p3.toNode = p4a.fromNode
    AND p3.toNode = p4b.fromNode
