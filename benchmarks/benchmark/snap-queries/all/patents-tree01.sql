SELECT COUNT(*) FROM patents p1, patents p2, patents p3, patents p4a, patents p4b
WHERE p1.toNode = p2.fromNode
AND p2.toNode = p3.fromNode
    AND p3.toNode = p4a.fromNode
    AND p3.toNode = p4b.fromNode
