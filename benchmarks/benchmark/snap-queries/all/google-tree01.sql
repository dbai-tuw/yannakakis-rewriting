SELECT COUNT(*) FROM google p1, google p2, google p3, google p4a, google p4b
WHERE p1.toNode = p2.fromNode
AND p2.toNode = p3.fromNode
    AND p3.toNode = p4a.fromNode
    AND p3.toNode = p4b.fromNode
