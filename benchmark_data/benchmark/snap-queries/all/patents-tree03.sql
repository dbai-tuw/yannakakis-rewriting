select count(*) from patents p1, patents p2a, patents p3a, patents p3b, patents p4a, patents p4b, patents p5a, patents p5b
    
    WHERE p1.toNode = p2a.fromNode
        AND p2a.toNode = p3a.fromNode
            AND p3a.toNode = p4a.fromNode
                AND p4a.toNode = p5a.fromNode
            AND p3a.toNode = p4b.fromNode
                AND p4b.toNode = p5b.fromNode
        AND p2a.toNode = p3b.fromNode
            