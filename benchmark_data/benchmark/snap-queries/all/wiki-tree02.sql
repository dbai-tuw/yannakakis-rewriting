select count(*) from wiki p1, wiki p2a, wiki p3a, wiki p3b, wiki p4a, wiki p4b
    
    WHERE p1.toNode = p2a.fromNode
        AND p2a.toNode = p3a.fromNode
            AND p3a.toNode = p4a.fromNode
            AND p3a.toNode = p4b.fromNode
        AND p2a.toNode = p3b.fromNode
            