select count(*) from dblp p1, dblp p2a, dblp p3a, dblp p3b, dblp p4a, dblp p4b, dblp p5a, dblp p5b
    
    WHERE p1.toNode = p2a.fromNode
        AND p2a.toNode = p3a.fromNode
            AND p3a.toNode = p4a.fromNode
                AND p4a.toNode = p5a.fromNode
            AND p3a.toNode = p4b.fromNode
                AND p4b.toNode = p5b.fromNode
        AND p2a.toNode = p3b.fromNode
            