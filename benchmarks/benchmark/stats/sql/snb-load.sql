COPY badges                    FROM 'PATHVAR/badges.csv'                    (DELIMITER ',', HEADER, FORMAT csv);
COPY comments		       FROM 'PATHVAR/comments.csv'                  (DELIMITER ',', HEADER, FORMAT csv);
COPY postHistory               FROM 'PATHVAR/postHistory.csv'               (DELIMITER ',', HEADER, FORMAT csv);
COPY postLinks                 FROM 'PATHVAR/postLinks.csv'                 (DELIMITER ',', HEADER, FORMAT csv);
COPY posts                     FROM 'PATHVAR/posts.csv'                     (DELIMITER ',', HEADER, FORMAT csv);
COPY tags                      FROM 'PATHVAR/tags.csv'                      (DELIMITER ',', HEADER, FORMAT csv);
COPY users                     FROM 'PATHVAR/users.csv'                     (DELIMITER ',', HEADER, FORMAT csv);
COPY votes                     FROM 'PATHVAR/votes.csv'                     (DELIMITER ',', HEADER, FORMAT csv);
