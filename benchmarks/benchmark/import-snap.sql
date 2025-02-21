
DROP TABLE IF EXISTS patents;
CREATE TABLE patents (fromNode integer, toNode integer);
\copy patents FROM 'snap/noheader/cit-Patents.txt' with (header false);

DROP TABLE IF EXISTS wiki;
CREATE TABLE wiki (fromNode integer, toNode integer);
\copy wiki FROM 'snap/noheader/wiki-topcats.txt' with (header false, delimiter ' ');

DROP TABLE IF EXISTS google;
CREATE TABLE google (fromNode integer, toNode integer);
\copy google FROM 'snap/noheader/web-Google.txt' with (header false);

DROP TABLE IF EXISTS dblp;
CREATE TABLE dblp (fromNode integer, toNode integer);
\copy dblp FROM 'snap/noheader/com-dblp.ungraph.txt' with (header false);
