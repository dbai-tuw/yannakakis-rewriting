#!/bin/bash

DATADIR=$(pwd)/lsqb/data/social-network-sf${SF}-merged-fk

PGPASSWORD=lsqb psql -h postgres -U lsqb -d lsqb -f lsqb/sql/drop.sql
PGPASSWORD=lsqb psql -h postgres -U lsqb -d lsqb -f lsqb/sql/schema.sql

sed "s|PATHVAR|${DATADIR}|" lsqb/sql/snb-load.sql | sed 's|COPY|\\copy|' | PGPASSWORD=lsqb psql -h postgres -U lsqb -d lsqb

PGPASSWORD=lsqb psql -h postgres -U lsqb -d lsqb -f lsqb/sql/views.sql

