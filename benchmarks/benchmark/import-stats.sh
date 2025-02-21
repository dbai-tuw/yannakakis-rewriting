#!/bin/bash

DATADIR=$(pwd)/stats/datasets

PGPASSWORD=stats psql -h postgres -U stats -d stats -f stats/sql/schema.sql

sed "s|PATHVAR|${DATADIR}|" stats/sql/snb-load.sql | sed 's|COPY|\\copy|' | PGPASSWORD=stats psql -h postgres -U stats -d stats
