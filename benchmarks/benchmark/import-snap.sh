#!/bin/bash

PGPASSWORD=snap psql -h postgres -U snap -d snap -f import-snap.sql
