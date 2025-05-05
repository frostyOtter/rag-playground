#!/bin/bash

# Run neo4j service to ingest single data to database
# uv run src/db_service/neo4j_service.py ingest data/INFORMATION-TECHNOLOGY/10089434.pdf

# Run neo4j service to ingest a folder to database
# uv run src/db_service/neo4j_service.py ingest data/INFORMATION-TECHNOLOGY/

# Run neo4j service to query data from database
uv run src/db_service/neo4j_service.py query "I am looking for person with linux skill, can you recommend?"