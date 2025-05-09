import sys
import os

sys.path.append(
    os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
)
from loguru import logger
from typing import Optional, List
from langchain_neo4j import Neo4jGraph, Neo4jVector
from langchain_core.embeddings import Embeddings
from src.db_service.db_base import VectorDBBase
from src.models.common_pdf_models import DocumentChunk
from src.llm_service.llm_base import LLMProviderBase
from pydantic import BaseModel
import json


class Neo4jService(VectorDBBase):
    """
    Neo4j vector database implementation with progressive knowledge graph building.
    Implements all phases for resume processing as listed in our discussion.
    """

    def __init__(
        self,
        uri: str,
        username: str,
        password: str,
        database: str = "neo4j",
        **kwargs,
    ):
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        self.graph: Optional[Neo4jGraph] = (
            kwargs.get("graph", None)
            if kwargs.get("graph") and isinstance(kwargs.get("graph"), Neo4jGraph)
            else None
        )

    def connect(self, **kwargs) -> None:
        """Connect to Neo4j instance."""
        try:
            if not self.graph:
                self.graph = Neo4jGraph(
                    url=self.uri,
                    username=self.username,
                    password=self.password,
                    database=self.database,
                )
            logger.info(f"Connected to Neo4j")
            self.graph.refresh_schema()
            logger.info("Graph schema refreshed")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise

    def disconnect(self) -> None:
        """Disconnect from Neo4j."""
        self.graph = None
        logger.info("Disconnected from Neo4j")

    # Phase 1: Basic chunk storage
    def add_documents(
        self, chunks: List[DocumentChunk], source_file: str, **kwargs
    ) -> None:
        """Store raw chunks in Neo4j - Phase 1: Minimal Viable Product."""
        if not self.graph:
            raise ConnectionError("Not connected to Neo4j")

        logger.info(f"Adding {len(chunks)} chunks from {source_file}")
        try:
            for chunk in chunks:
                query = """
                MERGE (c:Chunk {id: $chunk_id})
                SET c.text = $text,
                    c.source = $source,
                    c.page_number = $page_number,
                    c.chunk_index = $chunk_index,
                    c.created_at = datetime(),
                    c.metadata = $metadata
                """

                params = {
                    "chunk_id": chunk.chunk_id,  # or f"{source_file}_{chunk.chunk_index}",
                    "text": chunk.content,
                    "source": source_file,
                    "page_number": chunk.page_number,
                    "chunk_index": chunk.chunk_index,
                    "metadata": json.dumps(chunk.metadata),
                }

                self.graph.query(query, params)

        except Exception as e:
            logger.error(f"Error when adding documents: {str(e)}")
            raise

        logger.success(f"Stored {len(chunks)} chunks for source: {source_file}")

    # Phase 2: Add relationship layers
    def enhance_with_metadata(self, source_file: str) -> None:
        """Phase 2: Add metadata and create document structure."""
        if not self.graph:
            raise ConnectionError("Not connected to Neo4j")

        # Create Resume document node
        resume_query = """
        MERGE (r:Resume {id: $source})
        SET r.source = $source,
            r.created_at = datetime(),
            r.processed = false
        RETURN r.id as id
        """
        result = self.graph.query(resume_query, {"source": source_file})
        logger.info(f"Created Resume node for: {source_file}")

        # Link chunks to resume
        link_query = """
        MATCH (c:Chunk {source: $source}), (r:Resume {id: $source})
        MERGE (c)-[:PART_OF]->(r)
        RETURN count(*) as linked
        """
        result = self.graph.query(link_query, {"source": source_file})
        logger.info(f"Linked {result[0]['linked']} chunks to resume")

        # Create sequential relationships
        sequence_query = """
        MATCH (c1:Chunk {source: $source}), (c2:Chunk {source: $source})
        WHERE c1.chunk_index + 1 = c2.chunk_index
        MERGE (c1)-[:NEXT]->(c2)
        RETURN count(*) as connected
        """
        result = self.graph.query(sequence_query, {"source": source_file})
        logger.info(f"Created {result[0]['connected']} NEXT relationships")

    # Phase 3: Extracted entities
    def extract_entities(self, source_file: str, entity_schema: BaseModel, llm_service: Optional[LLMProviderBase]) -> None:
        """Phase 3: Extract entities from chunks using Gemini."""
        if not self.graph:
            raise ConnectionError("Not connected to Neo4j")

        # Entity extraction prompt
        extraction_prompt = """
        Follow the schema to extract the following information from the resume chunk.
        If information isn't found, use empty string or empty array.
        
        Resume chunk:
        {chunk_text}
        """

        chunks_query = """
        MATCH (c:Chunk {source: $source})
        RETURN c.id as chunk_id, c.text as text, c.chunk_index as index
        ORDER BY c.chunk_index ASC
        LIMIT 10
        """
        try:
            results = self.graph.query(chunks_query, {"source": source_file})
            if not results:
                logger.warning(f"No unprocessed chunks found for source: {source_file}")
                return
            logger.debug(f"Found {len(results)} unprocessed chunks.")
            for result in results:
                try:
                    # Prepare messages
                    messages = [
                        {
                            "role": "system",
                            "content": "You are an expert resume parser.",
                        },
                        {
                            "role": "human",
                            "content": extraction_prompt.replace(
                                "{chunk_text}", result["text"]
                            ),
                        },
                    ]
                    
                    entities = llm_service.generate_response(messages=messages, response_schema=entity_schema)

                    # Store entities in graph
                    self._store_extracted_entities(result["chunk_id"], entities)

                    # Mark chunk as processed
                    mark_processed_query = """
                    MATCH (c:Chunk {id: $chunk_id})
                    SET c.entities_extracted = true,
                        c.extraction_timestamp = datetime()
                    """
                    self.graph.query(
                        mark_processed_query, {"chunk_id": result["chunk_id"]}
                    )

                    logger.info(f"Extracted entities from chunk: {result['chunk_id']}")

                except Exception as e:
                    logger.error(
                        f"Error when extracting entities from chunk {result['chunk_id']}: {e}"
                    )
        except Exception as e:
            logger.error(f"Error in extract_entities: {e}")

    def _store_extracted_entities(self, chunk_id: str, entities: dict) -> None:
        """Store extracted entities in Neo4j graph."""
        try:
            # Store person information
            if entities.get("person") and entities["person"].get("name"):
                person = entities["person"]
                person_query = """
                MERGE (p:Person {name: $name})
                SET p.email = $email,
                    p.phone = $phone,
                    p.created_at = datetime()
                WITH p
                MATCH (c:Chunk {id: $chunk_id})
                MERGE (c)-[:CONTAINS]->(p)
                """
                self.graph.query(
                    person_query,
                    {
                        "name": person["name"],
                        "email": person.get("email", ""),
                        "phone": person.get("phone", ""),
                        "chunk_id": chunk_id,
                    },
                )

            # Store companies and work experience
            for company in entities.get("companies", []):
                if company.get("name"):
                    company_query = """
                    MERGE (comp:Company {name: $company_name})
                    SET comp.created_at = datetime()
                    MERGE (role:Role {title: $role})
                    SET role.created_at = datetime()
                    MERGE (exp:Experience {
                        id: $exp_id
                    })
                    SET exp.start_date = $start_date,
                        exp.end_date = $end_date,
                        exp.created_at = datetime()
                    MERGE (exp)-[:AT_COMPANY]->(comp)
                    MERGE (exp)-[:HAS_ROLE]->(role)
                    WITH comp, role, exp
                    MATCH (c:Chunk {id: $chunk_id})
                    MERGE (c)-[:CONTAINS]->(comp)
                    MERGE (c)-[:CONTAINS]->(role)
                    MERGE (c)-[:MENTIONS]->(exp)
                    """
                    exp_id = f"{company['name']}_{company.get('role', 'unknown')}_{company.get('start_date', '')}"
                    self.graph.query(
                        company_query,
                        {
                            "company_name": company["name"],
                            "role": company.get("role", ""),
                            "exp_id": exp_id,
                            "start_date": company.get("start_date", ""),
                            "end_date": company.get("end_date", ""),
                            "chunk_id": chunk_id,
                        },
                    )

            # Store skills
            for skill in entities.get("skills", []):
                if skill:
                    skill_query = """
                    MERGE (s:Skill {name: $skill})
                    SET s.created_at = datetime()
                    WITH s
                    MATCH (c:Chunk {id: $chunk_id})
                    MERGE (c)-[:CONTAINS]->(s)
                    """
                    self.graph.query(
                        skill_query, {"skill": skill, "chunk_id": chunk_id}
                    )

            # Store education
            for edu in entities.get("education", []):
                if edu.get("institution"):
                    edu_query = """
                    MERGE (e:Education {
                        id: $edu_id
                    })
                    SET e.institution = $institution,
                        e.degree = $degree,
                        e.year = $year,
                        e.created_at = datetime()
                    WITH e
                    MATCH (c:Chunk {id: $chunk_id})
                    MERGE (c)-[:CONTAINS]->(e)
                    """
                    edu_id = f"{edu['institution']}_{edu.get('degree', '')}"
                    self.graph.query(
                        edu_query,
                        {
                            "edu_id": edu_id,
                            "institution": edu["institution"],
                            "degree": edu.get("degree", ""),
                            "year": edu.get("year", ""),
                            "chunk_id": chunk_id,
                        },
                    )

            # Store certifications
            for cert in entities.get("certifications", []):
                if cert:
                    cert_query = """
                    MERGE (cert:Certification {name: $cert})
                    SET cert.created_at = datetime()
                    WITH cert
                    MATCH (c:Chunk {id: $chunk_id})
                    MERGE (c)-[:CONTAINS]->(cert)
                    """
                    self.graph.query(cert_query, {"cert": cert, "chunk_id": chunk_id})

        except Exception as e:
            logger.error(f"Error storing entities for chunk {chunk_id}: {e}")

    # Phase 4: Connect all entities and create advanced relationships
    def create_knowledge_graph(self, source_file: str) -> None:
        """Phase 4: Connect all entities and create advanced relationships."""
        if not self.graph:
            raise ConnectionError("Not connected to Neo4j")

        # Connect skills to person using existing entities
        connect_skills_query = """
        MATCH (p:Person), (s:Skill), (c:Chunk {source: $source})
        WHERE (c)-[:CONTAINS]->(p) AND (c)-[:CONTAINS]->(s)
        MERGE (p)-[:HAS_SKILL]->(s)
        RETURN count(DISTINCT s) as skills_linked
        """

        # Connect person to experiences
        connect_experience_query = """
        MATCH (p:Person), (exp:Experience), (c:Chunk {source: $source})
        WHERE (c)-[:CONTAINS]->(p) AND (c)-[:MENTIONS]->(exp)
        MERGE (p)-[:HAS_EXPERIENCE]->(exp)
        RETURN count(DISTINCT exp) as experiences_linked
        """
        # Connect person to educations
        connect_education_query = """
        MATCH (p:Person), (edu:Education), (c:Chunk {source: $source})
        WHERE (c)-[:CONTAINS]->(p) AND (c)-[:CONTAINS]->(edu)
        MERGE (p)-[:HAS_EDUCATION]->(edu)
        RETURN count(DISTINCT edu) as education_linked
        """
        
        # Create resume-level aggregations
        create_resume_summary_query = """
        MATCH (r:Resume {id: $source})
        MATCH (c:Chunk)-[:PART_OF]->(r)
        WITH r
        MATCH (p:Person), (s:Skill), (exp:Experience), (edu:Education)
        WHERE EXISTS((p)-[:HAS_SKILL]->(s)) AND 
            EXISTS((p)-[:HAS_EXPERIENCE]->(exp)) AND
            EXISTS((p)-[:HAS_EDUCATION]->(edu))
        MERGE (r)-[:DESCRIBES]->(p)
        MERGE (r)-[:CONTAINS_SKILLS]->(s)
        MERGE (r)-[:MENTIONS_EXPERIENCE]->(exp)
        MERGE (r)-[:INCLUDES_EDUCATION]->(edu)
        RETURN count(*) as summary_created
        """

        results = []
        logger.info(f"Connecting skills")
        results.append(self.graph.query(connect_skills_query, {"source": source_file}))
        logger.info(f"Connecting experiences")
        results.append(
            self.graph.query(connect_experience_query, {"source": source_file})
        )
        logger.info(f"Connecting educations")
        results.append(
            self.graph.query(connect_education_query, {"source": source_file})
        )
        logger.info(f"Creating resume summary")
        results.append(
            self.graph.query(create_resume_summary_query, {"source": source_file})
        )


    def get_vector_index(
        self, node_label:str = "Chunk", text_node_properties:List[str]=["text"], embedding_node_property:str="embedding", embedding_service:Embeddings=None
    ) -> Neo4jVector:
        """Get Neo4j Vector Index."""
        if not self.graph:
            raise ConnectionError("Not connected to Neo4j")
        
        vector_index = Neo4jVector.from_existing_graph(
            embedding=embedding_service,
            url=self.uri,
            username=self.username,
            password=self.password,
            index_name=self.database,
            node_label=node_label,
            text_node_properties=text_node_properties,
            embedding_node_property=embedding_node_property,
        )

        return vector_index

    def get_by_ids(self, ids: List[str], **kwargs):
        """Get chunks by their IDs."""
        if not self.graph:
            raise ConnectionError("Not connected to Neo4j")

        query = """
        UNWIND $ids as id
        MATCH (c:Chunk {id: id})
        RETURN c
        """

        results = self.graph.query(query, {"ids": ids})
        return results

    def delete(self, source_file: str, **kwargs) -> None:
        """Delete all data related to a resume."""
        if not self.graph:
            raise ConnectionError("Not connected to Neo4j")
        logger.warning(f"Deleting all data for resume: {source_file}")
        logger.warning(
            "Please rerun with kawrgs: {'confirm': True} to confirm deletion"
        )
        if not kwargs.get("confirm", False):
            return
        query = """
        MATCH (r:Resume {id: $source})
        OPTIONAL MATCH (r)<-[:PART_OF]-(c:Chunk)
        OPTIONAL MATCH (c)-[:CONTAINS]->(entity)
        DETACH DELETE r, c, entity
        """

        self.graph.query(query, {"source": source_file})
        logger.info(f"Deleted all data for resume: {source_file}")

    def get_collection_info(self, **kwargs):
        """Get information about the knowledge graph."""
        if not self.graph:
            raise ConnectionError("Not connected to Neo4j")

        info_query = """
        MATCH (n)
        RETURN labels(n) as label, count(n) as count
        ORDER BY count DESC
        """

        results = self.graph.query(info_query)
        return results

    def query(self, user_message: str, limit: int = 5, **kwargs) -> str:
        """Query the knowledge graph using natural language."""
        if not self.graph:
            raise ConnectionError("Not connected to Neo4j")
        
        if not self.llm_service:
            raise ValueError("LLM service not initialized. Call initialize_gemini_llm_service first.")
            
        try:
            # Initialize the Cypher QA chain
            qa_chain = self.llm_service.initialize_cypher_qa_chain(self.graph)
            
            # Run the query
            response = qa_chain.run(user_message)
            return response
            
        except Exception as e:
            logger.error(f"Error querying knowledge graph: {e}")
            raise
    
    def ask(self, user_message: str, limit: int = 5, **kwargs) -> str:
        raise DeprecationWarning("Beep Boop, The function ask is deprecated. Use query instead.")


if __name__ == "__main__":
    task = sys.argv[1]
    if task not in ["ingest", "query"]:
        print("Please provide a valid command: ingest or query")
        sys.exit(1)

    from dotenv import load_dotenv

    load_dotenv(override=True)
    # Get Neo4j credentials from environment variables
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    username = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")

    # Initialize service
    neo4j = Neo4jService(uri=uri, username=username, password=password)
    # Connect to Neo4j
    neo4j.connect()

    if task == "ingest":
        input_path = sys.argv[2]
        if len(input_path) == 0:
            print("Please provide input and output paths.")
            sys.exit(1)
        # Determine if input is a directory or file
        if not input_path.endswith(".pdf"):
            # Input is a directory
            all_files = [
                os.path.join(input_path, f)
                for f in os.listdir(input_path)
                if f.endswith(".pdf")
            ]
            if len(all_files) == 0:
                print("No PDF files found in the directory.")
                sys.exit(1)

        else:
            # Input is a single PDF file
            all_files = [input_path]

        # Initialize data processor
        from src.data_service.pymupdf4llm_processor import PyMuPDF4LLMProcessor

        data_processor = PyMuPDF4LLMProcessor()

        try:
            from tqdm import tqdm
            # Example resume file
            for file in tqdm(all_files, total=len(all_files), desc="Beep Boop Beep Boop"):
                source_file = os.path.basename(file).replace(".pdf", "")
                logger.info(f"Chunking document: {source_file}")
                chunks = data_processor.chunks_document(file)
                
                # Phase 1: Store raw chunks
                logger.info(f"Phase 1: Storing raw chunks")
                neo4j.add_documents(chunks, source_file)
                # Phase 2: Add metadata and structure
                logger.info(f"Phase 2: Adding metadata and structure")
                neo4j.enhance_with_metadata(source_file)

                # Phase 3: Extract entities
                logger.info(f"Phase 3: Extracting entities using LLM")
                gemini_api_key = os.getenv("GEMINI_API_KEY")
                from src.llm_service.gemini_service import GeminiServiceProvider
                gemini_llm_service = GeminiServiceProvider(api_key=gemini_api_key)
                from src.models.resume_models import ResumeEntities
                neo4j.extract_entities(source_file, ResumeEntities, gemini_llm_service)

                # Phase 4: Create knowledge graph
                logger.info(f"Phase 4: Creating knowledge graph")
                neo4j.create_knowledge_graph(source_file)

                logger.success(f"Successfully processed file: {file}")

        except Exception as e:
            logger.error(f"Error processing file {file}: {e}")
        finally:
            # Always disconnect
            neo4j.disconnect()
    if task == "query":
        user_message = sys.argv[2]
        try:
            from src.llm_service.gemini_service import GeminiServiceProvider
            gemini_api_key = os.getenv("GEMINI_API_KEY")
            gemini_llm_service = GeminiServiceProvider(api_key=gemini_api_key)
            cypher_qa_chain_client = gemini_llm_service.initialize_cypher_qa_chain(neo4j.graph)
            # Example query
            logger.info(f"Message: {user_message}")
            query_results = cypher_qa_chain_client.invoke({"query": user_message})
            # query_results = neo4j.query(user_message=user_message, llm_service=gemini_llm_service)
            logger.info(f"Query results: {query_results}")

        except Exception as e:
            logger.error(f"Error in Neo4j example run: {e}")
        finally:
            # Always disconnect
            neo4j.disconnect()
            logger.info("Example run completed.")
