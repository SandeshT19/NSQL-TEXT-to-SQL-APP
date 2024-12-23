import logging
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from llama_index.core import SQLDatabase, PromptTemplate
##mapping all taable schema informaation 
from llama_index.core.objects import ObjectIndex
from llama_index.core.objects import SQLTableNodeMapping, SQLTableSchema
import pandas as pd
from llama_index.core.indices.struct_store import SQLTableRetrieverQueryEngine
from llama_index.core.query_pipeline import (
    QueryPipeline as QP,
    FnComponent,
    InputComponent,
    Link,
    CustomQueryComponent,
)
from llama_index.llms.llama_cpp import LlamaCPP
from transformers import AutoTokenizer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.settings import Settings

import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO, force=True)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
from IPython.display import Markdown, display

from llama_index.core.indices.struct_store import SQLTableRetrieverQueryEngine
from llama_index.core import VectorStoreIndex
from llama_index.core.prompts.default_prompts import DEFAULT_TEXT_TO_SQL_PROMPT
from llama_index.core import PromptTemplate
#Testing connectivity with DB :
from sqlalchemy import create_engine, text

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Database connection setup
def setup_engine(connection_string):
    """Create and test database connection."""
    engine = create_engine(connection_string)
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        logging.info("Database connection established successfully.")
    except SQLAlchemyError as e:
        logging.error(f"Database connection failed: {e}")
        raise
    return engine

# Initialize tokenizer and embedding models
def setup_tokenizer_and_embeddings(tokenizer_path, embedding_model, cache_folder):
    """Set up tokenizer and embedding models."""
    AutoTokenizer.from_pretrained(tokenizer_path).encode
    Settings.embed_model = HuggingFaceEmbedding(model_name=embedding_model,cache_folder=cache_folder) # use Huggingface embeddings

    return Settings.embed_model





# Initialize LLM
def setup_llm(model_path, context_window=3900, n_gpu_layers=2):
    """Set up LlamaCPP LLM."""
    return LlamaCPP(
        model_path=model_path,
        temperature=0.4,
        max_new_tokens=256,
        context_window=context_window,
        model_kwargs={"n_gpu_layers": n_gpu_layers},
        verbose=False,
    )

# Set up table schemas
def create_table_schema_objects(sql_database, table_details):
    """Create SQL table schema objects."""
    table_schema_objs = []
    for table in sql_database._all_tables:
        if table in table_details:
            table_schema_objs.append(
                SQLTableSchema(table_name=table, context_str=table_details[table])
            )
    return table_schema_objs

def parse_sql_query(response: str) -> str:
    """Extracts SQL query from LLM response with error handling."""
    response=str(response)
    sql_start = response.find("SQLQuery:")
    if sql_start != -1:
        sql_query = response[sql_start + len("SQLQuery:"):].strip().strip("```").strip()
        if sql_query:
            return sql_query
        else:
            raise ValueError("SQL query is empty in LLM response.")
    else:
        raise ValueError("SQL query not found in LLM response.")

# Query Pipeline setup
def setup_query_pipeline(sql_database, table_schema_objs, llm, response_synthesis_prompt):
    """Create and set up a query pipeline."""
    sql_retriever = SQLTableRetrieverQueryEngine(
        sql_database, ObjectIndex.from_objects(
            table_schema_objs,
            SQLTableNodeMapping(sql_database),
            VectorStoreIndex
        ).as_retriever(similarity_top_k=1),
        llm
    )

    qp = QP(
        modules={
            "input": InputComponent(),
            "table_retriever": sql_retriever,
            "text2sql_prompt": DEFAULT_TEXT_TO_SQL_PROMPT.partial_format(
                dialect=sql_database.engine.dialect.name
            ),
            "text2sql_llm": llm,
            "sql_output_parser": FnComponent(fn=parse_sql_query),
            "sql_retriever": sql_retriever,
            "response_synthesis_prompt": response_synthesis_prompt,
            "response_synthesis_llm": llm,
        },
        verbose=True,
    )

    qp.add_chain(["input", "table_retriever"])
    qp.add_link("input", "text2sql_prompt", dest_key="query_str")
    qp.add_link("table_retriever", "text2sql_prompt", dest_key="schema")
    qp.add_chain(["text2sql_prompt", "text2sql_llm", "sql_output_parser", "sql_retriever"])
    qp.add_link("sql_output_parser", "response_synthesis_prompt", dest_key="sql_query")
    qp.add_link("sql_retriever", "response_synthesis_prompt", dest_key="context_str")
    qp.add_link("input", "response_synthesis_prompt", dest_key="query_str")
    qp.add_link("response_synthesis_prompt", "response_synthesis_llm")

    return qp

# Run Query Pipeline
def run_query_pipeline(qp, user_query):
    """Run query pipeline from user input to response."""
    try:
        response = qp.run(query=user_query)
        return response
    except ValueError as ve:
        logging.error(f"Value error in query pipeline: {ve}")
        return "An error occurred while processing your query. Please check your input and try again."
    except Exception as e:
        logging.error(f"Unexpected error in query pipeline: {e}")
        return "An unexpected error occurred. Please try again later."

# Main function
def main():
    # Configuration
    
# Construct the connection string
    CONNECTION_STRING = "your_connection_string_here"
    TOKENIZER_PATH = "your_tokenizer_path_here"
    EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
    CACHE_FOLDER = "path_to_cache_folder"
    MODEL_PATH = "path_to_llama_model"

    TABLE_DETAILS = {
        "customer": "Table storing essential details of customers",
        "actor": "Table storing essential details of actors",
        "staff": "Stores detailed information about staff in an organization"
    }

    # Initialize components
    engine = setup_engine(CONNECTION_STRING)
    setup_tokenizer_and_embeddings(TOKENIZER_PATH, EMBEDDING_MODEL, CACHE_FOLDER)
    llm = setup_llm(MODEL_PATH)
    sql_database = SQLDatabase(engine)
    table_schema_objs = create_table_schema_objects(sql_database, TABLE_DETAILS)
    response_synthesis_prompt = PromptTemplate(
        "Given an input question, synthesize a response from the query results.\n"
        "Query: {query_str}\n"
        "SQLQuery: {sql_query}\n"
        "SQLResult: {context_str}\n"
        "Response: "
    )
    qp = setup_query_pipeline(sql_database, table_schema_objs, llm, response_synthesis_prompt)

    # Example usage
    user_query = "provide first name only for top 5 actors?"
    response = run_query_pipeline(qp, user_query)
    print(response)

if __name__ == "__main__":
    main()
