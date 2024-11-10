from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.groq import Groq
from llama_index.core import SQLDatabase
import os

from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = os.getenv('groq_api')
HF_TOKEN = os.getenv('HF_TOKEN')

print(GROQ_API_KEY ," ",HF_TOKEN)

from IPython.display import Markdown, display

from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    String,
    Integer,
    select,
    Float
)


from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float
import pandas as pd


from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Step 1: Create an in-memory SQLite database
engine = create_engine("sqlite:///:memory:")
metadata_obj = MetaData()

# Step 2: Define the 'products' table
products_table = Table(
    'products',
    metadata_obj,
    Column('user_name', String),
    Column('user_cat', String),
    Column('usage', Integer),
    Column('cat_average', Float),
    Column('cat_importance', Float),
    Column('ratio', Float)
)

# Step 3: Create the table in the database
metadata_obj.create_all(engine)

# Step 4: Read data from a CSV file into a DataFrame
# Assuming your CSV file is named 'products.csv'
df = pd.read_csv('user_category_data.csv')

# Step 5: Drop the 'index' column from the DataFrame if it exists
df = df.drop(columns=['index'], errors='ignore')

# Step 6: Insert the data into the 'products' table without including the DataFrame index
df.to_sql('products', con=engine, if_exists='append', index=False)

# Verify by querying the data (Optional)
with engine.connect() as conn:
    result = conn.execute(products_table.select())
    for row in result:
        print(row)
sql_database = SQLDatabase(engine, include_tables=["products"])

llm = Groq(model="llama3-70b-8192", api_key=GROQ_API_KEY)
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")


from llama_index.core.query_engine import NLSQLTableQueryEngine

query_engine = NLSQLTableQueryEngine(
    sql_database=sql_database, tables=["products"], llm=llm, embed_model=embed_model
)

def llama_index_isgreat():
    query_str = "tell me about User1"
    response = query_engine.query(query_str)
    print(response)