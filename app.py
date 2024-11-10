from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.groq import Groq
from llama_index.core import SQLDatabase
from langchain.memory import ConversationBufferMemory
import os

from flask import Flask,jsonify,request,abort
app = Flask(__name__)

from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = os.getenv('groq_api')
HF_TOKEN = os.getenv('HF_TOKEN')

#rint(GROQ_API_KEY ," ",HF_TOKEN)

from IPython.display import Markdown, display

from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    String,
    Integer,
    select,
    Float,
    Boolean,
    DateTime
)


from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float
import pandas as pd

#from langchain.chains import LLMChain
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import requests
from bs4 import BeautifulSoup
from langchain.chains import ConversationalRetrievalChain
'''def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")'''


def get_webpage_text(urls):
    text = ""
    for url in urls:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            paragraphs = soup.find_all('p')
            for paragraph in paragraphs:
                text += paragraph.get_text() + "\n"
        else:
            print(f"Failed to retrieve content from {url}")
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index_webpages")

# Usage
urls = ["https://youth.europa.eu/get-involved/sustainable-development/how-reduce-my-carbon-footprint_en", 
        "https://news.climate.columbia.edu/2018/12/27/35-ways-reduce-carbon-footprint/"]

webpage_text = get_webpage_text(urls)
text_chunks = get_text_chunks(webpage_text)
get_vector_store(text_chunks)


engine = create_engine("sqlite:///:memory:")
metadata_obj = MetaData()

# Step 2: Define the 'user_statistics' table with only the specified columns
engine = create_engine("sqlite:///:memory:")
metadata_obj = MetaData()

# Step 2: Define the 'user_statistics' table with the specified columns, including ratio columns
user_statistics_table = Table(
    'user_statistics',
    metadata_obj,
    Column('microwave_usage', Integer),
    Column('refrigerator_usage', Float),
    Column('tv_usage', Float),
    Column('laptop_usage', Float),
    Column('eating_habits', Float),
    Column('total_power_consumption', Float),
    Column('vehicle_monthly_distance_km', Float),
    Column('frequency_of_traveling_by_air', Float),
    Column('vehicle_type', Float),
    Column('new_clothes_monthly', Float),
    Column('name', String),
    Column('role', String),
    Column('is_weekend', Boolean),
    Column('tv_usage_ratio', Float),
    Column('microwave_usage_ratio', Float),
    Column('laptop_usage_ratio', Float),
    Column('eating_habits_ratio', Float),
    Column('total_power_ratio', Float),
    Column('vehicle_distance_ratio', Float)
)

# Step 3: Create the table in the database
metadata_obj.create_all(engine)

# Step 4: Read data from a CSV file into a DataFrame
df = pd.read_csv('indian_family_lifestyle_data.csv')

# Step 5: Keep only the specified columns
columns_to_keep = [
    'microwave_usage', 'refrigerator_usage', 'tv_usage', 
    'laptop_usage', 'eating_habits', 'total_power_consumption', 
    'vehicle_monthly_distance_km', 'frequency_of_traveling_by_air', 
    'vehicle_type', 'new_clothes_monthly', 'name', 'role', 'is_weekend'
]
df = df[columns_to_keep]

# Step 6: Calculate usage ratios and add them as new columns
df['tv_usage_ratio'] = df['tv_usage'] / 171
df['microwave_usage_ratio'] = df['microwave_usage'] / 8
df['laptop_usage_ratio'] = df['laptop_usage'] / 500
df['eating_habits_ratio'] = df['eating_habits'] / 0.7
df['total_power_ratio'] = df['total_power_consumption'] / 15
df['vehicle_distance_ratio'] = df.apply(
    lambda row: (row['vehicle_monthly_distance_km'] / 1500) / 2 if row['vehicle_type'] == 'electric' else row['vehicle_monthly_distance_km'] / 1500,
    axis=1
)

# Step 7: Insert the data into the 'user_statistics' table without including the DataFrame index
df.to_sql('user_statistics', con=engine, if_exists='append', index=False)
metadata = MetaData()

# Reflect the existing tables from the database into the metadata
metadata.reflect(bind=engine)
user_statistics_table = metadata.tables['user_statistics']

# Print the table columns
for column in user_statistics_table.columns:
    print(column.name, column.type)

# Query the table and print the results
with engine.connect() as conn:
    result = conn.execute(user_statistics_table.select())
    for row in result:
        print("yo: ",row)
# Verify by querying the data (Optional)
with engine.connect() as conn:
    result = conn.execute(user_statistics_table.select())
    for row in result:
        print(row)

# Step 6: Insert the data into the 'user_statistics' table without including the DataFrame index
df.to_sql('user_statistics', con=engine, if_exists='append', index=False)

# Verify by querying the data (Optional)
with engine.connect() as conn:
    result = conn.execute(user_statistics_table.select())
    for row in result:
        print(row)



sql_database = SQLDatabase(engine, include_tables=["user_statistics"])
print(sql_database._all_tables)
# Step 1: Create an in-memory SQLite database


llm = Groq(model="llama3-70b-8192", api_key=GROQ_API_KEY)
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")


from llama_index.core.query_engine import NLSQLTableQueryEngine


#def llama_index_isgreat():
def llama_is_awesome(query_str,user_name):
    query_engine = NLSQLTableQueryEngine(
    sql_database=sql_database, tables=['user_statistics'], llm=llm, embed_model=embed_model
)
    print("Llama_indexes ",sql_database._all_tables)
    prompt = query_str 
    response = query_engine.query(query_str)
    print(response)
    return response

from langchain.chains.combine_documents import create_stuff_documents_chain
    
def get_conversational_chain():
    # Updated prompt template to align with input variables
    prompt_template = """
    You are an advanced environmentalist chatbot, dedicated to helping users reduce their carbon footprint and make sustainable choices.
     the data of user will be given in the form of data  will have many ratios like TV usage ratio. If the usage is greater than 1, it is using more than average, and if less than 1, it should be less than 1.
    With the data, you can help the user answer their query and help them reduce carbon footprints.

    You can highlight the part where the ratio's are less than 1 as positibe sentiments and reply those sentiments and one's with more than 1 as negative.
    
    You need to understand the sentiment of the query and reply with respect to that.
    when a user says I , me .... he is refering to the username so while when the user is saying can you help me analyze my carbon emissions you will refer the user with his user_name. Also the user_name will be provided in the data will be the same person we are talking about

    user: {user_name}
    query: {query}
    data: {data}
    context: {context}
    Answer:
    """

    # Define the language model and prompt for QA chain
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["user_name", "query", "data", "context"])
    
    # Use load_qa_chain with the specified chain type and prompt
    chain = create_stuff_documents_chain(llm, prompt)

    return chain

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Allow dangerous deserialization
new_db = FAISS.load_local("faiss_index_webpages", embeddings, allow_dangerous_deserialization=True)
#docs = new_db.similarity_search(query)
@app.route('/chat', methods=['POST'])
def user_input():
    print(sql_database._all_tables)
    data = request.get_json()
    user_name = data.get('user_name')
    query = data.get('query')
    

    docs = new_db.similarity_search(query)
    chain = get_conversational_chain()
    data = llama_is_awesome("Tell me all the ratio of {user_name} first occurrence", "tp")

    # Now no input_documents is passed
    response = chain.invoke(
        {"user_name": user_name, "query": query, "data": data, "context": docs}
    )

    return jsonify({"response": response})

'''while True:
    query = input("Query")
    user_input(query)'''