from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.groq import Groq
from llama_index.core import SQLDatabase
from langchain.memory import ConversationBufferMemory
import os

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
        "https://www.constellation.com/energy-101/energy-innovation/how-to-reduce-your-carbon-footprint.html",
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

# Step 1: Create an in-memory SQLite database


llm = Groq(model="llama3-70b-8192", api_key=GROQ_API_KEY)
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")


from llama_index.core.query_engine import NLSQLTableQueryEngine

query_engine = NLSQLTableQueryEngine(
    sql_database=sql_database, tables=["user_statistics"], llm=llm, embed_model=embed_model
)

#def llama_index_isgreat():
def llama_is_awesome(query_str):
    response = query_engine.query(query_str)
    print(response)
    return response

def get_query_analysis_chain():
    query_prompt_template = """
    You are an environmental analysis assistant. Your role is to determine if a user's query requires database access and generate appropriate database queries if needed.

    User Query Analysis Guidelines:
    1. For "General" type:
        - Determine if database access is needed
        - If needed, generate specific questions for the database in English
    2. For "Data", "Positive Data", or "Negative Data" types:
        - Always generate specific database queries to gather relevant information

    Current Input:
    User Name: {user_name}
    Type: {type}
    Question: {question}
    Context: {context}

    Please respond in the following format:
    NEEDS_DATABASE: [Yes/No]
    QUERIES: [List of specific questions for the database, if needed]
    ANALYSIS_TYPE: [Usage/Comparison/Trend/General]
    
    Remember: Generate specific, focused questions that will help gather relevant data about the user's environmental impact and usage patterns.

    Response:
    """
    
    query_prompt = PromptTemplate(
        template=query_prompt_template,
        input_variables=["context", "user_name", "type", "question"]
    )
    
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    return ConversationalRetrievalChain.from_llm(
        llm=model,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": query_prompt}
    )

def get_response_chain():
    response_prompt_template = """
    You are an advanced environmentalist chatbot with access to user environmental impact data. Use the provided database information to generate personalized, actionable insights.

    Context: {context}
    User Name: {user_name}
    Type: {type}
    Question: {question}
    Analysis Type: {analysis_type}

    Response Guidelines:
    1. For "General" type:
        - Provide clear, direct answers using database insights if available
        - Focus on practical environmental impact information

    2. For "Data" type:
        - Start with "Based on the data analysis:"
        - Present specific metrics and patterns from the database
        - Explain environmental impact implications

    3. For "Positive Data" type:
        - Start with "Great job, {user_name}!"
        - Highlight specific positive patterns from the data
        - Suggest ways to maintain or improve further

    4. For "Negative Data" type:
        - Start with "Thank you for being open to improvement, {user_name}."
        - Tactfully present areas needing attention
        - Provide specific, actionable recommendations based on the data

    Always:
    - Reference specific data points where available
    - Provide practical, actionable advice
    - Maintain an encouraging, supportive tone
    - Connect recommendations to environmental impact

    Response:
    """
    
    response_prompt = PromptTemplate(
        template=response_prompt_template,
        input_variables=["context", "user_name", "type", "question", "analysis_type"]
    )
    
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    return ConversationalRetrievalChain.from_llm(
        llm=model,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": response_prompt}
    )

def user_input(user_name, type_, question):
    # Initialize embedding and vector store
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.load_local("faiss_index_webpages", embeddings, allow_dangerous_deserialization=True)
    
    # First chain - Analyze query and generate database queries
    query_chain = get_query_analysis_chain()
    query_analysis = query_chain.invoke({
        "question": question,
        "user_name": user_name,
        "type": type_,
        "chat_history": []
    })
    
    # Parse the query analysis response
    analysis_lines = query_analysis['answer'].split('\n')
    needs_database = False
    database_queries = []
    analysis_type = "General"
    
    for line in analysis_lines:
        if "NEEDS_DATABASE: Yes" in line:
            needs_database = True
        elif "QUERIES:" in line:
            database_queries = line.replace("QUERIES:", "").strip().split(';')
        elif "ANALYSIS_TYPE:" in line:
            analysis_type = line.replace("ANALYSIS_TYPE:", "").strip()
    
    # If database access is needed, perform similarity search
    docs = []
    if needs_database and database_queries:
        for query in database_queries:
            docs.extend(vectorstore.similarity_search(query))
    
    # Second chain - Generate final response
    response_chain = get_response_chain()
    response = response_chain.invoke({
        "question": question,
        "user_name": user_name,
        "type": type_,
        "analysis_type": analysis_type,
        "chat_history": []
    })
    
    return response['answer']

def main():
    while True:
        user_name = input("Enter user name (or type 'exit' to stop): ")
        if user_name.lower() == 'exit':
            break
        
        type_ = input("Enter type (General, Data, Positive Data, Negative Data): ")
        question = input("Enter your question: ")

        try:
            response_text = user_input(user_name, type_, question)
            print("\nBot Response:")
            print(response_text)
            print("-" * 50)
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            print("-" * 50)

if __name__ == "__main__":
    main()