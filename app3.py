from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from sqlalchemy import create_engine, text, MetaData, Table, Column, Integer, Float, String, Boolean
import pandas as pd
import numpy as np

# Database functions remain the same
def create_in_memory_db():
    """Create an in-memory SQLite database with sample data"""
    engine = create_engine('sqlite:///:memory:')
    metadata = MetaData()
    
    # Create table
    user_statistics = Table('user_statistics', metadata,
        Column('id', Integer, primary_key=True),
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
    
    metadata.create_all(engine)
    
    # Generate sample data
    np.random.seed(42)
    sample_data = []
    names = ['Alice', 'Bob', 'Charlie', 'David', 'Eva']
    roles = ['Employee', 'Manager', 'Employee', 'Employee', 'Manager']
    
    for name, role in zip(names, roles):
        record = {
            'microwave_usage': np.random.randint(5, 15),
            'refrigerator_usage': np.random.uniform(100, 300),
            'tv_usage': np.random.uniform(100, 250),
            'laptop_usage': np.random.uniform(300, 700),
            'eating_habits': np.random.uniform(0.4, 1.0),
            'total_power_consumption': np.random.uniform(10, 20),
            'vehicle_monthly_distance_km': np.random.uniform(200, 1000),
            'frequency_of_traveling_by_air': np.random.uniform(0, 5),
            'vehicle_type': np.random.uniform(1, 3),
            'new_clothes_monthly': np.random.uniform(1, 10),
            'name': name,
            'role': role,
            'is_weekend': np.random.choice([True, False])
        }
        
        record['tv_usage_ratio'] = record['tv_usage'] / 171
        record['microwave_usage_ratio'] = record['microwave_usage'] / 8
        record['laptop_usage_ratio'] = record['laptop_usage'] / 500
        record['eating_habits_ratio'] = record['eating_habits'] / 0.7
        record['total_power_ratio'] = record['total_power_consumption'] / 15
        record['vehicle_distance_ratio'] = (record['vehicle_monthly_distance_km'] * 
                                          record['vehicle_type']) / 500
        
        sample_data.append(record)
    
    df = pd.DataFrame(sample_data)
    df.to_sql('user_statistics', engine, index=False, if_exists='append')
    
    return engine

def get_db_connection():
    return create_in_memory_db()

def execute_sql_query(query: str) -> pd.DataFrame:
    engine = get_db_connection()
    try:
        return pd.read_sql_query(query, engine)
    finally:
        engine.dispose()

def create_chains(vectorstore):
    """Create both query and response chains with proper configuration"""
    
    # Initialize the language model
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    
    # Create memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    
    # Create the prompt templates
    query_template = """
    You are an environmental analysis assistant analyzing environmental data.
    
    Given the user's input, generate appropriate SQL queries.
    
    Current Input: {input}
    History: {chat_history}
    
    Please respond in the following format:
    NEEDS_DATABASE: [Yes/No]
    SQL_QUERIES: [List of SQL queries, if needed]
    ANALYSIS_TYPE: [Usage/Comparison/Trend/General]
    """
    
    response_template = """
    You are an environmental analyst providing insights based on data.
    
    Input: {input}
    History: {chat_history}
    Analysis Results: {context}
    
    Please provide a clear analysis of the data.
    """
    
    # Create the chains
    query_chain = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": PromptTemplate(template=query_template, input_variables=["input", "chat_history"])}
    )
    
    response_chain = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": PromptTemplate(template=response_template, input_variables=["input", "chat_history", "context"])}
    )
    
    return query_chain, response_chain

def user_input(user_name: str, type_: str, question: str):
    """Process user input and generate response"""
    
    # Initialize embeddings and vector store
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    texts = [
        "Energy consumption patterns and their environmental impact.",
        "Television usage patterns and energy efficiency.",
        "Appliance usage statistics and recommendations.",
        "Environmental impact of daily activities.",
        "Energy conservation tips and best practices."
    ]
    vectorstore = FAISS.from_texts(texts, embeddings)
    
    # Create the chains
    query_chain, response_chain = create_chains(vectorstore)
    
    # Format the input
    formatted_input = f"User: {user_name}\nType: {type_}\nQuestion: {question}"
    
    try:
        # Get query analysis
        query_result = query_chain.invoke({"input": formatted_input})
        
        # Parse the analysis
        analysis_lines = query_result['answer'].split('\n')
        needs_database = False
        sql_queries = []
        analysis_type = "General"
        
        for line in analysis_lines:
            if "NEEDS_DATABASE: Yes" in line:
                needs_database = True
            elif "SQL_QUERIES:" in line:
                sql_queries = [q.strip() for q in line.replace("SQL_QUERIES:", "").strip().split(';') if q.strip()]
            elif "ANALYSIS_TYPE:" in line:
                analysis_type = line.replace("ANALYSIS_TYPE:", "").strip()
        
        # Execute SQL queries if needed
        data_results = []
        if needs_database and sql_queries:
            for query in sql_queries:
                try:
                    df = execute_sql_query(query)
                    data_results.append(df.to_dict('records'))
                except Exception as e:
                    print(f"SQL Error: {str(e)}")
        
        # Generate final response
        context = str(data_results) if data_results else "No specific data queried."
        response = response_chain.invoke({
            "input": formatted_input,
            "context": context
        })
        
        return response['answer']
    
    except Exception as e:
        return f"An error occurred while processing your request: {str(e)}"

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