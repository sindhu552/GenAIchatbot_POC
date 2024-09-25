#pip install flask-sqlalchemy

#pip install pypdf

#pip install faiss-cpu

import os
#from langchain.document_loaders import PyPDFLoader
from langchain_community.chat_models import ChatOpenAI
#from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.indexes import VectorstoreIndexCreator
#from langchain.llms.bedrock import Bedrock
# from langchain_google_genai import ChatGoogleGenerativeAI
# import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access the environment variables
api_key = os.getenv('api_key')


#api_key = 'AIzaSyB4V0YKmIoQk1BMlRkRATL011Uy9nUcFaA'
# genai.configure(api_key=api_key)

def company_data():
    #data_load = PyPDFLoader(file_path = 'Incedo_Benefits Program.pdf')
    data_load1 = JSONLoader(
    file_path="p360_payload.json",
    jq_schema=".",
    text_content=False,
    )
    data_load2 = JSONLoader(
    file_path="patient_journey_payload.json",
    jq_schema=".",
    text_content=False,
    )
    documents1 = data_load1.load()
    documents2 = data_load2.load()
    text = ""
    #for val in documents1:
    #    text += val.page_content
    for val in documents2:
        text += val.page_content
    pdf_split = RecursiveCharacterTextSplitter(chunk_size = 6000, chunk_overlap=500)
    text_chunks = pdf_split.split_text(text)
    
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def company_llm():

    prompt_template = """
    Thanks for asking! Answer the question as thoroughly as possible using the provided context.
    
    If the information is not available in the provided context, simply reply with: "Please ask question as per the context or check in the JSON document." Do not generate incorrect or wrong answers.

    Context:
    {context}
    
    please refer to the below structure of data along with the context to answer the questions - 
    
    context contain detailed information about patients, service requests, tasks, encounters, and other related healthcare data. Below is a summary of the structure:

    patient_id: used for uniquely identify patients
    name: defines name of patient

    patientType: define type of patient
    gatcf_active_flag: Value is either true or false
    
    Service Requests:
    sr_id: used for uniquely identify Service Requests
    other details related to Service request and their data types
    source: string
    start_date: string
    closed_date: string
    age_of_sr: integer
    sr_type:
    text: string
    case_number: string
    product: list of strings
    status: string
    foundation_BI_flag: boolean
    Detail_URL: string
    
    Tasks:
    id: used for uniquely identify Tasks
    other details related to Task and their data types
    source: string
    sr_id: string
    activity_type: string
    activity_date: string
    executionPeriod:
    start: string
    end: string
    sr_type: string
    Detail_URL: string
    
    Encounters:
    encounter_id: used for uniquely identify Encounter
    other details related to Encounters and their data types
    source: string
    start_date: string
    closed_date: string
    days_to_close: integer
    sr_type: string
    case_number: string
    product: list of strings
    record_type:
    text: string
    Detail_URL: string
    
    
    Communications: 
    
    Appointments: empty list
    
    Conditions: empty list
    
    Questionnaire Responses:
    qr_id: used for uniquely identify Question
    other details related to Questionnaire Responses and their data types
    source: string
    response_date: string
    sent_date: string
    Detail_URL: string
    
    Medication Requests:
    mr_id: used for uniquely identify Medication Requests
    other details related to Medication Requests and their data types
    source: string
    milestone_type: string
    treatment_cycle: string
    closed_date_time: string
    created_date: string
    Detail_URL: string
    
    Copay Enrollments: empty list
    
    start_date: string
    
    end_date: string
    

    Use this structure to answer the question. Make sure to provide all relevant details.

    Question:
    {question}

    Answer:
"""


    model = ChatOpenAI(openai_api_key=api_key, model_name="gpt-3.5-turbo", temperature=0)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain
    

def company_rag_response(question):
    #company_rag_query = index.query(question = question, llm = rag_llm)
    company_data()
    chain = company_llm()
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(question)

    
    response = chain(
        {"input_documents":docs, "question": question}
        , return_only_outputs=True)

    print(response)
    return response["output_text"]
    
