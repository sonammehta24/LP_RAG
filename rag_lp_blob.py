from fastapi import FastAPI
from pydantic import BaseModel
from azure.storage.blob import BlobServiceClient
import docx, fitz, numpy as np, pickle, logging
from sklearn.metrics.pairwise import cosine_similarity
from openai import AzureOpenAI
import os
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)

#Azure Configuration
BLOB_CONNECTION_STRING = os.getenv("AZURE_BLOB_CONN_STR")
BLOB_CONTAINER = os.getenv("AZURE_BLOB_CONTAINER")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
CHAT_MODEL = "gpt-35-turbo"


#Raising ValueError if any connection not found
if not BLOB_CONNECTION_STRING:
    raise ValueError("Azure Blob Storage connection string not found. Please set the AZURE_BLOB_CONN_STR environment variable.")
if not BLOB_CONTAINER:
    raise ValueError("Azure Blob Storage container name not found. Please set the AZURE_BLOB_CONTAINER environment variable.")
if not AZURE_OPENAI_KEY:
    raise ValueError("Azure OpenAI key not found. Please set the AZURE_OPENAI_KEY environment variable.")
if not AZURE_OPENAI_ENDPOINT:
    raise ValueError("Azure OpenAI endpoint not found. Please set the AZURE_OPENAI_ENDPOINT environment variable.")
if not AZURE_OPENAI_API_VERSION:
    raise ValueError("Azure OpenAI API version not found. Please set the AZURE_OPENAI_API_VERSION environment variable.")


# Initializing Clients(BLOB and OpenAI)
try:
    blob_service = BlobServiceClient.from_connection_string(BLOB_CONNECTION_STRING, connection_verify=False)
    container_client = blob_service.get_container_client(BLOB_CONTAINER)
except Exception as e:
    raise ConnectionError(f"Error connecting to Azure Blob Storage: {e}")

try:
    client = AzureOpenAI(
        api_key=AZURE_OPENAI_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT
    )
except Exception as e:
    raise ConnectionError(f"Error connecting to Azure OpenAI: {e}")



app = FastAPI()

#Tells the expected input format for the /query endpoint.
class QueryRequest(BaseModel):
    query: str


#Local file loader(to reduce Azure cost while testing)
def list_local_files(folder='local_docs'):
    local_files = []
    for filename in os.listdir(folder):
        if filename.endswith('.pdf') or filename.endswith('.docx'):
            local_files.append(os.path.join(folder, filename))
    return local_files


#Azure Blob list and Download, skips file that already exist locally
# def list_and_download_blobs():
#     blob_service_client = BlobServiceClient.from_connection_string(BLOB_CONNECTION_STRING)
#     container_client = blob_service_client.get_container_client(BLOB_CONTAINER)
#     blobs = container_client.list_blobs()
#     local_files = []

#     for blob in blobs:
#         blob_name = blob.name
#         local_path = os.path.join("temp_files", blob_name)
#         os.makedirs(os.path.dirname(local_path), exist_ok=True)
#         blob_client = container_client.get_blob_client(blob_name)

#         #To avoid re-downloading
#         if not os.path.exists(local_path):  
#             with open(local_path, "wb") as f:
#                 f.write(blob_client.download_blob().readall())
#             print(f"Downloaded: {blob_name}")
#         else:
#             print(f"Skipped (already exists): {blob_name}")

#         local_files.append(local_path)
#     return local_files


#Text Extraction
def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

def extract_text_from_pdf(file_path):
    pdf = fitz.open(file_path)
    return "\n".join([page.get_text() for page in pdf])

def chunk_text(text, chunk_size=300):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def get_embedding(text):
    response = client.embeddings.create(
        input=[text],
        model="text-embedding-3-small"
    )
    return response.data[0].embedding


#Embedding Cache(Saves and loads embeddings to avoid recomputation, speeds up development and testing)
def save_embeddings(chunks, embeddings, path="embedding_cache.pkl"):
    with open(path, "wb") as f:
        pickle.dump((chunks, embeddings), f)

def load_embeddings(path="embedding_cache.pkl"):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None, None


#Load and Process Files(loads cached embeddings if available, Otherwise extracts text, chunks it, generates the embeddings and caches them)
def prepare_documents():
    chunks, embeddings = load_embeddings()
    if chunks and embeddings is not None:
        logging.info("Loaded cached embeddings..")
        return chunks, embeddings
    
    chunks = []
    local_files = list_local_files()  #Reading from this for local file
    #local_files = list_and_download_blobs() #Switch when reading from blob

    for file_path in local_files:
        if file_path.endswith(".docx"):
            text = extract_text_from_docx(file_path)
        elif file_path.endswith(".pdf"):
            text = extract_text_from_pdf(file_path)
        else:
            continue

        for chunk in chunk_text(text):
            chunks.append({"text": chunk, "source": file_path})

    embeddings = np.array([get_embedding(chunk["text"]) for chunk in chunks])
    save_embeddings(chunks, embeddings)
    logging.info("Embeddings computed and cached..")
    return chunks, embeddings

chunks, embeddings = prepare_documents()

#FastAPI Endpoint
#Accepts query from user, generates its embedding, computes cosine similarity with document chunks, select top 3 relevant chunks,
#Sends them as context to Azure OpenAI chat model, returns the answer and sources

# @app.post("/query")
# async def query_rag(request: QueryRequest):
#     query_embedding = np.array(get_embedding(request.query)).reshape(1, -1)
#     similarities = cosine_similarity(query_embedding, embeddings)[0]
#     #top_indices = np.argsort(similarities)[-3:][::-1]
#     #top_chunks = [chunks[i] for i in top_indices]

#     #Setting a similarity threshold(i.e., 0.75)
#     threshold = 0.75
#     #Get top matches above threshold
#     top_indices = [i for i, score in enumerate(similarities) if score >= threshold]
#     #Sort by similarity descending
#     top_indices = sorted(top_indices, key=lambda i: similarities[i], reverse=True)[:3]
#     top_chunks = [chunks[i] for i in top_indices]


#     context = "\n\n\n".join([chunk["text"] for chunk in top_chunks])
#     response = client.chat.completions.create(
#         model=CHAT_MODEL,
#         messages=[
#             {"role": "system", "content": "You are a helpful AA (Automobile Association) Assistant. Only answer questions using the provided context. If the context is insufficient, say so. Always cite the source filenames used."},
#             {"role": "user", "content": f"Answer the question based on the following context:\n\n{context}\n\nQuestion: {request.query}"}
#         ]
#     )

#     #Removing duplicates and cleaning up sources
#     unique_sources = list({os.path.basename(chunk["source"]) for chunk in top_chunks})
#     return {
#         "answer": response.choices[0].message.content,
#         #"sources": [chunk["source"] for chunk in top_chunks]
#         "sources": unique_sources
#     }

@app.post("/query")
async def query_rag(request: QueryRequest):
    query_embedding = np.array(get_embedding(request.query)).reshape(1, -1)
    similarities = cosine_similarity(query_embedding, embeddings)[0]

    #Getting the index of the most similar chunk
    best_index = int(np.argmax(similarities))
    best_chunk = chunks[best_index]

    context = best_chunk["text"]
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful AA (Automobile Association) Assistant. Only answer questions using the provided context. If the context is insufficient, say so."},
            {"role": "user", "content": f"Answer the question based on the following context:\n\n{context}\n\nQuestion: {request.query}"}
        ]
    )

    #Returns just the filename
    source_file = os.path.basename(best_chunk["source"])

    return {
        "answer": response.choices[0].message.content,
        "source": source_file
    }

#RAGAS
# Place this at the top of your script
# import os
# os.environ["GIT_PYTHON_REFRESH"] = "quiet"

# Patch RAGAS to use AzureChatOpenAI
# import ragas.llms.base as ragas_llms_base
# from langchain_openai import AzureChatOpenAI

# azure_llm = AzureChatOpenAI(
#     deployment_name=CHAT_MODEL,  #actual deployment name
#     model=CHAT_MODEL,
#     api_key=AZURE_OPENAI_KEY,
#     azure_endpoint=AZURE_OPENAI_ENDPOINT,
#     api_version=AZURE_OPENAI_API_VERSION
# )

# ragas_llms_base.llm_factory = lambda: azure_llm

# RAGAS evaluation
# from ragas.metrics import faithfulness, answer_relevancy, context_precision
# from datasets import Dataset
# from ragas import evaluate

# evaluation_data = [
#     {
#         "query": "What is breakdown add-on?",
#         "user_input": "What is breakdown add-on?",
#         "answer": "Breakdown add-on provides roadside assistance and recovery services in case your vehicle breaks down.",
#         "contexts": [
#             "Breakdown cover is an optional add-on that provides roadside assistance, vehicle recovery, and onward travel if your car breaks down."
#         ],
#         "ground_truth": "Breakdown add-on provides roadside assistance, recovery, and onward travel if your car breaks down."
#     },
#     {
#         "query": "Is car hire available in UK?",
#         "user_input": "Is car hire available in UK?",
#         "answer": "Yes, car hire is available in the UK.",
#         "contexts": [
#             "Car hire is available across the UK through various providers. You can book online or via phone, and choose from a range of vehicles."
#         ],
#         "ground_truth": "Yes, car hire is available across the UK through various providers."
#     }
# ]

# dataset = Dataset.from_list(evaluation_data)

# results = evaluate(
#     dataset,
#     metrics=[faithfulness, answer_relevancy, context_precision]
# )

# results.to_pandas().to_csv("ragas_evaluation_results.csv", index=False)
