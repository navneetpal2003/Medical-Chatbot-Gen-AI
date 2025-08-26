from src.helper import load_pdf_file, text_split, download_huggingface_embeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"]=PINECONE_API_KEY

extracted_data=load_pdf_file(data='Data/')
text_chunks=text_split(extracted_data)
embeddings = download_huggingface_embeddings()

pc = Pinecone(api_key=PINECONE_API_KEY)
index_name= "medicalbot"
docsearch= PineconeVectorStore.from_documents(
    documents= text_chunks,
    index_name=index_name,
    embedding=embeddings
)
docsearch=PineconeVectorStore.from_existing_index(
    index_name= index_name,
    embedding= embeddings
)