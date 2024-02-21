from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
import os
import torch
from llama_index.legacy.llms.huggingface import HuggingFaceLLM
from langchain.chains.question_answering import load_qa_chain
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, DirectoryLoader

from langchain.llms import CTransformers
import urllib3

urllib3.disable_warnings()

directory = 'pdfdata/'


def load_docs(directory):
  documents_loader = DirectoryLoader('data', glob="./*.pdf", loader_cls=PyPDFLoader)
  #loader = DirectoryLoader(directory)
  documents = documents_loader.load()
  return documents

def split_docs(documents,chunk_size=250,chunk_overlap=20):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  docs = text_splitter.split_documents(documents)
  return docs

def get_similiar_docs(query, k=2, score=False):
  if score:
    similar_docs = db.similarity_search_with_score(query, k=k)
  else:
    similar_docs = db.similarity_search(query, k=k)
  return similar_docs

def get_answer(query):
  similar_docs = get_similiar_docs(query)
  answer = chain.run(input_documents=similar_docs, question=query)
  return answer

documents = load_docs(directory)
len(documents)
#txt_content = documents[0].page_content
#print(txt_content)

docs = split_docs(documents)
#print(len(docs))

embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Create and persist a Chroma vector database from the chunked documents
db = Chroma.from_documents(docs, embedding_function)

model_path="model/llama-2-7b-chat.ggmlv3.q8_0.bin"
model_type="llama"
max_new_tokens=512
temperature=0.7

# Additional error handling could be added here for corrupt files, etc.

llm = CTransformers(
    model=model_path,
    model_type=model_type,
    max_new_tokens=max_new_tokens,  # type: ignore
    temperature=temperature,  # type: ignore
)

chain = load_qa_chain(llm, chain_type="stuff",verbose=False)

#query = "Who is Virat Kohli?"
#query = "When was Messi born?"
query = "When was cricket recogonized?"
answer = get_answer(query)
print(answer)