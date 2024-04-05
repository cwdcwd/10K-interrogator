# Import the necessary modules and functions
import os
from dotenv import load_dotenv
import code
from unstructured.partition.pdf import partition_pdf
from pydantic import BaseModel
from typing import Any

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.retrievers import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_community.vectorstores.elasticsearch import ElasticsearchStore

from elasticsearch import Elasticsearch

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
  RunnableLambda,
  RunnablePassthrough
)
from langchain_core.documents import Document
from langchain.output_parsers import JsonOutputToolsParser

import uuid
from typing import Union
from operator import itemgetter
import pickle
from itertools import chain
from langchain_core.pydantic_v1 import SecretStr

# Load the .env file. By default, it looks for the .env file in the same directory as the script being run, or you can specify the path as an argument.
load_dotenv()

OPENAI_API_KEY=os.getenv("OPENAI_API_KEY", "")
API_KEY_UNSTRUCTURED = os.getenv("API_KEY_UNSTRUCTURED", "")
API_BASE_URL_UNSTRUCTURED = os.getenv("API_BASE_URL_UNSTRUCTURED", "")
ES_HOST = os.getenv("ES_HOST", "")
ES_PORT = int(os.getenv("ES_PORT", "9200"))
ES_INDEX = os.getenv("ES_INDEX", "")

model = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-1106", api_key=SecretStr(OPENAI_API_KEY) )
VectorStoreSingleton = None
ESSingleton = None

class Element(BaseModel):
  type: str
  text: Any

# TODO: These should really be loaded on the fly from the filesystem
# Define list containing pdf paths and pdf names to be used throughout later on
pdf_paths = [
            # "./docs/amazon/amazon-2019.pdf",
            #  "./docs/amazon/amazon-2020.pdf",
            #  "./docs/amazon/amazon-2021.pdf",
            #  "./docs/amazon/amazon-2022.pdf",
            #  "./docs/amazon/amazon-2023.pdf", 
             "./docs/amazon/amazon-2024.pdf",
            #  "./docs/alphabet/20210203-alphabet-10k",
            #  "./docs/alphabet/20220202-alphabet-10k",
             "./docs/alphabet/goog-10-k-2023.pdf",
            #  "./docs/alphabet/goog-10-k-q4-2022.pdf"
]
pdfs = [
        # "amazon-2019.pdf",
        # "amazon-2020.pdf", 
        # "amazon-2021.pdf", 
        # "amazon-2023.pdf", 
        "amazon-2024.pdf",
        # "20210203-alphabet-10k",
        # "20220202-alphabet-10k",
        "goog-10-k-2023.pdf",
        # "goog-10-k-q4-2022.pdf"
        ]

#################### ES / Vector Store Funcs ####################
def getVectorStore():
  global VectorStoreSingleton
  if VectorStoreSingleton is None:
    print("Setting up vector store")
    VectorStoreSingleton = ElasticsearchStore(
        # https://python.langchain.com/docs/integrations/vectorstores/elasticsearch
        embedding=OpenAIEmbeddings(model="text-embedding-3-small", api_key=SecretStr(OPENAI_API_KEY)),
        es_url="http://"+ES_HOST+":"+str(ES_PORT),
        index_name=ES_INDEX,
        strategy=ElasticsearchStore.ApproxRetrievalStrategy()
    )
  else:
    print("Vector store already set up")

  return VectorStoreSingleton

def get_es():
  global ESSingleton
  if ESSingleton is None:
    print("Setting up ES")
    ESSingleton = Elasticsearch([{'host': ES_HOST, 'port': ES_PORT, 'scheme': 'http'}])
  else:
    print("ES already set up")

  return ESSingleton

#################### Data Loading & Processing Funcs ####################
def processPDFsToPickles():
  raw_pdfs_elements = []

  # Get parsed elements for each PDF
  for i,pdf_path in enumerate(pdf_paths):
    print(f"processing: {pdf_path}")
    raw_pdfs_elements.append(
      partition_pdf(
        # https://unstructured-io.github.io/unstructured/apis/api_parameters.html
        filename=pdf_path,
        extract_images_in_pdf=False,
        infer_table_structure=True,
        chunking_strategy="by_title",
        max_characters=1800,
        new_after_n_chars=1500,
        combine_text_under_n_chars=1000,
        image_output_dir_path="./",
        url=API_BASE_URL_UNSTRUCTURED,
        token=API_KEY_UNSTRUCTURED,
        verbose=True
      )
    )

    # store the parsed elements as pickles to reuse them whenever necessary
    with open(f'{pdf_path}-{i}.pkl', 'wb') as f:
      print(f"saving: {pdf_path}-{i}")
      pickle.dump(raw_pdfs_elements[i], f)

  return raw_pdfs_elements

def loadDataFromPickles(pickle_paths):
  # Load from pickle
  raw_pdf_elements = []
  for pdf in pickle_paths:
    with open(f"{pdf}", 'rb') as f:
      raw_pdf_elements.append(pickle.load(f))
      
  return raw_pdf_elements

def processTablesAndText(raw_pdfs_elements):
  # Categorize by type
  print("Categorizing elements")
  categorized_elements = [
      [
          Element(type="table", text=str(element.metadata.text_as_html))
          if "unstructured.documents.elements.Table" in str(type(element))
          else Element(type="text", text=str(element))
          for element in raw_pdf_element
      ]
      for raw_pdf_element in raw_pdfs_elements
  ]

  print("Categorized table elements")
  table_elements = [ [e for e in categorized_element if e.type == "table"] for categorized_element in categorized_elements ]

  print("Categorized text elements")
  text_elements = [ [e for e in categorized_element if e.type == "text"] for categorized_element in categorized_elements ]
  
  return table_elements, text_elements

## chunks to docs
def get_docs(text_ele):
    print("Getting docs")
    pdf_docs = []
    pdf_docs.extend(
        [Document(page_content=ele.text, metadata={"pdf_title":t[1]}) for ele in t[0]]
        for i,t in enumerate(zip(text_ele,pdfs))
    )
    # Flattens the list of 3 lists
    print("Flattening list")
    pdf_docs = list(chain(*pdf_docs))
    print("Returning pdf docs")
    return pdf_docs

def runSummarizations(docs, prompt, pickle_file):
  print("Running summarizations")  
  summarize_prompt = ChatPromptTemplate.from_template(prompt)
  summarize_chain = {"element": RunnablePassthrough()} | summarize_prompt | model | StrOutputParser()

  texts = [text.page_content for text in docs]
  summaries = summarize_chain.batch(texts, {"max_concurrency": 2})
  with open(pickle_file, 'wb') as f:
    pickle.dump(summaries, f)
  
  ids = [str(uuid.uuid4()) for _ in docs]
  id_key = "doc_id"
  # Store summaries as documents and add IDs to them
  print("Storing summaries")
  summaries_docs = [
    Document(page_content=summaries[i], metadata={id_key:ids[i], "pdf_title":doc.metadata['pdf_title']})
    for i,doc in enumerate(docs)
  ]

  return summaries_docs, ids

def summarizeAndSaveToVectorStore(text_docs, table_docs):
  prompt_for_text_summarize = """You are an assistant tasked with summarizing text. \
    Give a concise summary of the text. Text chunk: {element} """
  prompt_for_table_summarize = """You are an assistant tasked with summarizing tables. \
    Give a concise summary of the table. Table chunk: {element} """
    
  ## process text
  print("Processing text")
  text_summaries_docs, text_ids = runSummarizations(text_docs, prompt_for_text_summarize, "text_summaries.pkl")
  ## process tables
  print("Processing tables")
  table_summaries_docs, table_ids = runSummarizations(table_docs, prompt_for_table_summarize, "table_summaries.pkl")

  vectorstore = getVectorStore()
  print("Adding text summary documents to vector store")
  vectorstore.add_documents(text_summaries_docs);
  print("Adding table summary documents to vector store")
  vectorstore.add_documents(table_summaries_docs);
  docs_w_ids = list(zip(text_ids+table_ids,text_docs+table_docs))
  return docs_w_ids

def loadAndProcessFiles():
# TODO: This should be done on the fly from the filesystem
  raw_pdfs_elementsFromPickles = loadDataFromPickles(["./docs/alphabet/goog-10-k-2023.pdf-1.pkl", "./docs/amazon/amazon-2024.pdf-0.pkl"])
  print("Getting text and table docs")

  table_elements,text_elements = processTablesAndText(raw_pdfs_elementsFromPickles)
  table_docs = get_docs(table_elements)
  text_docs = get_docs(text_elements)
  docs_w_ids = summarizeAndSaveToVectorStore(text_docs, table_docs)
  return docs_w_ids, text_docs, table_docs


#################### RAG Funcs ####################
def query_elasticsearch(index, field, value, fieldsToReturn):
  es = get_es()
  pageSize = 100
  fromPosition = 0
  hasResults = True
  results = []
  
  while hasResults:
    body = {
        "size": pageSize,
        "from": fromPosition,
        "_source": fieldsToReturn,
        "query": {
            "match": {
                field: value
            }
        }
    }

    response = es.search(index=index, body=body)
    fromPosition += pageSize

    if(len(response['hits']['hits'])==0):
      hasResults = False
    else:
     results+=response['hits']['hits']
    
  return results

def get_pdf_docs_from_es(pdfName):
  print("Getting pdf docs from ES for pdf: "+pdfName)

  results = query_elasticsearch(ES_INDEX, "metadata.pdf_title.keyword", pdfName, ["text", "metadata"])
  docs_found = []
  for hit in results:
    # print(hit["_source"])
    docs_found += [ (hit["_source"]["metadata"]["doc_id"], Document(page_content=hit["_source"]["text"], metadata=hit["_source"]["metadata"])) ]
  
  print("Returning docs found")
  # print(docs_found)
  return docs_found

# Function to get the original text chunks given the retrieved summary texts
def get_orig(summary_docs, docs_w_ids, id_key="doc_id"):
  out_docs = []
  for summary_doc in summary_docs:
    for docs in docs_w_ids:
      # print("Comparing: "+docs[0]+" with "+summary_doc.metadata[id_key])
      if docs[0] == summary_doc.metadata[id_key]:
        out_docs.append(docs[1])  
  
  return out_docs

#test helper function to search for similar documents
def searchForSimilarDocs(docs_w_ids=None):
  vectorstore = getVectorStore()
  print("searching for similar documents")
  searchResults = vectorstore.similarity_search("How much is Amazon investing in R&D?", k=2, filter=[{"term": {"metadata.pdf_title.keyword": "amazon-2019.pdf"}}])
  print("search results: "+str(searchResults))

# Function to get the context from the separated query and the respective PDFs
def get_context(pdf_response):
  vectorstore = getVectorStore()
  context_out = []
  for resp in pdf_response.split('\n'):
      pdfName = resp.split(',')[1].strip()
      docs_w_ids = get_pdf_docs_from_es(pdfName)
      vector_results = vectorstore.similarity_search(resp.split(',')[0], k=2, filter=[{"term": {"metadata.pdf_title.keyword": pdfName}}])
      print("Vector results found")
      # print(vector_results)
      context_out.append(
          get_orig(vector_results, docs_w_ids)
      )

  return context_out

# Format the the response to differentiate the contexts
def parse_context(contexts):
  print("Parsing context")
  # print(contexts)

  str_out = ""
  for context in contexts:
    if(len(context)==0):
      continue
  
    str_out += "CONTEXT FROM " + context[0].metadata['pdf_title'] + "\n"
    if len(context)==1:
      continue

    for c in context:
      str_out += c.page_content + "\n\n"

  return str_out

def buildContextChain():
  get_pdf_query = """You are an assistant tasked with generating additional questions from the given query. \
  Given a set of questions, give the relevant questions (in the format as shown) pertaining to each individual company \
  in the query IF there are more than one. Also give the report name it corresponds to.
  Report names:
  amazon-2019.pdf
  goog-10-k-2023.pdf

  <--example start-->
  Query: What are the equity compensation plans of Amazon and Alphabet?
  Answer:
  What are the equity compensation plans of Amazon?, amazon-2019.pdf
  What are the equity compensation plans of Alphabet?, goog-10-k-2023.pdf
  <--example end-->

  <--example start-->
  Are there any ongoing legal disputes with Amazon?
  Answer:
  Are there any ongoing legal disputes with Amazon?, amazon-2019.pdf
  <--example end-->

  Query: {user_query}
  Answer:
  """
  get_pdf_query_prompt = ChatPromptTemplate.from_template(get_pdf_query)
  get_pdf_query_chain = {"user_query": RunnablePassthrough()} | get_pdf_query_prompt | model | StrOutputParser()
    
  context_chain = get_pdf_query_chain | get_context | parse_context
  return context_chain

def buildRAGChain():
  rag_prompt_text = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question \
in as many words as required.
Feel free to go into the details of what's presented in the context down below.
If you don't know the answer, just say "I don't know."
Question: {question}
Context: {context}
Answer: 
"""
  context_chain = buildContextChain()
  rag_prompt = ChatPromptTemplate.from_template(rag_prompt_text)

  rag_chain = (
    {"question": RunnablePassthrough(), "context": context_chain}
    | rag_prompt
    | model
    | StrOutputParser()
  )
  
  return rag_chain

def interrogateLLM(query):
  ## interrogate the LLM
  context_chain = buildContextChain()
  return context_chain.invoke(query)

def interrogateLLMWithRAG(query):
  rag_chain = buildRAGChain()
  return rag_chain.invoke(query)


# print("Running queries")
# # searchForSimilarDocs()
# # query_result = interrogateLLM("How much is Amazon and Alphabet investing in R&D?")
# # print(query_result)

# searchResults = interrogateLLMWithRAG({"question":"What is Amazon's approach to sustainability and environmental impact?"})
# print(searchResults)

# searchResults = interrogateLLMWithRAG({"question":"Who are Alphabet's main competitors?"})
# print(searchResults)

functions = {
  'interrogateLLM': interrogateLLM,
  'interrogateLLMWithRAG': interrogateLLMWithRAG,
  'loadAndProcessFiles': loadAndProcessFiles,
}

repl = code.InteractiveConsole(locals=functions)
repl.interact(banner="How can I help?", exitmsg="Goodbye!")
# repl.runsource('function1()', '<input>', 'exec', functions)