from dotenv import load_dotenv
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
import os

load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_KEY")

""" 
    embeddings = OpenAIEmbeddings(api_key=OPENAI_KEY)
    text = "Algorithma is a data science school based in Indonesia and Supertype is data science consultancy"
    doc_embeddings = embeddings.embed_documents([text]) 
"""

embeddings = OpenAIEmbeddings(api_key=OPENAI_KEY)

loader = DirectoryLoader('news', glob="**/*.txt")
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=2500, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# Chroma vectorstore
vecstore = Chroma.from_documents(texts, embeddings)
retriever = vecstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")

llm = ChatOpenAI(api_key=OPENAI_KEY, model="gpt-3.5-turbo")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

res = rag_chain.invoke("What are the effects of legislations surrounding emissions on the Australian coal market?")
print(res)

""" qa = retrieval_qa.from_chain_type(
    llm=OpenAI(),
    chain_type="stuff",
    retriever=vecstore.as_retriever()
)

def query(q):
    print("Query: ", q)
    print("Answer: ", qa.run(q))

query("What are the effects of legislations surrounding emissions on the Australian coal market?")
query("What are China plans with renewable energy?") """

