import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
import sqlite3
from datetime import datetime
from dotenv import load_dotenv

import streamlit as st

st.set_page_config(page_title="RAG APP",layout="wide")
st.title("RAG-LTM-Chatbot")

# Step 1 : Sqlite3 DB - LTM Chat History
# -----------------------------------------
conn = sqlite3.connect('long_term_memory.db')
cursor = conn.cursor()

cursor.execute("""
create table if not exists chat_history(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_question TEXT,
  ai_answer TEXT,
  created_on TEXT
)""")
conn.commit()

@st.cache_resource
def load_vectorstores():
    loader = TextLoader("my_docs.txt")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=500,chunk_overlap=50)
    doc_chunks = text_splitter.split_documents(documents)
    embedding_model = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    document_vectorstore = FAISS.from_documents(doc_chunks,embedding_model)
    memory_docs = [Document(page_content="initial memory placeholder")]
    memory_vectorstore = FAISS.from_documents(memory_docs,embedding_model)
    return document_vectorstore,memory_vectorstore,embedding_model


document_vectorstore,memory_vectorstore,embedding_model = load_vectorstores()
retriever_obj = document_vectorstore.as_retriever(search_kwargs={"K": 2})
llm_obj = ChatGroq(model="llama-3.1-8b-instant",api_key = os.getenv('GROQ_API_KEY'))
#llm_obj = ChatGroq(model="llama-3.1-8b-instant",api_key = "")

my_prompt = PromptTemplate(
    input_variables = ["context","question","memory"],
    template = """
    You are a strict QA Assistant
    use ONLY the document context and previous memory.

    Previous memory:
    {memory}
    
    Document context:
    {context}

    Question:
    {question}

    If answer is not present in context and memory , reply exactly
    "I don't Know, the document doesnot contain this information"
    Answer:
    """
)

def save_chat_memory(question,answer):
  cursor.execute("""insert into chat_history(user_question,ai_answer,created_on)
  values(?,?,?)""",(question,answer,datetime.now().isoformat()))
  conn.commit()
  # Save to FAISS memory
  memory_vectorstore.add_texts([f"Question:{question}\nAnswer:{answer}"])

def retrieve_memory(query,k=2):
  memory_results = memory_vectorstore.similarity_search(query,k=k)
  return "\n".join([doc.page_content for doc in memory_results])

def ask_rag(question):
  previous_memory = retrieve_memory(question)
  retrieved_docs = retriever_obj.invoke(question)
  context = "\n".join([doc.page_content for doc in retrieved_docs])
  final_prompt = my_prompt.format(memory = previous_memory,context = context, question=question)
  response = llm_obj.invoke(final_prompt)
  answer = response.content
  save_chat_memory(question,answer)
  return answer

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

#st.write(st.session_state.chat_history)

user_input = st.text_input("Ask Something:")

if st.button("Submit") and user_input:
    answer = ask_rag(user_input)
    st.session_state.chat_history.append(("You:",user_input))
    st.session_state.chat_history.append(("AI:",answer))

#st.success(st.session_state.chat_history)

for var in st.session_state.chat_history:
    st.write(f'{var[0]}')
    st.success(f'{var[1]}')
    
    #if role == "You":
    #   st.markdown(f"***You:***{msg}")
    #else:
    #    st.markdown(f"***AI:***{msg}")
                    

  

