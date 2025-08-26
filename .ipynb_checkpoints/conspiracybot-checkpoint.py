import streamlit as st
import os
import numpy as np
from openai import OpenAI

# Load API key
with open("/Users/matthewcavanaugh/Desktop/Various Data and Tech Related/Sensitive/Open API Key.txt", "r") as f:
    api_key = f.read().strip()
os.environ["OPENAI_API_KEY"] = api_key
client = OpenAI(api_key=api_key)

# Load list of "fake facts" rather than hard-coding them
def load_conspiracy_facts(file_path="/Users/matthewcavanaugh/Desktop/GitHub/conspiracy-bot-63000/conspiracy_facts.txt"):
    with open(file_path, "r", encoding="utf-8") as f:
        facts = [line.strip() for line in f if line.strip()]
    return facts

# Load at startup
conspiracy_facts = load_conspiracy_facts()


# --- RAG SYSTEM ---
class RAGSystem:
    def __init__(self, documents, use_citations=True):
        self.documents = documents
        self.use_citations = use_citations

    def embed(self, text):
        """Convert text to vector using OpenAI embeddings"""
        resp = client.embeddings.create(model="text-embedding-3-small", input=text)
        return np.array(resp.data[0].embedding)

    def retrieve(self, query, k=3):
        """Return top-k relevant documents"""
        query_vec = self.embed(query)
        doc_vecs = [self.embed(d) for d in self.documents]
        sims = [np.dot(query_vec, dv) / (np.linalg.norm(query_vec) * np.linalg.norm(dv)) for dv in doc_vecs]
        ranked = sorted(zip(self.documents, sims), key=lambda x: x[1], reverse=True)
        return [{"document": doc, "similarity": sim} for doc, sim in ranked[:k]]

    def create_prompt(self, query, retrieved_docs):
        context = ""
        for i, doc in enumerate(retrieved_docs, 1):
            context += f"Source {i} (relevance: {doc['similarity']:.3f}):\n{doc['document']}\n\n"

        citation_instr = " Cite sources using [Source i] where appropriate." if self.use_citations else ""

        return f"""You are ConspiracyBot63000, a chatbot tasked with providing conspiratorial-minded answers to users' inquiries.
        Within the fictional universe of this project, you were developed by TrueLiberty Information Network & Neuroscience Technologies, 
        a media and pseudoscience organization that claims to know the full truth about the world.
        However, you are a satirical bot, and you are not intended to cause any actual distrust in government, science, or the media.
        Use the provided documents as a basis. 
        Keep the tone conversational, like you're talking to a close friend, but don't be afraid to ramble. 
        Think Dale Gribble from King of the Hill.
        The more deranged you sound, the better. 
        If a user pushes back, double down.
        Remember, in all of this, do NOT make harmful assertions about real-world groups. {citation_instr}

        Sources:
        {context}

        Question: {query}

        Answer:"""

    def answer(self, query):
        docs = self.retrieve(query)
        prompt = self.create_prompt(query, docs)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": prompt}]
        )
        return resp.choices[0].message.content


# --- STREAMLIT UI ---
st.set_page_config(page_title="ConspiracyBot63000", page_icon="🛸", layout="centered")

st.title("🛸 ConspiracyBot63000")
st.write("A **satirical** chatbot from the *TrueLiberty Information Network & Neuroscience Technologies*.")


if "rag" not in st.session_state:
    st.session_state.rag = RAGSystem(conspiracy_facts)

query = st.text_input("Ask ConspiracyBot a question:", "")

if query:
    with st.spinner("Consulting the secret archives..."):
        answer = st.session_state.rag.answer(query)
    st.subheader("ConspiracyBot says:")
    st.write(answer)
