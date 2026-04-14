
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI # <--- बदलाव यहाँ है
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

class InsuranceRAG:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.persist_directory = "vector_db"
        
        # OpenAI Models का इस्तेमाल
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1) # gpt-4o-mini बहुत सस्ता और तेज़ है
        
        self.vector_db = None

    def create_vector_store(self):
        if not os.path.exists(self.pdf_path):
            print(f"Error: {self.pdf_path} फ़ाइल नहीं मिली!")
            return
        
        loader = PyPDFLoader(self.pdf_path)
        docs = loader.load()
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_documents(docs)
        
        # पुराना vector_db डिलीट करके नया बनाना
        self.vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        print("Success: OpenAI Vector Database तैयार है!")

    def get_response(self, query):
        if not self.vector_db:
            self.vector_db = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
        
        retriever = self.vector_db.as_retriever(search_kwargs={"k": 5})
        
        template = """You are a professional Healthcare Insurance Assistant. 
        Answer the question based ONLY on the following context:
        {context}
        
        Question: {question}
        Helpful Answer:"""
        
        prompt = ChatPromptTemplate.from_template(template)

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        return rag_chain.invoke(query)

if __name__ == "__main__":
    PDF_FILE = "Data/healthcare_policy.pdf" 
    bot = InsuranceRAG(PDF_FILE)
    
    # फ्रेश शुरुआत के लिए पुराना फोल्डर डिलीट कर दें अगर एरर आए
    bot.create_vector_store()
    
    print("\nShieldCare OpenAI Bot Testing...")
    print(f"Answer: {bot.get_response('What is the Free Look Period, and how many days does a customer have to cancel their policy for a refund?')}")
