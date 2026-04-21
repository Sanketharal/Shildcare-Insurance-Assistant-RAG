import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

class InsuranceRAG:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.persist_directory = "vector_db"
        
        # load embeddings model
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # load LLM model
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"), temperature=0.1)
        
        self.vector_db = None

    def create_vector_store(self):
        if not os.path.exists(self.pdf_path):
            print(f"Error: {self.pdf_path} File Not Found")
            return
        
        print("PDF loading and vector creation in progress...")
        loader = PyPDFLoader(self.pdf_path)
        docs = loader.load()
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
        chunks = splitter.split_documents(docs)
        
        # Clear existing vector database if it exists
        if os.path.exists(self.persist_directory):
            import shutil
            shutil.rmtree(self.persist_directory)

        self.vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        print("Success: Vector Database is ready!")

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

    PDF_FILE = "d:/RAG-Project/data/healthcare_policy.pdf" 
    
    bot = InsuranceRAG(PDF_FILE)
    
    # bot.create_vector_store()
    
    print("\nShieldCare Bot Testing...")
    
    # test question
    question = "Explain me Family Floater Plan in details?"
    print(f"Question: {question}")
    print(f"Answer: {bot.get_response(question)}")