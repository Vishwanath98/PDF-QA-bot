#import os
#from dotenv import load_dotenv
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings #_community.
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from  langchain.chains import RetrievalQA

#load_dotenv()

from langchain.llms import GooglePalm
#api_key='AIzaSyBJsNVLoUWq_svCzc_Ga4ylv3kzX96SNFA'

llm= GooglePalm(google_api_key='AIzaSyBJsNVLoUWq_svCzc_Ga4ylv3kzX96SNFA', temperature=0.1)


instructor_embeddings = HuggingFaceInstructEmbeddings()
vdb_file_path='faiss-index'
def create_vector_db():
    loader = CSVLoader(file_path='codebasics_faqs.csv', source_column='prompt')
    data = loader.load()
    vectordb= FAISS.from_documents(documents = data, embedding = instructor_embeddings)
    vectordb.save_local(vdb_file_path)


def get_qa_chain():
    vectordb = FAISS.load_local(vdb_file_path, instructor_embeddings)

    retriever = vectordb.as_retriever(score_threshold=0.7)


    prompt_template = """Given the context and a question, generate an answer based on this context.
        Try to provide as much text as possible from the 'response' section in the source document.
        If the answer is not found in the context, kindly state 'i dONT Know'. Don't try to make up an answer
        CONTEXT : {context}
        QUESTION : {question}"""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])

    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type='stuff',  # mapreduce
                                        retriever=retriever,
                                        input_key='query',
                                        return_source_documents=True,
                                        chain_type_kwargs={'prompt': PROMPT})

    return chain
if __name__=='__main__':
    chain= get_qa_chain()

