
"""Baseline RAG chain without agent orchestration."""
import os
from dotenv import load_dotenv
load_dotenv()
import pinecone
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA

pinecone.init(api_key=os.environ['PINECONE_API_KEY'],
              environment=os.environ['PINECONE_ENVIRONMENT'])
embedding = AzureOpenAIEmbeddings(
    deployment=os.environ['AZURE_OPENAI_DEPLOYMENT_EMBEDDING'],
    model='text-embedding-3-large',
    openai_api_base=os.environ['AZURE_OPENAI_ENDPOINT'],
    openai_api_type='azure'
)
vectorstore = Pinecone(
    index_name='dev-assistant',
    embedding=embedding,
    namespace='knowledge_assistant'
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
llm = AzureChatOpenAI(
    deployment_name=os.environ['AZURE_OPENAI_DEPLOYMENT_GPT4'],
    model_name='gpt-4',
    temperature=0,
    openai_api_base=os.environ['AZURE_OPENAI_ENDPOINT'],
    openai_api_type='azure'
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=True
)

if __name__ == '__main__':
    query = "Explain our micro‑service CI/CD pipeline."
    result = qa(query)
    print(result['result'])
    for d in result['source_documents']:
        print(d.metadata)
