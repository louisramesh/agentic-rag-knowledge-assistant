
"""Full Agentic RAG pipeline using LangGraph, LangChain, Pinecone, Azure OpenAI."""
import os, json
from dotenv import load_dotenv
load_dotenv()

import pinecone
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.vectorstores import Pinecone
from langgraph.graph import StateGraph, END

# Initialise Pinecone
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

llm = AzureChatOpenAI(
    deployment_name=os.environ['AZURE_OPENAI_DEPLOYMENT_GPT4'],
    model_name='gpt-4',
    temperature=0,
    openai_api_base=os.environ['AZURE_OPENAI_ENDPOINT'],
    openai_api_type='azure'
)

# ------- LangGraph nodes --------
graph = StateGraph()

@graph.node
def planner(state):
    query = state['query']
    # In real impl. call an LLM to break into sub tasks
    state['sub_query'] = query
    # Example metadata filter generation
    state['metadata_filter'] = {
        "department": {"$eq": "Engineering"},
        "access_level": {"$lte": 2}
    }
    return state

@graph.node
def retrieve(state):
    retriever = vectorstore.as_retriever(search_kwargs={
        "filter": state['metadata_filter'],
        "k": 5
    })
    docs = retriever.get_relevant_documents(state['sub_query'])
    state['docs'] = docs
    return state

@graph.node
def reason(state):
    context = "\n\n".join([d.page_content for d in state['docs']])
    prompt = f"""You are an enterprise developer assistant. 
    Answer the QUESTION using only the CONTEXT.

    CONTEXT:
    {context}

    QUESTION: {state['query']}
    """
    response = llm.invoke(prompt)
    state['answer'] = response.content
    return state

@graph.node
def final(state):
    print("Answer:\n", state['answer'])
    print("\nSources:")
    for d in state['docs']:
        print(d.metadata)
    return END

graph.edge(planner, retrieve)
graph.edge(retrieve, reason)
graph.edge(reason, final)

compiled = graph.compile()

if __name__ == '__main__':
    compiled.invoke({"query": "How do I integrate our payment gateway SDK in Java?"})
