from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import tool
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
import os

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")

model = ChatOpenAI(
    base_url=endpoint,
    api_key=api_key,
    model=deployment_name
)

loaders = [
    PyPDFLoader("cloudxeus-acceptable-use-policy.pdf"),
    PyPDFLoader("cloudxeus-refund-policy.pdf"),
    PyPDFLoader("cloudxeus-service-level-agreement.pdf"),
]

documents = []
for loader in loaders:
    documents.extend(loader.load())

print(f"Loaded {len(documents)} pages across all policy documents.")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)

chunks = splitter.split_documents(documents)
print(f"Split into {len(chunks)} chunks.")

print("Building vector store...")

embeddings = OpenAIEmbeddings(
    base_url=endpoint,
    api_key=api_key,
    model="text-embedding-3-small",
)

vector_store = FAISS.from_documents(chunks, embeddings)
print("Vector store ready.")


@tool
def search_cloudxeus_policies(query: str) -> str:
    """
    Search the CloudXeus policy documents — Acceptable Use Policy,
    Refund Policy, and Service Level Agreement — to answer questions
    about CloudXeus rules, eligibility, credits, and procedures.
    """
    results = vector_store.similarity_search(query, k=3)
    if not results:
        return "No relevant policy information found."

    output = []
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get("source", "Unknown policy")
        output.append(f"[Excerpt {i} from {source}]\n{doc.page_content}")

    return "\n\n".join(output)


tools = [search_cloudxeus_policies]
model_with_tools = model.bind_tools(tools)


def call_model(state: MessagesState):
    system_message = {
        "role": "system",
        "content": (
            "You are the CloudXeus policy assistant. "
            "Answer questions accurately using the search_cloudxeus_policies tool. "
            "Always base your answers on the retrieved policy content. "
            "If you cannot find the answer in the policies, say so clearly."
        )
    }
    messages = [system_message] + state["messages"]
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}


def should_continue(state: MessagesState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END


graph_builder = StateGraph(MessagesState)
graph_builder.add_node("call_model", call_model)
graph_builder.add_node("tools", ToolNode(tools))
graph_builder.add_edge(START, "call_model")
graph_builder.add_conditional_edges("call_model", should_continue)
graph_builder.add_edge("tools", "call_model")

graph = graph_builder.compile()
print("Agent ready.\n")

questions = [
    "What is the refund window for a CloudXeus Pro subscription?",
    "What uptime does CloudXeus guarantee for Enterprise customers?",
    "Can I mine cryptocurrency on CloudXeus compute resources?",
    "What happens if CloudXeus suspends my account — do I get a refund?",
]

for question in questions:
    print(f"Question: {question}")
    result = graph.invoke({
        "messages": [{"role": "user", "content": question}]
    })
    print(f"Answer: {result['messages'][-1].content}")
    print("-" * 60)
