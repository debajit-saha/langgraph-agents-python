"""
LangGraph HITL + interrupt() demo
---------------------------------
Install:
    pip install langgraph langchain-openai python-dotenv

Set OPENAI_API_KEY in your environment (or .env file).

Run:
    python hitl_approval.py

The graph generates a proposed SQL query, interrupts for approval, and only
executes the simulated query when the human approves it.
"""

import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))


class AgentState(TypedDict):
    request: str
    sql: str
    approved: bool
    result: str
    messages: list[str]


endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")

llm = ChatOpenAI(
    base_url=endpoint,
    api_key=api_key,
    model=deployment_name
)


def generate_sql(state: AgentState):
    """Generate a SQL proposal; no database is touched here."""
    response = llm.invoke(
        "Write one read-only SQL Server SELECT query for this request. "
        "Assume a table named Employees with columns Id, Name, Department, Salary. "
        "Return only SQL, without markdown.\n\n"
        f"Request: {state['request']}"
    )
    sql = response.content.strip().replace("```sql", "").replace("```", "").strip()
    return {"sql": sql, "messages": state["messages"] + ["SQL generated."]}


def approval_node(state: AgentState):
    """Pause graph execution and wait for a human decision."""
    decision = interrupt(
        {
            "type": "sql_approval",
            "question": "Approve this SQL query for execution?",
            "sql": state["sql"],
            "options": ["approve", "reject"],
        }
    )

    approved = str(decision).strip().lower() in {
        "approve", "approved", "yes", "y"}
    return {
        "approved": approved,
        "messages": state["messages"] + [f"Human decision: {decision}"],
    }


def execute_sql(state: AgentState):
    """Demo execution only. Replace this with a real, safely parameterized DB call."""
    if not state["approved"]:
        return {"result": "SQL execution cancelled by human.", "messages": state["messages"]}

    # Safe demo: simulate database execution instead of running arbitrary SQL.
    return {
        "result": f"[DEMO] Approved query would execute:\n{state['sql']}",
        "messages": state["messages"] + ["Demo SQL execution completed."],
    }


def final_node(state: AgentState):
    response = llm.invoke(
        "Summarize this result for the user in a concise answer.\n"
        f"Original request: {state['request']}\n"
        f"SQL: {state['sql']}\n"
        f"Result: {state['result']}"
    )
    return {"messages": state["messages"] + [str(response.content)]}


def route_after_approval(state: AgentState):
    return "execute_sql" if state["approved"] else "final"


builder = StateGraph(AgentState)
builder.add_node("generate_sql", generate_sql)
builder.add_node("approval", approval_node)
builder.add_node("execute_sql", execute_sql)
builder.add_node("final", final_node)

builder.add_edge(START, "generate_sql")
builder.add_edge("generate_sql", "approval")
builder.add_conditional_edges(
    "approval",
    route_after_approval
)
builder.add_edge("execute_sql", "final")
builder.add_edge("final", END)

# MemorySaver is for local learning/demo. Use a durable DB checkpointer in production.
graph = builder.compile(checkpointer=MemorySaver())


def main():
    thread_id = "sql-approval-demo-1"
    config = {"configurable": {"thread_id": thread_id}}

    initial_state: AgentState = {
        "request": input("What SQL data do you want? "),
        "sql": "",
        "approved": False,
        "result": "",
        "messages": [],
    }

    # First run pauses at interrupt() and returns control to this program.
    paused = graph.invoke(initial_state, config=config)
    print("\n--- APPROVAL REQUIRED ---")
    print(paused["__interrupt__"][0].value)

    decision = input("\nType approve or reject: ").strip().lower()

    # Resume the same thread. The node continues after interrupt().
    completed = graph.invoke(Command(resume=decision), config=config)

    print("\n--- FINAL RESULT ---")
    print(completed["result"])
    print("\n--- AGENT RESPONSE ---")
    print(completed["messages"][-1])


if __name__ == "__main__":
    main()
