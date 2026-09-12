"""
Wires the three agents into a LangGraph state machine:

    retriever -> synthesis -> critic --approved--> END
                     ^             |
                     |--needs_rewrite
                     |
    retriever <------|--needs_retrieval

This is the "feedback loop to maximize cited content" from the project spec:
the critic can send control back to either node depending on *why* it
rejected the draft, instead of a flat linear pipeline.
"""
from functools import partial

from langgraph.graph import END, StateGraph
from qdrant_client import QdrantClient

from src.agents.critic_agent import critic_node
from src.agents.retriever_agent import retriever_node
from src.agents.state import AgentState
from src.agents.synthesis_agent import synthesis_node
from src.config import settings
from src.vectorstore import get_qdrant_client


def _route_after_critic(state: AgentState) -> str:
    verdict = state.get("critic_verdict")
    # "unverified" is the critic giving up at the retry cap. It is terminal:
    # routing it anywhere but END would restart the loop it exists to stop.
    if verdict in ("approved", "unverified"):
        return END
    if verdict == "needs_retrieval":
        return "retriever"
    return "synthesis"  # needs_rewrite


def build_graph(client: QdrantClient, keyword_weight: float = 0.3):
    graph = StateGraph(AgentState)

    graph.add_node("retriever", partial(retriever_node, client=client, keyword_weight=keyword_weight))
    graph.add_node("synthesis", synthesis_node)
    graph.add_node("critic", critic_node)

    graph.set_entry_point("retriever")
    graph.add_edge("retriever", "synthesis")
    graph.add_edge("synthesis", "critic")
    graph.add_conditional_edges(
        "critic",
        _route_after_critic,
        {END: END, "retriever": "retriever", "synthesis": "synthesis"},
    )

    return graph.compile()


def answer_question(question: str, client: QdrantClient | None = None, keyword_weight: float = 0.3) -> AgentState:
    client = client or get_qdrant_client()
    app = build_graph(client, keyword_weight=keyword_weight)
    initial_state: AgentState = {"question": question, "retry_count": 0}
    result = app.invoke(initial_state)
    return result


if __name__ == "__main__":
    import argparse
    import json as _json

    parser = argparse.ArgumentParser(description="Ask the multi-agent research assistant a question")
    parser.add_argument("question")
    args = parser.parse_args()

    result = answer_question(args.question)
    print(_json.dumps({
        "question": result["question"],
        "search_query": result.get("search_query"),
        "final_answer": result.get("final_answer"),
        "sources_cited": result.get("source_ids_cited"),
        "retries": result.get("retry_count"),
    }, indent=2))
