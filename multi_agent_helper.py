"""Multi Agent Helper"""
from langgraph.graph import END,StateGraph
from typing import TypedDict,Annotated,Sequence
from langgraph_util import show_graph
from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage,HumanMessage,AIMessage
from langgraph.checkpoint.memory import MemorySaver
import operator
import uuid
import streamlit as st

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

llm = ChatOllama(
    model="llama3.2:latest"
)

class multiagent(TypedDict):
    messages : Annotated[Sequence[BaseMessage], operator.add]
    user_input : str
    
def math_calculation_node(state:multiagent)->dict:
        """Makes mathematical calculations"""
        chat_history = "\n".join([msg.content for msg in state["messages"]])
        prompt = f"You are a Mathematical assistant who answers mathematical questions clearly and precisly.. . Here is the previous conversation :\n{chat_history}\nNow answer: {state['user_input']}"
        output= llm.invoke(prompt).content
        return {'messages':[AIMessage(content=output, additional_kwargs={"agent":"Math Agent"})]}

def router(state:multiagent)->str:
    """Routes to either math_agent or knowledge_agent based on the user input """
    latest_message = state['messages'][-1].content.lower()
    if any(char.isdigit() for char in latest_message) or any(
        opr in latest_message for opr in ['+', '-', '*', '%', '/', 'add', 'subtract', 'multiply', 'divide', 'calculate', 'equation', 'expression']
    ):
        return "math_agent"
    return "knowledge_agent"

def knowledge_node(state:multiagent)->dict:
        """Makes Knowledge based answers"""
        chat_history = "\n".join([msg.content for msg in state["messages"]])
        prompt = f"You are a knowledgeable assistant who answers factual, general, and conversational questions clearly and precisly.. Here is the previous conversation :\n{chat_history}\nNow answer: {state['user_input']}"
        output= llm.invoke(prompt).content
        return {'messages':[AIMessage(content=output, additional_kwargs={"agent":"Knowledge Agent"})]}

def router_passthrough(state:multiagent):
    return state

graph = StateGraph(multiagent)
graph.add_node("math_calculation_node", math_calculation_node)
graph.add_node("knowledge_node", knowledge_node)
graph.add_node("router", router_passthrough)
graph.set_entry_point("router")
graph.add_conditional_edges(
    "router",
    router,
    {"math_agent": "math_calculation_node", "knowledge_agent": "knowledge_node"},
)
graph.add_edge("math_calculation_node", END)
graph.add_edge("knowledge_node", END)

checkpointer = MemorySaver()
if "user_input" not in st.session_state:
    st.session_state.user_input = ""
user_query = st.text_area("Ask a question:",key="user_input")
workflow=graph.compile(checkpointer)
show_graph(workflow)
col1, col2 = st.columns([1, 1])
with col1:
        send_clicked = st.button("Send",use_container_width=True)
with col2:
        reset_clicked = st.button("Reset Chat",use_container_width=True)

if reset_clicked:
    for key in ["messages", "thread_id", "user_input"]:
        if key in st.session_state:
            del st.session_state[key]
    st.session_state.thread_id = str(uuid.uuid4())
    st.rerun()
        
if send_clicked and user_query:
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    updated_messages = st.session_state.messages + [HumanMessage(content=user_query)]
    state = {
        "messages": updated_messages,
        "user_input": user_query
    }
    output = workflow.invoke(state, config=config)
    st.session_state.messages = output["messages"]

if st.session_state.messages:
        last_msg=st.session_state.messages[-1]
        if isinstance(last_msg, AIMessage):
            agent = last_msg.additional_kwargs.get("agent", "AI")
            st.markdown(f"**{agent} is answering:** {last_msg.content}")
