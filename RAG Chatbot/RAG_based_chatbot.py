from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama,OllamaEmbeddings
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import gradio as gr
import uuid

load_dotenv(override=True)

prompt=PromptTemplate.from_template("""
            You are a movie trivia chat expert. Use the following context to answer the question.
            Context:
            {context}
            Conversation:
            {question}                         
            """)

loader=PyPDFLoader("movies_trivia.pdf")
documents=loader.load()
text_splitter=CharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=20,)
chunks = text_splitter.split_documents(documents)

vector_store = Chroma(
    embedding_function=OllamaEmbeddings(model="granite-embedding:latest"),
    persist_directory='my_chroma_db',
    collection_name='movie_trivia')

if len(vector_store.get()["ids"]) == 0:
    vector_store.add_documents(chunks)

store={}
def get_session_history(session_id):
    if session_id not in store:
        store[session_id]=InMemoryChatMessageHistory()
    return store[session_id]

def movie_chatbot(input,session_id,history_state,temperature):
    if history_state is None:
        history_state=[]

    if not input.strip():
        error_message="Please enter valid input!"
        history_state.append((input,error_message))
        return error_message,session_id,history_state
   
    llm=ChatOllama(model="llama3.2:latest",temperature=temperature)

    retriever_qa_output = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        input_key="question",
        return_source_documents=True
    )
    chat_history = get_session_history(session_id)
    chat_history_messages = chat_history.messages

    history_text = ""
    for msgs in chat_history_messages:
        if msgs.type == "human":
            history_text += f"User:{msgs.content}\n"
        elif msgs.type == "ai":
            history_text += f"Chatbot:{msgs.content}\n"

    question = f"{history_text}\nUser:{input}"

    result=retriever_qa_output.invoke({"question": question},config={"configurable": {"session_id": session_id}})
    answer =result.get('result',"Couldn't find answer to your question")

    chat_history.add_user_message(input)
    chat_history.add_ai_message(answer)

    history_state.append((input,answer))
    return answer,session_id,history_state

def clear_history(session_id):
    if session_id in store:
        store[session_id].clear()
    return '','',str(uuid.uuid4()),None,0.5

with gr.Blocks() as moviebot:
    gr.Markdown("## Hello,I am MovieBot,your movie trivia expert.Ask me anything about films!")
    input_text=gr.Textbox(label="Input",placeholder="Type your question here")
    output_text=gr.Textbox(label="Output")
    history_state=gr.State(value=None)
    session_id=gr.State(value=str(uuid.uuid4()))
    with gr.Row():
        submit_button=gr.Button(value="Submit")
        clear_button=gr.Button(value="Clear History")
    temperature = gr.Slider(
        minimum=0.0,
        maximum=1.0,
        value=0.5,
        label="Temperature",
        step=0.1
    )
    submit_button.click(
        fn=movie_chatbot,
        inputs=[input_text,session_id,history_state,temperature],
        outputs=[output_text,session_id,history_state]
    )
    clear_button.click(
        fn=clear_history,
        inputs=[session_id],
        outputs=[input_text,output_text,session_id,history_state,temperature]
    )

moviebot.launch()