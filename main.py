
import os
import pandas as pd
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.embeddings import CohereEmbeddings
import replicate
from dotenv import load_dotenv

__import__('pysqlite3')

import sys

sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

def create_store_vectors(embedding_function, persist_directory:str, data_file_path:str)->list:

    print('Vectorizing')
    total_documents = pd.read_csv(data_file_path)
    print('#'*100)

    data_file_path = '/Users/prabhatkumarprabhakar/PycharmProjects/pythonProject/Restaurant reviews.csv'
    total_documents = pd.read_csv(data_file_path)
    columns = ['Restaurant',
               'Reviewer',
               'Review',
               'Rating']

    total_documents = total_documents[columns]
    loader = DataFrameLoader(total_documents, page_content_column="Review")
    db = Chroma(persist_directory='/Users/prabhatkumarprabhakar/PycharmProjects/pythonProject/vec_embeddings/',
                embedding_function=embedding_function)
    db.add_documents(documents=loader.load())


def get_prompt(context,question):
    SYSTEM_PROMPT = """You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'. You are acting as a waiter in a restaurant named Beyond flavours. Which serves Indian Italian and Thai foods only."""

    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

    SYSTEM_PROMPT_template = B_SYS + SYSTEM_PROMPT + E_SYS
    context_instruction_template = f"CONTEXT::\n{context}\nUsing the context answer the following question\n\nQUESTION:{question}"
    final_prompt = '<s>'+B_INST+SYSTEM_PROMPT_template+context_instruction_template+E_INST

    return final_prompt

def generate_llama2_response(question):
    string_dialogue = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'."

    # Query Chromadb for the 10 most similar titles to the user prompt.
    context = db.similarity_search(question, k=10)
    prompt = get_prompt(context,question)
    output = replicate.run("meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",input={"prompt": prompt,
                                    "temperature": temperature,
                                    "top_p": top_p,
                                    "max_length": max_length,
                                    "repetition_penalty": 1})
    return output

persist_directory = 'vec_embeddings/'
data_file_path = 'Restaurant reviews.csv'
# embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
embedding_function = CohereEmbeddings(model="embed-english-light-v3.0", cohere_api_key = "1uaehlzgSiCUtH61q5GTAYugWieilxvFzFWHfxaj")
# create_store_vectors(embedding_function, persist_directory, data_file_path)


db = Chroma(persist_directory=persist_directory, embedding_function=embedding_function, collection_metadata={"hnsw:space":"cosine"})
# retriever = db.as_retriever(search_kwargs={'k':4}, search_type='similarity',chain_type="map-rerank")

# App title
st.set_page_config(page_title="🦙💬 Llama 2 Chatbot")


load_dotenv()
# Replicate Credentials
with st.sidebar:
    st.title('🦙💬 Llama 2 Chatbot')
    # if 'REPLICATE_API_TOKEN' in st.secrets:
    #     st.success('API key already provided!', icon='✅')
    #     # replicate_api = st.secrets['REPLICATE_API_TOKEN']
    #     replicate_api = os.getenv('REPLICATE_KEY')
    #
    # else:
    #     replicate_api = st.text_input('Enter Replicate API token:', type='password')
    #     if not (replicate_api.startswith('r8_') and len(replicate_api) == 40):
    #         st.warning('Please enter your credentials!', icon='⚠️')
    #     else:
    #         st.success('Proceed to entering your prompt message!', icon='👉')
    #
    # Refactored from https://github.com/a16z-infra/llama2-chatbot
    st.subheader('Models and parameters')
    selected_model = st.sidebar.selectbox('Choose a Llama2 model', ['Llama2-70B'],
                                          key='selected_model')
    if selected_model == 'Llama2-70B':
        llm = 'r8_cgkgOim7f8kbOS9QoVaKWEKNVqF5JwP0gpUlV'

    temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=5.0, value=0.1, step=0.01)
    top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
    max_length = st.sidebar.slider('max_length', min_value=64, max_value=4096, value=512, step=8)
    

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]


st.sidebar.button('Clear Chat History', on_click=clear_chat_history)


# User-provided prompt
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)


source_documents_op = []
# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        # start_time = time.time()
        with st.spinner("Thinking..."):
            response = generate_llama2_response(prompt)
            # source_documents_op = json.dumps(jsonable_encoder(source_documents), indent=4)
            placeholder = st.empty()
            full_response = ''
            for item in response:
                full_response += item
                placeholder.markdown(full_response)
                # time.sleep(0.00008)
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)


    # st.json(source_documents_op)