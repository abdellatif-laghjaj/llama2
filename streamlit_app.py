import streamlit as st
import replicate
import os
from dotenv import load_dotenv

# App title and favicon
st.set_page_config(
    page_title="Llama2 🦙 Chatbot",
    page_icon="🦙"
)

# Load environment variables from .env
load_dotenv()
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")


# --- Functions ---
def generate_llama2_response(prompt_input, model, temperature, top_p, max_length):
    string_dialogue = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'."
    for message in st.session_state.messages:
        if message["role"] == "user":
            string_dialogue += "User: " + message["content"] + "\n\n"
        else:
            string_dialogue += "Assistant: " + message["content"] + "\n\n"

    try:
        output = replicate.run(
            model,
            input={
                "prompt": f"{string_dialogue} {prompt_input} Assistant: ",
                "temperature": temperature,
                "top_p": top_p,
                "max_length": max_length,
                "repetition_penalty": 1.1
            }
        )
        return ''.join(output)
    except replicate.exceptions.ReplicateError as e:
        st.error(f"Error generating response: {e}")
        return ""


def display_chat_history():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])


# --- Sidebar ---
st.sidebar.title('🦙💬 Llama 2 Chatbot')
st.sidebar.write('This chatbot is powered by Meta\'s Llama 2 LLM.')

# Model Selection
selected_model = st.sidebar.selectbox(
    'Choose a Llama2 model',
    [
        "Llama2-7B (Faster)",
        "Llama2-13B (More Powerful)"
    ],
    key='selected_model'
)

# Model Mapping
model_dict = {
    "Llama2-7B (Faster)": "a16z-infra/llama7b-v2-chat:4f0a4744c7295c024a1de15e1a63c880d3da035fa1f49bfd344fe076074c8eea",
    "Llama2-13B (More Powerful)": "a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5"
}
llm = model_dict[selected_model]

# Advanced Parameters (Collapsible)
with st.sidebar.expander("Advanced Parameters"):
    temperature = st.slider(
        'Temperature',
        min_value=0.01,
        max_value=1.0,
        value=0.1,
        step=0.01,
        help="Controls the creativity of the response. Higher is more random."
    )
    top_p = st.slider(
        'Top P',
        min_value=0.01,
        max_value=1.0,
        value=0.9,
        step=0.01,
        help="Controls the focus of the response. Lower is more focused."
    )
    max_length = st.slider(
        'Max Length',
        min_value=32,
        max_value=512,
        value=256,
        step=8,
        help="Maximum length of the generated response."
    )

# Clear Chat History
if st.sidebar.button('Clear Chat History'):
    st.session_state.messages = [
        {"role": "assistant", "content": "How may I assist you today?"}
    ]

# --- Main Chat Area ---
st.title("🦙💬 Llama 2 Chatbot")

# Initialize chat history
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "How may I assist you today?"}
    ]

# Display chat history
display_chat_history()

# User Input
if prompt := st.chat_input("Enter your message here:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Generate and display assistant's response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_llama2_response(
                prompt, llm, temperature, top_p, max_length
            )
            st.write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
