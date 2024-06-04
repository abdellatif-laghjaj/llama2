import streamlit as st
import replicate
import os

# App title
st.set_page_config(page_title="Llama2 🦙 Chatbot")

# Replicate API key (replace with your actual API key)
REPLICATE_API_TOKEN = "r8_QJrQ4jZsbQ1QUdZQsmAcZSibKsNeOzI3PbUUa"
os.environ['REPLICATE_API_TOKEN'] = REPLICATE_API_TOKEN

# Sidebar for model selection and parameters
st.sidebar.title('🦙💬 Llama 2 Chatbot')
st.sidebar.write('This chatbot is powered by the open-source Llama 2 LLM.')

selected_model = st.sidebar.selectbox('Choose a Llama2 model', ['Llama2-7B', 'Llama2-13B'], key='selected_model')
if selected_model == 'Llama2-7B':
    llm = 'a16z-infra/llama7b-v2-chat:4f0a4744c7295c024a1de15e1a63c880d3da035fa1f49bfd344fe076074c8eea'
elif selected_model == 'Llama2-13B':
    llm = 'a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5'

temperature = st.sidebar.slider('Temperature', min_value=0.01, max_value=1.0, value=0.1, step=0.01,
                                help="Controls the randomness of the generated text. Higher values make the output more creative.")
top_p = st.sidebar.slider('Top P', min_value=0.01, max_value=1.0, value=0.9, step=0.01,
                          help="Controls the diversity of the generated text. Lower values make the output more focused.")
max_length = st.sidebar.slider('Max Length', min_value=32, max_value=256, value=120, step=8,
                               help="Maximum number of tokens in the generated response.")

# Store messages in session state
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


# Clear chat history function
def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]


st.sidebar.button('Clear Chat History', on_click=clear_chat_history)


# Generate LLaMA2 response
def generate_llama2_response(prompt_input):
    string_dialogue = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'."
    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            string_dialogue += "User: " + dict_message["content"] + "\n\n"
        else:
            string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"
    try:
        output = replicate.run(
            llm,
            input={"prompt": f"{string_dialogue} {prompt_input} Assistant: ",
                   "temperature": temperature,
                   "top_p": top_p,
                   "max_length": max_length,
                   "repetition_penalty": 1.1}
        )
        return output
    except replicate.exceptions.ReplicateError as e:
        st.error(f"Error generating response: {e}")
        return []


# User input
if prompt := st.chat_input("Enter your message here:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Generate and display assistant's response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_llama2_response(prompt)
                full_response = ''.join(response)  # Combine the response chunks
                st.write(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
