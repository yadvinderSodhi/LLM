import streamlit as st
import requests

# Initialize session state for chat history if not already present
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Set the title of the app
st.title("Ask Adam")

# Display chat history
for chat in st.session_state.chat_history:
    if "user" in chat:
        st.write(f"**You:** {chat['user']}")
    if "bot" in chat:
        st.write(f"**Bot:** {chat['bot']}")

# Input for the query
id = st.text_input("Enter your ID:")

# Button to trigger search with a custom style
if st.button("→", key="submit"):
        params = {
         "id": id
        }
        # Make a request to the FastAPI backend
        try:

            # Change the request to send the correct JSON structure
            response = requests.post("http://127.0.0.1:8000/process_params", json=params)
            #{"user_query": user_query})

            st.success("Params sent sucecesfully");
            st.json(response.json())
            #answer = response.json().get("response")


        except requests.exceptions.HTTPError as err:
            st.error(f"Error: {err}")

