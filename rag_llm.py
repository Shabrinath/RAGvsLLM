import streamlit as st
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
import os

# Streamlit title and description
st.title("Wikipedia + LLM Response")
st.write("This application demonstrates three approaches to generating responses:")

# Bullet points explanation
st.markdown("""
- **Wikipedia API (Langchain Tool):** Retrieves information directly from Wikipedia using the Langchain tool, which interacts with the Wikipedia API to return the most relevant data.
- **LLM (Without External Sources):** Generates responses solely based on its internal training data, without relying on any external sources like Wikipedia.
- **LLM + Wikipedia (Retrieval + Generation):** Combines the power of Wikipedia retrieval with an LLM. First, it fetches data using the Wikipedia API, then enhances the response with the LLM to generate a more accurate and relevant answer.
""")

# The query from the user
user_prompt = st.text_input("Enter your query", "iphone 16?")

if user_prompt:
    # Initialize Wikipedia API wrapper
    api_wrapper = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=2000)

    # Create a Wikipedia search tool
    wiki_search = WikipediaQueryRun(api_wrapper=api_wrapper)

    # Use the tool to search Wikipedia for a specific query
    result = wiki_search.invoke({"query": user_prompt})

    # Wikipedia response
    st.subheader("Wikipedia response")
    st.write(result)

    # Response from LLM only
    llm = OpenAI(temperature=0.6)
    llm_response = llm.invoke(user_prompt)
    
    st.subheader("LLM response")
    st.write(llm_response)

    # Wiki + LLM combined response
    template = """
    You are an expert in answering general questions. Here's information retrieved from Wikipedia about "{query}":
    {wikipedia_content}

    Please check if the retrieved info is relevant to the query or not. If it is not relevant, please feel free to say that and add your inputs to make it relevant. If the info is relevant, please answer the query and summarize the answer in simple terms.
    """
    
    # Create a PromptTemplate with placeholders for the query and content
    prompt_template = PromptTemplate(input_variables=["query", "wikipedia_content"], template=template)
    
    # Format the prompt using the template
    formatted_prompt = prompt_template.format(query=user_prompt, wikipedia_content=result)
    
    # Use the LLM to generate a summary based on the Wikipedia result
    summary = llm.invoke(formatted_prompt)
    
    # Wiki + LLM response
    st.subheader("Wiki + LLM response")
    st.write(summary)
