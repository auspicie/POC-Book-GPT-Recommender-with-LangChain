import os
from dotenv import load_dotenv

import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper

# Load .env and API key
load_dotenv()

# App UI
st.title('📚🔗 Book GPT Recommender 📚')
prompt = st.text_input("Type in a book title, you'll get back some good recommendations:")

# Prompt templates
title_template = PromptTemplate(
    input_variables=['topic'],
    template='Give me some book titles about {topic}'
)

script_template = PromptTemplate(
    input_variables=['title', 'wikipedia_research'],
    template='Write me a summary script of the given topic based on this title TITLE: {title} while leveraging this Wikipedia research: {wikipedia_research}'
)

# Memory
title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')

# LLM
llm = OpenAI(temperature=0.9)

# Chains
title_chain = LLMChain(llm=llm, prompt=title_template, output_key='title', memory=title_memory, verbose=True)
script_chain = LLMChain(llm=llm, prompt=script_template, output_key='script', memory=script_memory, verbose=True)

# Wikipedia API
wiki = WikipediaAPIWrapper()

# Main logic
if prompt:
    title = title_chain.run(prompt)

    st.subheader("📘 Recommended Book Titles")
    for book in title.split('\n'):
        if book.strip():
            search_url = f"https://www.google.com/search?q={book.replace(' ', '+')}+book"
            st.markdown(f"- [{book}]({search_url})")

    wiki_research = wiki.run(prompt)
    script = script_chain.run(title=title, wikipedia_research=wiki_research)

    st.subheader("📄 GPT Summary Script")
    st.write(script)

    with st.expander('🧠 Title History'):
        st.info(title_memory.buffer)

    with st.expander('🧠 Script History'):
        st.info(script_memory.buffer)

    with st.expander('🌐 Wikipedia Research'):
        st.info(wiki_research)
st.markdown(
    '[📘 Learn more about this app](https://github.com/auspicie/POC-Book-GPT-Recommender-with-LangChain) &nbsp;|&nbsp; [🚀 Powered by LangChain](https://www.langchain.com/) &nbsp;|&nbsp; [🔑 Get your OpenAI API key](https://platform.openai.com/account/api-keys)',
    unsafe_allow_html=True
)
