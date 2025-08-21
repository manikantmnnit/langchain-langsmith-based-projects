import langchain
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_perplexity import ChatPerplexity
import os
import streamlit as st
import langsmith
load_dotenv()

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")


st.title('Research Paper Writer with Perplexity AI')
api_key=st.text_input("Enter your Perplexity API Key:", type="password",help="You can get your API key from https://www.perplexity.ai/")


temperature=st.slider("Select Temperature:", min_value=0.0, max_value=2.0, step=0.1, value=0.7)


# LLM
llm = ChatPerplexity(
    model="sonar",
    temperature=temperature,
    pplx_api_key=os.getenv("PERPLEXITY_API_KEY") if api_key=="" else api_key
)

# Prompt
template = """ 
write a short article about {keyword} with {word_count} words like mentioned in a research paper and references should be in {reference_style} format.
     
Follow these instructions:
1. Use the keyword "{keyword}" naturally throughout the article.
2. The tone should be engaging and suitable for a general audience.
3. Ensure the article is approximately {word_count} words long.
4. Contents should be simple in language and easy to understand.
5. Also provide papers in reference section and cite it in the short paper.
6 . The reference style should be {reference_style}. No need to copy the references mentioned in the reference style. That is just for reference
"""

prompt = PromptTemplate(
    template=template,
    input_variables=['keyword', 'word_count', 'reference_style']
)

parser = StrOutputParser()

# Chain
chain = prompt | llm | parser

def generate_article(keyword, word_count,temperature=0.7):
    result = chain.invoke({
        "keyword": keyword,
        "word_count": word_count,
        "reference_style": reference_format 
    })
    return result

keyword = st.text_input("Enter a keyword:")
word_count = st.slider("Select word count:", min_value=300, max_value=1000, step=100, value=300)
reference_format=st.text_input(label="Reference Format",  help="You can choose from APA, MLA, Chicago, Harvard, etc.")
submit_button = st.button("Generate Article")
if submit_button:
    message=st.empty()
    message.text("Thinking")
    article = generate_article(keyword, word_count,reference_format)
    message.text("")
    st.write(article)
    st.download_button(
        label="Download article",
        data=article,
        file_name= 'Article.txt',
        mime='text/txt',

    )

