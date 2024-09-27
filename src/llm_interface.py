# src/llm_interface.py

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from config.constants import MODEL_NAME, GROQ_API_KEY_ENV


class LLMInterface:
    def __init__(self):
        load_dotenv()
        api_key = os.getenv(GROQ_API_KEY_ENV)
        self.llm = ChatGroq(model=MODEL_NAME, groq_api_key=api_key)

    def invoke_llm(self, template: str, **kwargs) -> str:
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm
        return chain.invoke(kwargs).content

llm_interface = LLMInterface()