# мой провайдер https://api.vsegpt.ru/v1

import openai
import os
import getpass
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings # настройка глобальных параметров фреймворка
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import SQLDatabase
import sqlalchemy as db
from llama_index.core.query_engine import NLSQLTableQueryEngine

base_url  = input("Введите OpenAI API провайдера:")
key = input("Введите OpenAI API Key:")

os.environ["OPENAI_API_KEY"] = key
os.environ["OPENAI_BASE_URL"] = base_url
os.environ["BASE_URL"] = base_url

engine = db.create_engine("sqlite:///./goods.db")
metadata_obj = db.MetaData().create_all(engine)
sql_database = SQLDatabase(engine, metadata_obj)

# Эмбеддинги HGFACE (нужен токен)
Settings.embed_model = HuggingFaceEmbedding(
    model_name="cointegrated/rubert-tiny2",
    device="cpu",
    embed_batch_size=8
)

Settings.llm = OpenAI(
    model_name="gpt-3.5-turbo", 
    temperature=0.1,
    api_base=base_url,
    api_key=key,
    request_timeout=1000,
    max_retries=3,
)

Settings.chunk_size = 512

query_engine = NLSQLTableQueryEngine(sql_database=sql_database)
query = "Сколько всего мелкой бытовой техники в филиале №2?"

message_template =f"""
Ты Нейро-ассистент. Отвечай в соответствии с Источником. Проверь, есть ли в Источнике упоминания о ключевых словах Вопроса.
 Не придумывай!
Вопрос: {query}
"""

response = query_engine.query(message_template)

print('Ответ:')
print(response)