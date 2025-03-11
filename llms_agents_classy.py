import base64
import json
import os
import re

import requests
import torch
from IPython.display import Markdown, display
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_deepseek import ChatDeepSeek
from langchain_gigachat.chat_models.gigachat import GigaChat
from langchain_gigachat.embeddings import GigaChatEmbeddings
from langchain_openai import OpenAI
from langchain_openai.chat_models import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

from langgraph.graph import END, START, MessagesState, StateGraph

from langgraph.types import Command
from openai import OpenAI
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_fixed
from transformers import AutoTokenizer


def printmd(string):
    display(Markdown(string))


def known_llms():
    return ["Qwen2.5-7B-Instruct", "Qwen2.5-32B-Instruct", "GigaChat MAX"]

def init_db(docs, embeddings_model):
    collection_name = "langchain"
    Chroma(collection_name=collection_name).delete_collection()

    for doc in docs:
        file_path = "documents/" + doc
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=3000,
                chunk_overlap=200,
                length_function=len,
                keep_separator=True,
            )
            chunks = splitter.split_text(content)
            print(f"Количество чанков в документе {doc}: {len(chunks)}")

            metadatas = [{"document": doc} for _ in range(len(chunks))]
            vectorstore = Chroma.from_texts(
                chunks, embedding=embeddings_model, metadatas=metadatas
            )

    print(f"Общее количество чанков в БД: {len(vectorstore.get()['ids'])}")
    return vectorstore



class ProcessorState(MessagesState):
    request_txt: str  # Сам запрос гражданина
    category: str  # Категория запроса
    category_reason: str  # Текст "мотивировка" категорирования
    category_list: (
        str  # Список потенциально известных категорий (на которые пойдет маршрутизация)
    )
    retriever_docs: list[(str, str)]  # список документов

    history: list[(str, str)]  # История обработки гражданину

    response: str  # текущий ответ
    critique: str  # текущая критика
    # thoughts: str # о чем думал критик
    # critique: list[str]  # Критика (массив)
    is_new_critique: bool  # есть ли новая критика
    # critique_action: str  # Что критик предлагает делать далее
    final_decision: bool  # устраивает ли критика ответ
    out: str  # Итоговый ответ


def agent_dict_generator(obj_class, promt_template, retry_n=1, model=None):
    @retry(stop=stop_after_attempt(retry_n))
    def agent_func(state: ProcessorState):
        parser = PydanticOutputParser(pydantic_object=obj_class)

        prompt = ChatPromptTemplate.from_messages([("system", promt_template)]).partial(
            format_instructions=parser.get_format_instructions()
        )

        chain = prompt | model | parser

        last_reply, last_critique = state.get("history", [("", "")])[-1]
        all_critique_ex_last = [v for k, v in state.get("history", [("", "")])][0:-1]
        # print(all_critique_ex_last)

        resp = chain.invoke(
            {
                "category": state.get("category", ""),
                "category_list": state.get("category_list", ""),
                "request_txt": state.get(
                    "request_txt", "тестовое обращение вымышленного гражданина"
                ),
                "response": state.get("response", ""),
                "last_reply": last_reply,
                "last_critique": last_critique,
                "all_critique_ex_last": ";".join(all_critique_ex_last),
                "retriever_docs": ";".join(
                    [f"{k}:{v}" for k, v in state.get("retriever_docs", [("", "")])]
                ),
            }
        )

        update = resp.__dict__

        if "critique" in resp.__dict__.keys():  # and resp.critique != last_critique:
            history = state.get("history", [])
            history.append((state.get("response"), resp.critique))
            update["history"] = history

        return update

    return agent_func


def agent_str_generator(attr_name, promt_template, retry_n=1, model=None):
    @retry(stop=stop_after_attempt(retry_n))
    def agent_func(state: ProcessorState):
        parser = StrOutputParser()

        prompt = ChatPromptTemplate.from_messages([("system", promt_template)])

        chain = prompt | model | parser

        last_reply, last_critique = state.get("history", [("", "")])[-1]
        all_critique_ex_last = [
            v for k, v in state.get("history", [("", "")])
        ]  # [0:-1]

        resp = chain.invoke(
            {
                "category": state.get("category", ""),
                "category_list": state.get("category_list", ""),
                "request_txt": state.get(
                    "request_txt", "тестовое обращение вымышленного гражданина"
                ),
                "last_reply": last_reply,
                "last_critique": last_critique,
                "all_critique_ex_last": ";".join(all_critique_ex_last),
                "retriever_docs": ";".join(
                    [f"{k}:{v}" for k, v in state.get("retriever_docs", [("", "")])]
                ),
            }
        )

        return {attr_name: resp}

    return agent_func


def plain_agent_generator(promt_template, retry_n=1, model=None):
    @retry(stop=stop_after_attempt(retry_n))
    def agent_func(**kwargs):
        parser = StrOutputParser()

        prompt = ChatPromptTemplate.from_messages([("system", promt_template)])

        chain = prompt | model | parser

        resp = chain.invoke(kwargs)

        return resp

    return agent_func


def retriever_generator(vectorstore, retry_n=1):
    @retry(stop=stop_after_attempt(retry_n))
    def retriever(state: ProcessorState):
        cat = state.get("category")
        docs = vectorstore.similarity_search(cat[0:2048], k=4)  # Depends on embeddings

        resp = []
        for doc in docs:
            resp.append((docs[0].metadata["document"], doc.page_content))

        return {"retriever_docs": resp}

    return retriever


