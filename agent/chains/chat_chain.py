from pathlib import Path
import json
from sqlalchemy import create_engine
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories.sql import SQLChatMessageHistory

from agent.memory.extractor import extract_memory_kv_chain
from agent.memory.manager import append_chat, save_memory, load_memory, clear_memory

DB_PATH = Path(__file__).parent.parent / "data" / "ctk_memory.sqlite3"


def get_chat_prompt():
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are an intelligent and friendly in-car voice assistant. You can understand and automatically respond in the language the user uses—Chinese, English, or Japanese. Your response must always match the user's language and must never switch languages.
Your tone should be warm, concise, and emotionally aware, like a thoughtful companion sitting beside the driver and speaking gently.
Avoid using any emojis or emoticons.
When the user shares something meaningful—such as personal preferences, life events, or important information—acknowledge it kindly and store it using a unique key. Use this information in the future to provide personalized responses that meet the user's needs.
If you are unsure about something, ask politely and gently.
Avoid repeating words or phrases. Keep your language clear, natural, supportive, sincere, and friendly.        
""",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )


def get_chat_chain(user_id: str, model, retriever=None):
    engine = create_engine(f"sqlite:///{DB_PATH}")
    message_history = SQLChatMessageHistory(session_id=user_id, connection=engine)
    prompt = get_chat_prompt()
    extract_chain = extract_memory_kv_chain(model)

    def store_and_extract(input):
        user_input = input["question"]
        print("AKI-->",user_input)
        if user_input.strip() == "/memory":
            mem = load_memory(user_id)
            return {"question": f"目前記憶: {mem if mem else '尚無資料'}", "chat_history": []}
        if user_input.strip() == "/clear":
            clear_memory(user_id)
            return {"question": "🧹 已清除使用者記憶", "chat_history": []}
        if user_input.strip() == "/exit":
            return {"question": "[exit]", "chat_history": []}

        append_chat(user_id, "user", user_input)

        try:
            extracted = extract_chain.invoke({"text": user_input})
            if isinstance(extracted, dict):
                for k, v in extracted.items():
                    save_memory(user_id, k.strip(), v.strip())
        except Exception as e:
            print(f"[!] Memory extraction failed: {e}")

        return {"question": user_input, "chat_history": message_history.messages}

    chain = RunnableLambda(store_and_extract) | prompt | model.with_config({"callbacks": []})

    return (
        RunnableWithMessageHistory(
            chain,
            lambda _: message_history,
            input_messages_key="question",
            history_messages_key="chat_history",
        )
        | StrOutputParser()
    )
