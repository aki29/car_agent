import os, signal, pytz, time, uuid
from termcolor import colored

from agent.chains.chat_chain import get_chat_chain
from agent.rag.retriever import get_retriever
from agent.memory.manager import init_db
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from dotenv import load_dotenv, find_dotenv

timezone = pytz.timezone("Asia/Taipei")
llm_config = {"callbacks": [StreamingStdOutCallbackHandler()]}
persist_directory = os.environ.get("PERSIST_DIRECTORY", "db")


def signal_handler(sig, frame):
    print("\nInterrupted!!")
    exit(0)


def init_model():
    model = ChatOllama(
        model=os.environ.get("LLM_MODEL_NAME", "gemma3:1b"),
        base_url="http://localhost:11434",
        temperature=0.7,
    )
    embeddings = OllamaEmbeddings(
        base_url="http://localhost:11434",
        model=os.environ.get("EMBEDDINGS_MODEL_NAME", "nomic-embed-text:latest"),
    )
    return model, embeddings


def main():
    signal.signal(signal.SIGINT, signal_handler)
    model, embed = init_model()
    init_db()
    retriever = get_retriever(embed)
    user_id = input("Please enter your user ID: ").strip() or str(uuid.uuid4())
    chat = get_chat_chain(user_id, model, retriever)

    print("\n[In-Car Assistant activated. Type /exit to end the conversation.]")

    while True:
        query = input("\nQuery: ").strip()
        if not query:
            continue

        start = time.time()
        try:
            response = chat.invoke(
                {"question": query}, config={"configurable": {"session_id": user_id}}
            )
            if response == "[exit]":
                print("bye！")
                break
            print("Assistant:", response)
        except Exception as e:
            print(f"[error] {e}")
        print(colored(f"({time.time() - start:.2f}s)", "blue"))


if __name__ == "__main__":
    main()
