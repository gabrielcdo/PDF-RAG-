import os
from langchain_openai import ChatOpenAI
from utils.functions import create_retriever, create_rag_chain

# Set up your API keys
# os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"
# os.environ["GROQ_API_KEY"] = "YOUR_GROQ_API_KEY"
# os.environ["LANGCHAIN_API_KEY"] = "YOUR_LANGCHAIN_API_KEY"
# os.environ["LANGCHAIN_TRACING_V2"] = "true"

def main():
    """
    Main function to run the RAG chatbot.
    """
    # Create the retriever
    retriever = create_retriever(
        persist_directory="./chroma_db", docstore_path="./docstore"
    )

    # Create the RAG chain
    model = ChatOpenAI(model="gpt-4o-mini")
    chain = create_rag_chain(retriever, model)

    print("RAG Chatbot is ready. Type 'exit' to quit.")
    while True:
        question = input("You: ")
        if question.lower() == 'exit':
            break
        response = chain.invoke(question)
        print("Bot:", response)

if __name__ == "__main__":
    main()
