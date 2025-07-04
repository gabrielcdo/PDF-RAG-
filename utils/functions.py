import base64
from IPython.display import Image, display
from unstructured.partition.pdf import partition_pdf
from langchain.schema.document import Document
import uuid
from langchain_chroma import Chroma
from langchain.storage import LocalFileStore
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from tqdm import tqdm

def partition_pdf_elements(file_path):
    """
    Partitions a PDF into text, and images.
    """
    return partition_pdf(
        filename=file_path,
        infer_table_structure=True,
        strategy="hi_res",
        extract_image_block_types=["Image"],
        extract_image_block_to_payload=True,
        chunking_strategy="by_title",
        max_characters=10000,
        combine_text_under_n_chars=2000,
        new_after_n_chars=6000,
    )

def get_images_base64(chunks):
    """
    Get base64 encoded images from the chunks
    """
    images_b64 = []
    for chunk in chunks:
        if "CompositeElement" in str(type(chunk)):
            chunk_els = chunk.metadata.orig_elements
            for el in chunk_els:
                if "Image" in str(type(el)):
                    images_b64.append(el.metadata.image_base64)
    return images_b64

def display_base64_image(base64_code):
    """
    Displays a base64 encoded image.
    """
    image_data = base64.b64decode(base64_code)
    display(Image(data=image_data))

def summarize_elements(elements, model):
    """
    Summarizes a list of elements using a language model.
    """
    prompt_text = """
    Você é um assistente encarregado de descrever textos.
    Forneça um resumo conciso do texto

    Responda apenas com o resumo, sem comentários adicionais
    Não comece sua mensagem dizendo "Aqui está um resumo" ou algo parecido.
    Apenas forneça o resumo como ele é.

    trecho de texto: {element}
    """
    prompt = ChatPromptTemplate.from_template(prompt_text)
    summarize_chain = prompt | model | StrOutputParser()
    return summarize_chain.batch(elements, {"max_concurrency": 3})

def summarize_images(images, model):
    """
    Summarizes a list of images using a multimodal language model.
    Processes in batches of 2 with a 30-second pause between batches.
    """
    prompt_text = (
        "Descreva a imagem em detalhes. Para contexto, "
        "a imagem faz parte de um artigo de pesquisa que explica a arquitetura de transformers. "
        "Seja específico em relação aos gráficos, como gráficos de barras."
    )

    results = []

    for i in tqdm(range(0, len(images), 2)):
        batch = images[i:i + 2]
        batch_summaries = []

        for image_base64 in batch:
            messages = [
                ("user", [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                ])
            ]
            prompt = ChatPromptTemplate.from_messages(messages)
            chain = prompt | model | StrOutputParser()

            try:
                summary = chain.invoke({})
                print(summary)
                batch_summaries.append(summary)
            except Exception as e:
                print(f"Error processing image: {e}")
                batch_summaries.append("Erro ao processar a imagem.")

        results.extend(batch_summaries)

        
        import time
        time.sleep(30) 

    return results

def create_retriever(
    collection_name="multi_modal_rag",
    persist_directory="./indexes/chroma_db",
    docstore_path="./indexes/docstore",
):
    """
    Creates a persistent multi-vector retriever.
    """
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=OpenAIEmbeddings(),
        persist_directory=persist_directory,
    )

    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        byte_store=LocalFileStore(docstore_path),  # persistent docstore
        id_key="doc_id",
    )

    return retriever


import uuid

def add_documents_to_retriever(retriever, docs, summaries):
    """
    Adds documents and their summaries to the retriever.
    """
    doc_ids = [str(uuid.uuid4()) for _ in docs]

    # Adiciona os summaries ao vectorstore
    summary_texts = [
        Document(page_content=summary, metadata={"doc_id": doc_ids[i]})
        for i, summary in enumerate(summaries)
    ]
    retriever.vectorstore.add_documents(summary_texts)

    # Converte os docs para instâncias de Document
    doc_objects = [
        Document(page_content=docs[i], metadata={"doc_id": doc_ids[i]})
        for i in range(len(docs))
    ]
    retriever.docstore.mset(list(zip(doc_ids, doc_objects)))


def parse_docs(docs):
    """Split base64-encoded images and texts"""
    b64 = []
    text = []
    for doc in docs:
        if doc.metadata.get("source") == "image":
            b64.append(doc.page_content)
        else:
            text.append(doc.page_content)
    return {"images": b64, "texts": text}


def build_prompt(kwargs):
    """
    Builds a prompt for the RAG chain.
    """
    docs_by_type = kwargs["context"]
    user_question = kwargs["question"]

    context_text = ""
    if len(docs_by_type["texts"]) > 0:
        context_text = "\n\n".join(docs_by_type["texts"])


    prompt_template = f"""
    Responda à pergunta com base apenas no seguinte contexto, que pode incluir texto e a imagem abaixo.
    Contexto: {context_text}
    Pergunta: {user_question}
    """

    prompt_content = [{"type": "text", "text": prompt_template}]

    if len(docs_by_type["images"]) > 0:
        for image in docs_by_type["images"]:
            prompt_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                }
            )

    return ChatPromptTemplate.from_messages(
        [
            HumanMessage(content=prompt_content),
        ]
    )

def create_rag_chain(retriever, model):
    """
    Creates a RAG chain.
    """
    chain = (
        {
            "context": retriever | RunnableLambda(parse_docs),
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(build_prompt)
        | model
        | StrOutputParser()
    )
    return chain
