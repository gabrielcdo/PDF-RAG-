import base64
from IPython.display import Image, display
from unstructured.partition.pdf import partition_pdf
from langchain.schema.document import Document
import uuid
from langchain.vectorstores import Chroma
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
    """
    prompt_template = """Descreva a imagem em detalhes. Para contexto,
    a imagem faz parte de um artigo de pesquisa que explica a arquitetura de transformers.
    Seja específico em relação aos gráficos, como gráficos de barras."""
    messages = [
        (
            "user",
            [
                {"type": "text", "text": prompt_template},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,{image}"},
                },
            ],
        )
    ]
    prompt = ChatPromptTemplate.from_messages(messages)
    chain = prompt | model | StrOutputParser()
    # run in batchs of 2 and then put a sleep of 15 seconds 
    for i in tqdm(range(0, len(images), 2)):
        batch = images[i:i + 2]
        summaries = chain.batch(batch, {"max_concurrency": 3})
        for summary in summaries:
            print(summary)
        if i + 2 < len(images):
            import time
            time.sleep(30)
            
    return summaries


def create_retriever(
    vectorstore_collection_name="multi_modal_rag_summaries",
    docstore_collection_name="multi_modal_rag_documents",
    persist_directory="./indexes/chroma_db",
):
    """
    Creates a multi-vector retriever.
    The vectorstore (Chroma) stores the summaries.
    The docstore (also Chroma) stores the raw documents.
    """
    # Create two Chroma vector stores
    vectorstore = Chroma(
        collection_name=vectorstore_collection_name,
        embedding_function=OpenAIEmbeddings(),
        persist_directory=persist_directory,
    )
    docstore = Chroma(
        collection_name=docstore_collection_name,
        embedding_function=OpenAIEmbeddings(), # docstore needs an embedding function too
        persist_directory=persist_directory,
    )
    id_key = "doc_id"

    # The retriever will fetch summaries from the vectorstore...
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        # ...and use the doc_ids to fetch the original documents from the docstore
        docstore=docstore,
        id_key=id_key,
    )
    return retriever


def add_documents_to_retriever(retriever, docs, summaries):
    """
    Adds documents and their summaries to the retriever's stores.
    """
    doc_ids = [str(uuid.uuid4()) for _ in docs]

    # Add summaries to the vectorstore
    summary_docs = [
        Document(page_content=summary, metadata={retriever.id_key: doc_ids[i]})
        for i, summary in enumerate(summaries)
    ]
    retriever.vectorstore.add_documents(summary_docs)

    # Add original documents to the docstore
    # We need to convert the raw text/image content to Document objects
    original_docs = []
    for i, doc_content in enumerate(docs):
        # Check if the content is an image (base64) or text
        if isinstance(doc_content, str) and doc_content.startswith('iVBOR') or len(doc_content) > 1000:
             # It's likely a base64 image
             doc = Document(page_content=doc_content, metadata={retriever.id_key: doc_ids[i]})
        elif hasattr(doc_content, 'text'): # For unstructured elements
             doc = Document(page_content=doc_content.text, metadata={retriever.id_key: doc_ids[i]})
        else: # Plain text
             doc = Document(page_content=str(doc_content), metadata={retriever.id_key: doc_ids[i]})
        original_docs.append(doc)

    retriever.docstore.add_documents(original_docs)

def parse_docs(docs):
    """Split base64-encoded images and texts"""
    b64 = []
    text = []
    for doc in docs:
        try:
            # This is a hacky way to check if the doc is a base64 string
            if len(doc) > 1000 and doc.endswith('='):
                b64.append(doc)
            else:
                text.append(doc)
        except Exception as e:
            text.append(doc)
    return {"images": b64, "texts": text}


def build_prompt(kwargs):
    """
    Builds a prompt for the RAG chain.
    """
    docs_by_type = kwargs["context"]
    user_question = kwargs["question"]

    context_text = ""
    if len(docs_by_type["texts"]) > 0:
        for text_element in docs_by_type["texts"]:
            context_text += text_element.page_content

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
