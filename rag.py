from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from policy_loader import load_policy_documents

EMBED_MODEL = "nomic-embed-text"
COLLECTION_NAME = "university_it_policies"

## The vector store is built once and then reused, so that the
## documents are not embedded again for every question.
_vector_store = None


def build_vector_store():

    documents = load_policy_documents()

    embeddings = OllamaEmbeddings(
        model=EMBED_MODEL
    )

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings
    )

    vector_store.add_documents(documents)

    return vector_store


def get_vector_store():

    global _vector_store

    if _vector_store is None:
        _vector_store = build_vector_store()

    return _vector_store


def search_policies(question, k=3):

    vector_store = get_vector_store()

    results = vector_store.similarity_search(
        question,
        k=k
    )

    return results
