from rag import search_policies

TOOL_NAME = "search_university_policies"

TOOL_DESCRIPTION = (
    "Search the university IT policy documents for the passages "
    "that are closest in meaning to the question. Returns the "
    "matching passages with their policy id and topic."
)


def search_university_policies(question, k=3):

    results = search_policies(question, k=k)

    if not results:
        return "No matching policy was found in the knowledge base."

    context = ""

    for document in results:
        context += (
            "["
            + document.metadata["policy_id"]
            + " | "
            + document.metadata["topic"]
            + " | "
            + document.metadata["source"]
            + "]\n"
        )
        context += document.page_content
        context += "\n\n"

    return context.strip()
