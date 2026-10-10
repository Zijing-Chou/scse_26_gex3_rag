from ollama import chat

from rag_tool import search_university_policies

MODEL = "qwen3:0.6b"

ANSWER_SYSTEM = (
    "You are a university IT support assistant. /no_think\n"
    "Answer the user using only the policy extracts given below.\n"
    "If the extracts do not state the exact fact that the user asked for, "
    "say that this information is not available in the policy documents.\n"
    "Never guess and never invent a number, a phone number or a contact detail.\n"
    "Do not answer with general advice when the requested fact is missing."
)

REVIEW_SYSTEM = (
    "You are a reviewer. /no_think\n"
    "Check the draft answer against the policy extracts.\n"
    "The draft fails if it adds any fact that is not written in the extracts, "
    "for example a phone number that is not there.\n"
    "Reply with exactly one word: PASS or REVISE."
)

REVISE_SYSTEM = (
    "You are a reviser. /no_think\n"
    "Rewrite the draft answer so that it is supported by the policy extracts only. "
    "Remove every invented fact, especially a phone number. "
    "Reply with the corrected answer only."
)

QUESTIONS = [
    "What exact phone number should I call for the University IT Service Desk?",
    "How long are unreleased print jobs kept before they are deleted?",
    "My account is locked after failed sign-in attempts. What should I do?"
]


def call_qwen(system, user):

    response = chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": system
            },
            {
                "role": "user",
                "content": user
            }
        ]
    )

    return response.message.content.strip()


def answer_question(question):

    ## RAG: retrieve the passages that are closest to the question
    context = search_university_policies(question)

    ## Reflection, step 1: first draft
    draft = call_qwen(
        ANSWER_SYSTEM + "\n\nPolicy extracts:\n" + context,
        question
    )

    ## Reflection, step 2: the reviewer decides PASS or REVISE
    review = call_qwen(
        REVIEW_SYSTEM,
        "Question:\n"
        + question
        + "\n\nPolicy extracts:\n"
        + context
        + "\n\nDraft answer:\n"
        + draft
    )

    ## Reflection, step 3: the reviser improves the answer when it was rejected
    if review.upper().startswith("PASS"):
        final = draft
    else:
        final = call_qwen(
            REVISE_SYSTEM,
            "Question:\n"
            + question
            + "\n\nPolicy extracts:\n"
            + context
            + "\n\nDraft answer:\n"
            + draft
            + "\n\nReview:\n"
            + review
        )

    return draft, review, final


for question in QUESTIONS:

    draft, review, final = answer_question(question)

    print("\n====================")
    print("QUESTION:")
    print(question)

    print("\nDRAFT:")
    print(draft)

    print("\nREVIEW:")
    print(review)

    print("\nFINAL ANSWER:")
    print(final)
