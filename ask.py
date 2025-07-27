from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

def ask_query(vectorstore, llm, query):
    # Step 1: Retrieve documents from FAISS
    docs = vectorstore.similarity_search(query, k=3)

    # If nothing found, return fallback message
    if not docs:
        return "❗ No relevant information found."

    # Step 2: Use retrieved content with LLM (if available)
    if llm:  # If you're using a local LLM
        prompt = PromptTemplate(
            input_variables=["context", "input"],
            template="""
Use the context below to answer the question. Be concise and avoid repeating.

Context:
{context}

Question: {input}
Answer:"""
        )
        chain = create_stuff_documents_chain(llm, prompt)
        response = chain.invoke({"input_documents": docs, "input": query})
        answer = response["output_text"].strip()

        if (
            not answer
            or answer.lower().count("the consent") > 3
            or answer.lower().count("the medical procedure") > 3
        ):
            return "❗ Sorry, I couldn't find a reliable answer based on the documents."

        return answer

    # Step 3: If no LLM provided, return raw document snippet
    content = "\n\n".join([doc.page_content for doc in docs])
    return content[:2000]




