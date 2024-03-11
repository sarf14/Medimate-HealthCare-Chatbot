import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain import PromptTemplate

DB_FAISS_PATH = 'vectorstores/db_faiss'

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])
    return prompt

st.cache(allow_output_mutation=True)
def load_model_and_data():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = CTransformers(model="TheBloke/Llama-2-7B-Chat-GGML", model_type="llama", max_new_tokens=512, temperature=0.5)
    qa_prompt = set_custom_prompt()
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=db.as_retriever(search_kwargs={'k': 2}),
                                     return_source_documents=True, chain_type_kwargs={'prompt': qa_prompt})
    return qa

def qa_bot(qa, query):
    # Implement the question-answering logic here
    response = qa({'query': query})
    return response['result']

def main():
    # This should be the first Streamlit command in your script
    st.set_page_config(page_title="Llama-2-GGML Medical Chatbot")

    st.title("Llama-2-GGML Medical Chatbot")
    
    # Load the model and data
    qa = load_model_and_data()

    query = st.text_input("Ask your question here:")

    if st.button("Get Answer"):
        if query:
            with st.spinner("Processing your question..."):
                # Call your QA function
                answer = qa_bot(qa,query)
                st.write(f"Answer: {answer}")
        else:
            st.warning("Please input a question.")

if __name__ == "__main__":
    main()
