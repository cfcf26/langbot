import os
from dotenv import load_dotenv
import streamlit as st
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

# 환경 변수 로드
load_dotenv()

# 경로 설정
root = 'langchain/libs/'
community = root + 'community/langchain_community'
core = root + 'core/langchain_core'
experimental = root + 'experimental/langchain_experimental'
langchain = root + 'langchain/langchain'
partners = root + 'partners'
text_splitter = root + 'text_splitter/langchain_text_splitter'
cookbook = 'langchain/cookbook'

# 문서 로드 함수
@st.cache
def load_documents(paths, suffixes):
    documents = []
    for path in paths:
        loader = GenericLoader.from_filesystem(
            path=path,
            glob="**/*",
            suffixes=suffixes,
            parser=LanguageParser(language=Language.PYTHON, parser_threshold=30),
        )
        documents.extend(loader.load())
    return documents

# MDX 문서 로드 함수
@st.cache
def load_mdx_documents(root_dir):
    documents = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for file in filenames:
            if file.endswith(".mdx") and "*venv/" not in dirpath:
                try:
                    loader = TextLoader(os.path.join(dirpath, file), encoding="utf-8")
                    documents.extend(loader.load())
                except Exception:
                    pass
    return documents

# 문서 분할 함수
@st.cache
def split_documents(documents, language, chunk_size=2000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter.from_language(
        language=language, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    ) if language else RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)

# 메인 로직
def main():
    paths = [community, core, experimental, langchain, partners, text_splitter, cookbook]
    py_documents = load_documents(paths, [".py"])
    mdx_documents = load_mdx_documents("langchain")

    py_docs = split_documents(py_documents, Language.PYTHON)
    mdx_docs = split_documents(mdx_documents, None)

    combined_documents = mdx_docs + py_docs

    embeddings = OpenAIEmbeddings(model='text-embedding-3-large')
    index_name = os.getenv("PINECONE_INDEX_NAME")
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)


    bm25_retriever = BM25Retriever.from_documents(combined_documents)
    bm25_retriever.k = 10

    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 10})
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, retriever],
        weights=[0.6, 0.4],
        search_type="mmr",
    )

    prompt_template = """You are a 20-year AI developer. Your task is to answer the given question using the information in the documentation to the best of your ability. Be sure to answer in Korean.
    The documentation contains information about Python code, so please include detailed code snippets of the Python code when writing your answer.
    Please be as detailed as possible and answer in Korean. If you can't find the answer in the given documentation, please write "The documentation does not answer the question."
    Be sure to cite the source of your answer.

    Translated with DeepL.com (free version)


    #references:
    {context}

    #question:
    {question}

    #answer: 

    #source:
    - source1
    - source2
    - ...                             
    """

    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
    )

    rag_chain = (
        {"context": ensemble_retriever, "question": RunnablePassthrough()}
        | PromptTemplate.from_template(prompt_template)
        | llm
        | StrOutputParser()
    )
    # Streamlit 앱의 메인 함수
    st.title("LangChain 문서 검색")

    # 사용자 입력 받기
    user_query = st.text_input("검색어를 입력하세요:", "")

    if user_query:  # 사용자가 검색어를 입력한 경우
    # 세션 상태에서 'combined_documents'를 사용하여 검색을 수행합니다.
        response = rag_chain.invoke(user_query)
        st.write("검색 결과:")
        st.markdown(response)  # 검색 결과를 화면에 출력합니다.

if __name__ == "__main__":
    main()

