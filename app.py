import os
import streamlit as st
from langchain_chroma import Chroma
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.schema import Document
from langchain.embeddings.huggingface import HuggingFaceInferenceAPIEmbeddings
from pydantic import SecretStr
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from pathlib import Path
import tomllib
import shutil
import sys
from snowflake.connector import connect
from langchain.vectorstores import SnowflakeVectorStore
from snowflake.connector import connect

# Suprimir avisos de depreciação
import warnings
warnings.filterwarnings('ignore')

# Suprimir avisos do TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


with open("config.toml", "rb") as f:
    config = tomllib.load(f)

os.environ["GROQ_API_KEY"] = config["api_keys"]["groq_api_key"]
os.environ["HF_API_KEY"] = config["api_keys"]["hf_api_key"]

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# Carregar variáveis de ambiente
load_dotenv()
snowflake_config = {
    "account": st.secrets["snowflake"]["account"],
    "user": st.secrets["snowflake"]["user"],
    "password": st.secrets["snowflake"]["password"],
    "warehouse": st.secrets["snowflake"]["warehouse"],
    "database": st.secrets["snowflake"]["database"],
    "schema": st.secrets["snowflake"]["schema"]
}
# Verificar a chave API
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY não encontrada no arquivo .env")

# Inicializar o modelo Groq
chat_model = ChatGroq(
    model_name="llama-3.2-3b-preview",
    temperature=0.7,
    max_tokens=512
)

@st.cache_resource
def get_embeddings():
    """Inicializa e retorna o modelo de embeddings usando HuggingFace Inference API."""
    try:
        # Obtenha a chave de API do Hugging Face do arquivo .env
        hf_api_key = os.getenv("HF_API_KEY")
        if not hf_api_key:
            raise ValueError("A chave da API do Hugging Face (HF_API_KEY) não foi encontrada no .env")

        return HuggingFaceInferenceAPIEmbeddings(
            api_key=SecretStr(hf_api_key),  # Chave da API
            model_name="sentence-transformers/all-MiniLM-L6-v2",  # Modelo a ser utilizado
        )
    except Exception as e:
        st.error(f"Erro ao carregar embeddings da HuggingFace API: {str(e)}")
        raise

def get_app_directories() -> tuple[str, str, str]:
    """Configura os diretórios da aplicação na pasta do usuário."""
    base_dir = os.path.expanduser("~/chatbot_documents")
    docs_dir = os.path.join(base_dir, "documents")
    index_dir = os.path.join(base_dir, "vector_store")
    
    for directory in [base_dir, docs_dir, index_dir]:
        os.makedirs(directory, exist_ok=True)
        
    return base_dir, docs_dir, index_dir

# Configurar SQLite

def load_documents(folder_path: str) -> list[Document]:
    """Carrega documentos TXT e PDF de uma pasta."""
    documents = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        try:
            if file_name.lower().endswith(".txt"):
                # Leitura manual para arquivos de texto
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()
                    documents.append(Document(
                        page_content=text,
                        metadata={"source": file_path}
                    ))
            elif file_name.lower().endswith(".pdf"):
                pdf_reader = PdfReader(file_path)
                text = "\n".join(page.extract_text() for page in pdf_reader.pages)
                documents.append(Document(
                    page_content=text,
                    metadata={"source": file_path}
                ))
        except Exception as e:
            st.warning(f"Erro ao carregar {file_name}: {str(e)}")
    return documents

@st.cache_resource
def create_or_load_vector_store(_embeddings, docs_dir: str, index_dir: str):
    """Cria ou carrega o índice Chroma no diretório local."""
    persist_directory = os.path.join(index_dir, "chroma_db")

    try:
        if os.path.exists(persist_directory) and os.listdir(persist_directory):
            st.info("Carregando índice Chroma existente...")
            return Chroma(
                persist_directory=persist_directory,
                embedding_function=_embeddings
            )

        st.info("Criando novo índice Chroma...")
        documents = load_documents(docs_dir)
        if not documents:
            raise RuntimeError("Nenhum documento válido encontrado.")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)

        vector_store = SnowflakeVectorStore(
          connection=init_snowflake_connection(),
             embeddings=embeddings,
             table_name="EMBEDDINGS_TABLE",
               dimension=1536  # dimensão para OpenAI embeddings
                 )

        vector_store.persist()
        return vector_store

    except Exception as e:
        raise RuntimeError(f"Erro ao criar/carregar o índice Chroma: {str(e)}")

def upload_files(uploaded_files, docs_dir: str) -> list[str]:
    """Salva múltiplos arquivos enviados no diretório de documentos."""
    saved_files = []
    for uploaded_file in uploaded_files:
        try:
            file_path = os.path.join(docs_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            saved_files.append(uploaded_file.name)
        except Exception as e:
            st.error(f"Erro ao salvar arquivo {uploaded_file.name}: {e}")
    return saved_files

def main():
    st.title("Chatbot com Dr. Kinho")

    # Configurar diretórios
    base_dir, docs_dir, index_dir = get_app_directories()

    # Obter embeddings
    embeddings = get_embeddings()

    with st.sidebar:
        image_path = Path(__file__).parent / "static" / "images" / "app_header.png"
        st.image(str(image_path), caption="Dr. Kinho", use_container_width=True)
        st.header("Gerenciamento de Documentos")

        # Upload de documentos
        uploaded_files = st.file_uploader(
            "Envie documentos (TXT ou PDF)",
            type=["txt", "pdf"],
            accept_multiple_files=True
        )

        if uploaded_files:
            with st.spinner("Salvando arquivos..."):
                saved_files = upload_files(uploaded_files, docs_dir)
                if saved_files:
                    st.success(f"Arquivos salvos: {', '.join(saved_files)}")
                    st.cache_resource.clear()
                    st.rerun()

        # Listar documentos existentes
        st.header("Documentos Disponíveis")
        files = os.listdir(docs_dir)

        if files:
            file_data = [
                {
                    "Arquivo": file,
                    "Tamanho (KB)": f"{os.path.getsize(os.path.join(docs_dir, file)) / 1024:.1f}"
                }
                for file in files
            ]

            st.dataframe(
                file_data,
                hide_index=True,
                use_container_width=True
            )

            # Opção para deletar arquivos
            files_to_delete = st.multiselect("Selecione arquivos para deletar:", files)

            if files_to_delete and st.button("Deletar Selecionados"):
                with st.spinner("Deletando arquivos..."):
                    deleted_files = []
                    for file in files_to_delete:
                        try:
                            os.remove(os.path.join(docs_dir, file))
                            deleted_files.append(file)
                        except Exception as e:
                            st.error(f"Erro ao deletar {file}: {e}")
                    if deleted_files:
                        st.success(f"Arquivos deletados: {', '.join(deleted_files)}")
                        st.cache_resource.clear()
                        st.rerun()
        else:
            st.info("Nenhum documento carregado.")

        # Recriar índice vetorial
        if st.button("Recriar Banco de Dados"):
            with st.spinner("Recriando índice vetorial..."):
                try:
                    st.cache_resource.clear()
                    st.session_state.vector_store = create_or_load_vector_store(
                        _embeddings=embeddings,
                        docs_dir=docs_dir,
                        index_dir=index_dir
                    )
                    st.success("Banco de dados recriado com sucesso!")
                except Exception as e:
                    st.error(f"Erro ao recriar banco de dados: {e}")
                st.rerun()
            st.header("Informações")
            st.write(f"📁 Base: {base_dir}")
            st.write(f"📄 Documentos: {docs_dir}")
            st.write(f"📊 Índices: {index_dir}")

    # Configurar banco de dados (Chroma)
    try:
        if 'vector_store' not in st.session_state:
            with st.spinner("Configurando banco de dados..."):
                st.session_state.vector_store = create_or_load_vector_store(
                    _embeddings=embeddings,
                    docs_dir=docs_dir,
                    index_dir=index_dir
                )
                st.success("Banco de dados configurado!")

        # Configurar o retriever para busca
        retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 5})

        # Entrada de perguntas
        user_question = st.text_input("Faça sua pergunta sobre os documentos:")
        if user_question:
            with st.spinner("Processando..."):
                # Recuperar os documentos relevantes
                context = retriever.get_relevant_documents(user_question)

                # Configurar o modelo de chat (Groq ou similar)
                chat_model = ChatGroq(
                    api_key=GROQ_API_KEY,
                    model_name="llama-3.2-3b-preview",
                    temperature=0.4,
                    max_tokens=512
                )

                # Montar mensagens
                messages = [
                    ("system", "Você é um assistente que responde com base no contexto fornecido."),
                    ("user", f"""
                    Contexto: {' '.join(doc.page_content for doc in context)}
                    Pergunta: {user_question}
                    """)
                ]

                # Obter resposta do modelo
                response = chat_model.invoke(messages)

                # Exibir resposta e fontes
                with st.container():
                    st.markdown("### Resposta:")
                    st.write(response.content)

                    st.markdown("#### Fontes consultadas:")
                    sources = set(doc.metadata.get('source', 'Desconhecido') for doc in context)
                    for source in sources:
                        st.write(f"- {os.path.basename(source)}")

    except Exception as e:
        st.error(f"Erro: {str(e)}")


if __name__ == "__main__":
    main()
