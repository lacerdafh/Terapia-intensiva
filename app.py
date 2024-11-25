import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma  # Substituindo FAISS por Chroma
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pathlib import Path

# Suprimir avisos do TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Carregar variáveis de ambiente
load_dotenv()

# Verificar a chave API
if 'GROQ_API_KEY' in st.secrets:
    GROQ_API_KEY = st.secrets['GROQ_API_KEY']
else:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY não encontrada!")
    st.stop()

@st.cache_resource
def get_embeddings():
    """Inicializa e retorna o modelo de embeddings."""
    try:
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    except Exception as e:
        st.error(f"Erro ao carregar modelo de embeddings: {str(e)}")
        raise

def get_app_directories() -> tuple[str, str, str]:
    """Configura os diretórios da aplicação."""
    # Usar diretórios temporários para o Streamlit Cloud
    base_dir = os.path.join(os.getcwd(), "temp_data")
    docs_dir = os.path.join(base_dir, "documents")
    db_dir = os.path.join(base_dir, "db")
    
    for directory in [base_dir, docs_dir, db_dir]:
        os.makedirs(directory, exist_ok=True)
        
    return base_dir, docs_dir, db_dir

def load_documents(folder_path: str) -> list[Document]:
    """Carrega documentos TXT e PDF."""
    documents = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        try:
            if file_name.lower().endswith(".txt"):
                loader = TextLoader(file_path)
                documents.extend(loader.load())
            elif file_name.lower().endswith(".pdf"):
                pdf_reader = PdfReader(file_path)
                text = "\n".join(page.extract_text() for page in pdf_reader.pages)
                documents.append(Document(page_content=text, metadata={"source": file_path}))
        except Exception as e:
            st.warning(f"Erro ao carregar {file_name}: {str(e)}")
    return documents

@st.cache_resource
def create_or_load_vector_store(_docs_dir: str, _db_dir: str, embeddings):
    """Cria ou carrega o vector store usando Chroma."""
    try:
        # Criar nova instância do Chroma
        documents = load_documents(_docs_dir)
        if not documents:
            raise RuntimeError("Nenhum documento válido encontrado.")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        texts = text_splitter.split_documents(documents)
        
        # Usar Chroma em vez de FAISS
        vector_store = Chroma.from_documents(
            documents=texts,
            embedding=embeddings,
            persist_directory=_db_dir
        )
        
        return vector_store
    except Exception as e:
        raise RuntimeError(f"Erro ao criar/carregar o vector store: {str(e)}")

def clean_response(response_content: str) -> str:
    """Limpa a resposta removendo metadados."""
    try:
        content_lines = []
        skip_section = False
        
        for line in response_content.split('\n'):
            if any(meta in line for meta in [
                "additional_kwargs",
                "response_metadata",
                "type",
                "name",
                "id",
                "example",
                "tool_calls",
                "invalid_tool_calls",
                "usage_metadata"
            ]):
                skip_section = True
                continue
                
            if not line.strip():
                skip_section = False
                continue
                
            if not skip_section:
                content_lines.append(line)
        
        return '\n'.join(content_lines).strip()
    except Exception:
        return response_content

def main():
    st.title("Chatbot com Documentos Locais")
    
    base_dir, docs_dir, db_dir = get_app_directories()
    embeddings = get_embeddings()
    
    with st.sidebar:
        st.header("Controles")
        
        uploaded_files = st.file_uploader(
            "Enviar documentos",
            type=['txt', 'pdf'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            with st.spinner("Salvando arquivos..."):
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(docs_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                st.success("Arquivos salvos com sucesso!")
                st.cache_resource.clear()
                st.rerun()
        
        st.header("Documentos Disponíveis")
        files = os.listdir(docs_dir)
        
        if files:
            file_data = []
            for file in files:
                file_path = os.path.join(docs_dir, file)
                size_kb = os.path.getsize(file_path) / 1024
                file_data.append({
                    "Arquivo": file,
                    "Tamanho (KB)": f"{size_kb:.1f}"
                })
            
            st.dataframe(
                file_data,
                column_config={
                    "Arquivo": "Arquivo",
                    "Tamanho (KB)": st.column_config.NumberColumn(
                        "Tamanho (KB)",
                        format="%.1f KB"
                    )
                },
                hide_index=True,
                use_container_width=True
            )
            
            files_to_delete = st.multiselect(
                "Selecione arquivos para deletar:",
                files
            )
            
            if files_to_delete and st.button("Deletar Selecionados", type="primary"):
                for file in files_to_delete:
                    try:
                        os.remove(os.path.join(docs_dir, file))
                    except Exception as e:
                        st.error(f"Erro ao deletar {file}: {e}")
                st.success("Arquivos deletados com sucesso!")
                st.cache_resource.clear()
                st.rerun()
        else:
            st.info("Nenhum documento carregado")
        
        if st.button("Recriar Índice"):
            st.cache_resource.clear()
            st.rerun()

    try:
        if 'vector_store' not in st.session_state:
            with st.spinner("Configurando banco de dados..."):
                st.session_state.vector_store = create_or_load_vector_store(
                    docs_dir, db_dir, embeddings
                )
                st.success("Banco de dados configurado!")

        chat_model = ChatGroq(
            temperature=0.4,
            max_tokens=1024,
            model_name="mixtral-8x7b-32768",
        )

        user_question = st.text_input("Faça sua pergunta sobre os documentos:")
        if user_question:
            with st.spinner("Processando..."):
                retriever = st.session_state.vector_store.as_retriever(
                    search_kwargs={"k": 5}
                )
                context = retriever.get_relevant_documents(user_question)
                
                messages = [
                    ("system", "Você é um assistente prestativo que responde perguntas baseado apenas no contexto fornecido."),
                    ("user", f"""
                    Contexto: {' '.join(doc.page_content for doc in context)}
                    
                    Pergunta: {user_question}
                    """)
                ]
                
                response = chat_model.invoke(messages)
                
                with st.container():
                    st.markdown("### Resposta:")
                    clean_content = clean_response(response.content)
                    st.write(clean_content)
                    
                    st.markdown("#### Fontes consultadas:")
                    sources = set(doc.metadata.get('source', 'Desconhecido') for doc in context)
                    for source in sources:
                        st.write(f"- {os.path.basename(source)}")

    except Exception as e:
        st.error(f"Erro: {str(e)}")

if __name__ == "__main__":
    main()