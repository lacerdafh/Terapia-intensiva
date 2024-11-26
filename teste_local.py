import os
import streamlit as st
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pathlib import Path


# Suprimir avisos do TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Carregar variáveis de ambiente
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Inicializar cliente Groq
groq_client = ChatGroq(api_key=GROQ_API_KEY)


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
    """
    Configura os diretórios da aplicação na pasta do usuário.
    Retorna tupla com (diretório_base, diretório_documentos, diretório_indices).
    """
    base_dir = os.path.expanduser("~/chatbot_documents")
    docs_dir = os.path.join(base_dir, "documents")
    index_dir = os.path.join(base_dir, "vector_store")
    for directory in [base_dir, docs_dir, index_dir]:
        os.makedirs(directory, exist_ok=True)
    return base_dir, docs_dir, index_dir


def load_documents(folder_path: str) -> list[Document]:
    """Carrega documentos TXT e PDF de uma pasta."""
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
def create_or_load_vector_store(_embeddings, docs_dir: str, index_dir: str):
    """Cria ou carrega o índice FAISS no diretório local."""
    index_path = os.path.join(index_dir, "faiss_index")
    try:
        if os.path.exists(index_path):
            st.info("Carregando índice FAISS existente...")
            return FAISS.load_local(index_path, _embeddings, allow_dangerous_deserialization=True)

        st.info("Criando novo índice FAISS...")
        documents = load_documents(docs_dir)
        if not documents:
            raise RuntimeError("Nenhum documento válido encontrado.")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        vector_store = FAISS.from_documents(texts, _embeddings)

        # Salvar o índice
        os.makedirs(index_path, exist_ok=True)
        vector_store.save_local(index_path)
        return vector_store
    except Exception as e:
        raise RuntimeError(f"Erro ao criar/carregar o vector store: {str(e)}")



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
    
    # Inicializar embeddings
    embeddings = get_embeddings()

    # Sidebar para controles e informações
       
    with st.sidebar:
        image_path = Path(__file__).parent / "static" / "images" / "app_header.png"
        st.image(str(image_path), caption="Dr. Kinho", use_column_width=True)

        # Gerenciamento de arquivos existentes
        st.header("Documentos Disponíveis")
        files = os.listdir(docs_dir)
        
        # Mostrar arquivos existentes em uma tabela
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
                use_container_width=True  # Atualizado de use_column_width para use_container_width
            )
            
            # Seleção de arquivos para deletar
            files_to_delete = st.multiselect(
                "Selecione arquivos para deletar:",
                files
            )
            
            if files_to_delete:
                if st.button("Deletar Selecionados", type="primary"):
                    deleted = []
                    for file in files_to_delete:
                        file_path = os.path.join(docs_dir, file)
                        try:
                            os.remove(file_path)
                            deleted.append(file)
                        except Exception as e:
                            st.error(f"Erro ao deletar {file}: {e}")
                    
                    if deleted:
                        st.success(f"Arquivos deletados: {', '.join(deleted)}")
                        st.cache_resource.clear()
                        st.rerun()
        else:
            st.info("Nenhum documento carregado")
        
        # Botão para recriar índice
        if st.button("Recriar Índice"):
            st.cache_resource.clear()
            st.rerun()

        st.header("Informações")
        st.write(f"📁 Base: {base_dir}")
        st.write(f"📄 Documentos: {docs_dir}")
        st.write(f"📊 Índices: {index_dir}")

        st.header("Documentos Disponíveis")
        files = os.listdir(docs_dir)
        if files:
            for file in files:
                st.write(f"📄 {file}")
        else:
            st.warning("Nenhum documento encontrado")

    try:
        # Inicializar ou carregar vector store
        if 'vector_store' not in st.session_state:
            with st.spinner("Configurando banco de dados..."):
                st.session_state.vector_store = create_or_load_vector_store(embeddings, docs_dir, index_dir)
                st.success("Banco de dados configurado!")

        # Configurar retriever
        retriever = st.session_state.vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 5})

        # Template para o prompt
        template = ChatPromptTemplate.from_template("""
        Responda a pergunta baseando-se apenas no seguinte contexto:
        {context}

        Pergunta: {question}
        """)


        # Configurar modelo Groq
        model = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name="llama-3.2-3b-preview",
            temperature=0.4,
            max_tokens=512
        )

        # Interface principal
        user_question = st.text_input("Faça sua pergunta sobre os documentos:")
        if user_question:
            with st.spinner("Processando..."):
                context = retriever.get_relevant_documents(user_question)
                
                messages = [
                    ("system", "Você é um assistente prestativo que responde perguntas baseado apenas no contexto fornecido."),
                    ("user", f"""
                    Contexto: {' '.join(doc.page_content for doc in context)}
                    
                    Pergunta: {user_question}
                    """)
                ]
                
                response = model.invoke(messages)
                
                with st.container():
                    st.markdown("### Resposta:")
                    # Limpa a resposta antes de mostrar
                    clean_content = response.content
                    st.write(clean_content)
                    
                    st.markdown("#### Fontes consultadas:")
                    sources = set(doc.metadata.get('source', 'Desconhecido') for doc in context)
                    for source in sources:
                        st.write(f"- {os.path.basename(source)}")
                
       
    except Exception as e:
        st.error(f"Erro: {str(e)}")


if __name__ == "__main__":
    main()
