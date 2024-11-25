# Chatbot com Documentos Locais

Este é um chatbot construído com Streamlit que permite fazer perguntas sobre documentos locais usando LangChain, FAISS e Groq.

## Requisitos

- Python 3.8+
- Ollama instalado e rodando localmente
- Chave API da Groq

## Instalação

1. Clone o repositório:
```bash
git clone https://github.com/lacerdafh/Terapia-intensiva.git
cd seu-repositorio
```

2. Crie um ambiente virtual e ative-o:
```bash
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

4. Configure o arquivo .env:
```
GROQ_API_KEY=sua_chave_api_aqui
DOCUMENTS_FOLDER=./data
FAISS_INDEX_PATH=./vector_store
```

## Uso

1. Coloque seus documentos (PDF ou TXT) na pasta `data/`

2. Execute a aplicação:
```bash
streamlit run app.py
```

3. Acesse http://localhost:8501 no seu navegador

## Estrutura do Projeto

```
projeto/
├── .env                  # Variáveis de ambiente (não versionado)
├── .gitignore           # Arquivos ignorados pelo Git
├── README.md            # Este arquivo
├── app.py               # Código principal
├── requirements.txt     # Dependências
├── data/                # Pasta para documentos (não versionada)
└── vector_store/        # Pasta para índice FAISS (não versionada)
```

## Funcionalidades

- Processamento de documentos PDF e TXT
- Indexação vetorial com FAISS
- Interface amigável com Streamlit
- Busca semântica em documentos
- Respostas geradas via API da Groq

## Contribuindo

1. Faça um Fork do projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request