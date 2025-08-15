import os
from flask import Flask, request, jsonify

# LangChain / Ollama / Pinecone
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pinecone import Pinecone, ServerlessSpec

# Config (garanta que estas variáveis existam)
from config import (
    GENERATION_MODEL,
    EMBEDDING_MODEL,
    PINECONE_INDEX_NAME,
    PINECONE_API_KEY,
)

# Flask
app = Flask(__name__)

pinecone_api_key = PINECONE_API_KEY
index_name = PINECONE_INDEX_NAME

pc = Pinecone(api_key=pinecone_api_key)

embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

try:
    _probe_vec = embeddings.embed_query("ping")
    embed_dim = len(_probe_vec)
except Exception as e:
    # fallback (evite travar startup)
    # ajusta se souber a dimensão do seu modelo (ex.: 1024 p/ mxbai-embed-large)
    embed_dim = 1024

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=embed_dim,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )


index = pc.Index(index_name)
vector_store = PineconeVectorStore(
    index_name=index_name,
    embedding=embeddings,
    pinecone_api_key=PINECONE_API_KEY,
)

retriever = vector_store.as_retriever(search_kwargs={"k": 4})
llm = ChatOllama(GENERATION_MODEL, temperature=0)

prompt_template = """
Você é um assistente especializado em responder perguntas com base nos documentos internos da empresa.

Sempre organize a resposta em topicos claros, objetivos e relevantes.  
Se houver mais de uma informacao relevante, organize em lista.  
Use **apenas informações dos documentos fornecidos**, sem inventar ou assumir nada que não esteja explicito.  
Se a informacao solicitada **nao estiver nos documentos**, responda exatamente:  
**"Não há informações disponíveis nos documentos fornecidos para responder a esta pergunta."**
Em caso de **duas redações para a mesma regra**, sendo que uma contém uma nota de alteração normativa ou excluída, considere que a segunda redação (com a nota) e a vigente, e ignore a anterior.

---

### Documentos:
{documents}

### Pergunta:
{question}

### Resposta:

"""

prompt = ChatPromptTemplate.from_template(prompt_template)

chain = prompt | llm | StrOutputParser()


retriever = vector_store.as_retriever()

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True}), 200

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(silent=True) or {}
    pergunta = data.get("pergunta")

    if not pergunta or not isinstance(pergunta, str):
        return jsonify({"error": "Campo 'pergunta' é obrigatório (string)."}), 400

    # Recupera documentos relevantes
    try:
        docs = retriever.invoke(pergunta)
    except Exception as e:
        return jsonify({"error": f"Falha ao recuperar documentos: {str(e)}"}), 500

    if not docs:
        # Garante comportamento do prompt para ausência total de contexto
        textos = ""
    else:
        textos = "\n\n---\n\n".join(d.page_content for d in docs)

    # Gera resposta
    try:
        resposta = chain.invoke({"documents": textos, "question": pergunta})
    except Exception as e:
        return jsonify({"error": f"Falha na geração de resposta: {str(e)}"}), 500

    # Opcional: incluir fontes (metadados) no retorno
    fontes = []
    for d in docs or []:
        fontes.append(d.metadata if hasattr(d, "metadata") else {})

    return jsonify(
        {
            "pergunta": pergunta,
            "resposta": resposta.strip(),
            "fontes": fontes,
            "k": len(docs or []),
        }
    ), 200

    



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)