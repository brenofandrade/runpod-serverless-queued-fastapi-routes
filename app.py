import os
from flask import Flask, request, jsonify

# LangChain / Ollama / FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document
# from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# FAISS nativo
import faiss

# Config (garanta que estas variáveis existam)
from config import (
    GENERATION_MODEL,
    EMBEDDING_MODEL,
)

app = Flask(__name__)

embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

try:
    _probe_vec = embeddings.embed_query("ping")
    embed_dim = len(_probe_vec)
except Exception:
    embed_dim = 1024

FAISS_DIR = os.getenv("FAISS_DIR", "faiss_index")

def load_or_create_faiss():
    try:
        vs = FAISS.load_local(
            FAISS_DIR,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        return vs
    except Exception:
        index = faiss.IndexFlatIP(embed_dim)
        vs = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=InMemoryDocstore({}),
            index_to_docstore_id={},
        )
        return vs

vector_store = load_or_create_faiss()

# -----------------------------
# 10 documentos seed para teste
# -----------------------------
SEED_DOCS = [
    {
        "page_content": (
            "Política de Férias (v2, 2024-10). Colaboradores têm direito a 30 dias anuais. "
            "Solicitações devem ser feitas no portal RH com 30 dias de antecedência. "
            "Regra: períodos podem ser fracionados em até 3 partes (mínimo 5 dias cada)."
        ),
        "metadata": {"source": "RH/politica_ferias_v2.pdf", "section": "Ferias", "version": "2"},
    },
    {
        "page_content": (
            "Processo de Reembolso de Despesas (v3). Itens elegíveis: transporte, hospedagem, alimentação. "
            "Exige notas fiscais em PDF. Prazo de submissão: até 10 dias após a viagem. Aprovação: gestor direto."
        ),
        "metadata": {"source": "Financeiro/reembolso_v3.pdf", "section": "Reembolso", "version": "3"},
    },
    {
        "page_content": (
            "Acesso a Sistemas. Para solicitar acesso a sistemas corporativos, abrir chamado no Service Desk, "
            "indicando gestor aprovador e justificativa. Prazos: até 2 dias úteis para provisionamento."
        ),
        "metadata": {"source": "TI/acessos.md", "section": "Acessos", "version": "1"},
    },
    {
        "page_content": (
            "SLA de Suporte TI (v1). Incidentes críticos: resposta em até 1h e solução em 8h. "
            "Incidentes médios: resposta em 4h e solução em 24h. Solicitações simples: até 3 dias úteis."
        ),
        "metadata": {"source": "TI/sla_suporte.pdf", "section": "Suporte", "version": "1"},
    },
    {
        "page_content": (
            "Política de Segurança da Informação (trecho). Classificação de dados: Público, Interno, Confidencial. "
            "Dados Confidenciais exigem criptografia e acesso restrito. Reporte incidentes via canal oficial."
        ),
        "metadata": {"source": "SegInfo/politica_si.pdf", "section": "Seguranca", "version": "1"},
    },
    {
        "page_content": (
            "Onboarding (Integração). Novos colaboradores devem completar trilha obrigatória de treinamento "
            "em 15 dias. Acessos iniciais são provisionados automaticamente após admissão confirmada."
        ),
        "metadata": {"source": "RH/onboarding.md", "section": "Onboarding", "version": "1"},
    },
    {
        "page_content": (
            "Compras e Aprovação. Pedidos acima de R$ 5.000 requerem aprovação do gerente e do financeiro. "
            "Cotações: mínimo de 3 propostas quando aplicável. Uso de fornecedores homologados é preferencial."
        ),
        "metadata": {"source": "Financeiro/compras.pdf", "section": "Compras", "version": "2"},
    },
    {
        "page_content": (
            "Trabalho Remoto. Permitido até 3 dias por semana mediante acordo com gestor. "
            "Equipamentos devem seguir padrão corporativo e políticas de segurança."
        ),
        "metadata": {"source": "RH/trabalho_remoto.md", "section": "Remoto", "version": "1"},
    },
    {
        "page_content": (
            "Norma de Backup. Backups diários automáticos para diretórios de projeto. "
            "Restauração disponível mediante chamado. Retenção: 30 dias."
        ),
        "metadata": {"source": "TI/backup.pdf", "section": "Backup", "version": "1"},
    },
    {
        "page_content": (
            "Padronização de Documentos. Apresentações devem usar o template oficial. "
            "Nomenclatura: PROJETO-AREA-AAAA-MM-DD-vX.pptx."
        ),
        "metadata": {"source": "Comunicacao/estilo.pptx", "section": "Estilo", "version": "1"},
    },
]

def ensure_seed_data():
    # Evita duplicar se o índice já tiver conteúdo:
    # Heurística simples: se docstore estiver vazio, semeia.
    if hasattr(vector_store, "docstore") and getattr(vector_store.docstore, "dict", {}) == {}:
        docs = [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in SEED_DOCS]
        vector_store.add_documents(docs)
        vector_store.save_local(FAISS_DIR)

ensure_seed_data()

retriever = vector_store.as_retriever(search_kwargs={"k": 4})

llm = ChatOllama(model=GENERATION_MODEL, temperature=0)

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

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True}), 200

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(silent=True) or {}
    print(data)
    pergunta = data.get("pergunta")

    if not pergunta or not isinstance(pergunta, str):
        return jsonify({"error": "Campo 'pergunta' é obrigatório (string)."}), 400

    try:
        docs = retriever.invoke(pergunta)
    except Exception as e:
        return jsonify({"error": f"Falha ao recuperar documentos: {str(e)}"}), 500

    textos = "" if not docs else "\n\n---\n\n".join(d.page_content for d in docs)

    try:
        resposta = chain.invoke({"documents": textos, "question": pergunta})
    except Exception as e:
        return jsonify({"error": F"Falha na geração de resposta: {str(e)}"}), 500

    fontes = [getattr(d, "metadata", {}) or {} for d in (docs or [])]

    return jsonify(
        {"pergunta": pergunta, "resposta": resposta.strip(), "fontes": fontes, "k": len(docs or [])}
    ), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
