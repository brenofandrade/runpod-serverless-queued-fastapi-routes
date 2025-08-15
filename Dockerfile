# Use uma imagem Python enxuta
FROM python:3.9-slim

# Defina o diretório de trabalho
WORKDIR /app

# Copie os arquivos do projeto para dentro do contêiner
COPY . /app

# Instale dependências do sistema necessárias para o curl e o Ollama
RUN apt-get update && apt-get install -y curl

# Instale o Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Instale as dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# Baixe os modelos do Ollama
RUN ollama pull llama3.2:latest \
 && ollama pull mxbai-embed-large

# Exponha a porta da aplicação
EXPOSE 8080 11434

# Comando para iniciar o servidor do Ollama em background e depois rodar seu script
CMD ["sh", "-c", "ollama serve & sleep 5 && python app/main.py"]
