# setando imagem base

# caso tenha GPU
#FROM tensorflow/tensorflow:nightly-gpu-jupyter

#caso não tenha GPU
FROM tensorflow/tensorflow:nightly-jupyter

# setando area de trabalho no ambiente
WORKDIR /code

# copiando requisitos para o ambiente
COPY requirements.txt .

# instalando dependencias
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# instalando wget
RUN  apt-get install -y wget

# copiando source para o ambiente
COPY . .

# inicializando jupyter notebook
CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]