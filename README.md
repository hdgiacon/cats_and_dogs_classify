# Cats and Dogs Classify

Projeto de visão computacional para análise e comparação de modelos convolucionais a fim de classificar imagens em **gato** ou **cachorro** utilizando [Transfer-learning](https://www.tensorflow.org/tutorials/images/transfer_learning?hl=pt-br) e [GRAD-CAM](https://arxiv.org/pdf/1610.02391). Foram testados os modelos [*EfficientNetB3*](https://www.tensorflow.org/api_docs/python/tf/keras/applications/EfficientNetB3), [*MobileNetV2*](https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV2) e [*ResNet50*](https://www.tensorflow.org/api_docs/python/tf/keras/applications/ResNet50), todos utilizando *TensorFlow*.

A seção [Estrutura do Repositório](#estrutura-do-repositório) contém uma descrição e link para cada pasta e arquivo do projeto.

## Exemplos

As imagens passadas aos modelos foram avaliadas por eles e o resultado é mostrado através do gradiente sobreposto às figuras.

### *EfficientNetB3*

![efficient_net_dog](assets/efficient_net/class_dog.jpg)

### *MobileNetV2*

![mobile_net_cat](assets/mobile_net/class_cat_cat.jpg)

### *ResNet50*

![res_net_dog_dog](assets/res_net/dog_dog.jpg)

## Estrutura do Repositório

* [**assets/**](assets/): pasta contendo imagens de saída dos modelos com o gradiente;

* [**data/**](data/): pasta gerada em tempo de execução com a base de dados de imagem;

* [**notebooks/**](notebooks/): pasta contendo os arquivos dos modelos em *Jupyter Notebook*;
  * [*data_analysis.ipynb*](notebooks/data_analysis.ipynb): análise inicial sobre os dados e a sua distribuição;
  * [*efficientNet.ipynb*](notebooks/efficientNet.ipynb): definição, execução e testes do modelo *EfficientNetB3*;
  * [*mobileNet.ipynb*](notebooks/mobileNet.ipynb): definição, execução e testes do modelo *MobileNetV2*;
  * [*resNet.ipynb*](notebooks/resNet.ipynb): definição, execução e testes do modelo *ResNet50*.

* [**output/**](output/): pasta gerada em tempo de execução com os modelos e históricos de treino salvos;
  * [**models/**](output/models/): pasta contendo os modelos treinado salvos;
  * [**history/**](output/history/): pasta contendo os históricos de treinamento salvos.

* [**utils/**](utils/): pasta contendo implementações em *Python* comuns à todos os modelos;
  * [*extract_data.py*](utils/extract_data.py): extração dos dados e criação do *DataFrame*;
  * [*plot_heatmap.py*](utils/plot_heatmap.py): plotagem de gráficos e imagens para análise dos resultados;
  * [*split_data.py*](utils/split_data.py): divisão dos dados em treino, teste e validação.

* [*.gitignore*](.gitignore): arquivo de instrução do *Git* informando quais arquivos e pastas devem ser ignorados;

* [*Dockerfile*](Dockerfile): arquivo de instrução para a criação da imagem *Docker* para o projeto;

* [*LICENSE*](LICENSE): arquivo informando qual a licença de software vigente no projeto;

* [*README.md*](_): arquivo de ajuda e instrução;

* [*requirements.txt*](requirements.txt): arquivo de instrução contendo as bibliotecas necessárias para o projeto.
