import os
import sys
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from IPython.display import Image as display_image
from IPython.display import display
from types import FunctionType

src_folder_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(src_folder_path)


def plot_accuracy_loss(history) -> None:
    '''
    Plota dois gráficos: `acurácia` do treino e validação, e `taxa de aprendizagem` do treino e validação.

    Args:
        * `history`: histórico de treinamento do modelo.
    '''

    acc = history.history['acc']
    val_acc = history.history['val_acc']

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    max_v = max(history.epoch) + 1
    epochs_range = range(max_v)

    plt.figure(figsize = (8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label = 'Training accuracy')
    plt.plot(epochs_range, val_acc, label = 'Validation accuracy')
    plt.legend(loc = 'lower right')
    plt.title('Training and Validation accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label = 'Training Loss')
    plt.plot(epochs_range, val_loss, label = 'Validation Loss')
    plt.legend(loc = 'upper right')
    plt.title('Training and Validation Loss')
    plt.show()


def select_preprocess_input_method(preprocess_input_method: FunctionType, image: np.ndarray) -> np.ndarray:
    '''
    Mediante ao `preprocess_input_method`, seleciona o preprocess_input para `resnet`, `efficientnet` ou `mobilenet_v2`.

    Args:
        * `preprocess_input_method`: tipo de pré-processamento de entrada mediante ao modelo escolhido;
            * tf.keras.applications.resnet.preprocess_input
            * tf.keras.applications.efficientnet.preprocess_input
            * tf.keras.applications.mobilenet_v2.preprocess_input

        * `image`: imagem que será passada para preprocess_input.

    Return:
        Imagem aplicada ao preprocess_input escolhido.
    '''

    if preprocess_input_method is tf.keras.applications.resnet.preprocess_input:
        return tf.keras.applications.resnet.preprocess_input(image)

    elif preprocess_input_method is tf.keras.applications.efficientnet.preprocess_input:
        return tf.keras.applications.efficientnet.preprocess_input(image)

    else:
        return tf.keras.applications.mobilenet_v2.preprocess_input(image)


def process_sample(image: Image, fig_size: int, preprocess_input_method: FunctionType) -> np.ndarray:
    '''
    Processa uma imagem de amostra para ser retornada como um array Numpy.

    Args:
        * `image`: imagem carregada em memória utilizando a biblioteca PIL;
        * `fig_size`: tamanho da imagem;
        * `preprocess_input_method`: tipo de pré-processamento de entrada mediante ao modelo escolhido.

    Return:
        Imagem processada e retornada como um array Numpy.
    '''

    if image.mode != "RGB":
        image = image.convert("RGB")

    image = image.resize((fig_size, fig_size))

    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis = 0)

    image = select_preprocess_input_method(preprocess_input_method, image)

    return image


def predict_sample(
    array_sample: np.ndarray, 
    train_generator, 
    model
) -> str | np.ndarray:
    '''
    Aplica uma imagem de amostra sobre o modelo treinado e realiza a predição.

    Args:
        * `array_sample`: imagem de amostra como um Numpy array;
        * `train_generator`: ImageGenerator de treino;
        * `model`: modelo atual já treinado.

    Returns:
        Valor predito `(str)`, Numpy array com os valores preditos para as `duas classes`
    '''

    ypred = model.predict(array_sample)

    train_generator.class_indices

    dict_classes = {v: k for k,v in train_generator.class_indices.items()}

    return dict_classes[np.argmax(ypred)], ypred


def get_last_conv_layer_name(model) -> str:
    '''
    Busca qual o nome da ultima camada convolucional do modelo atual.

    Args:
        * `model`: modelo atual já treinado.

    Return:
        Nome da ultima camada convolucional.
    '''

    for layer in model.layers[-10:]:
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name


def make_gradcam_heatmap(
    img_array: np.ndarray, 
    model, 
    last_conv_layer_name: str, 
    pred_index: int = None
) -> np.ndarray:
    '''
    Cria o `gradiente` como um mapa de calor (heatmap) a partir da ultima camada convolucional de um determinado modelo. O gradiente
    consiste no `mapa de ativação` das features na ultima camada, ou seja, os locais onde o modelo de classificação detectou as 
    características.

    Args:
        * `img_array`: imagem de teste como um array Numpy;
        * `model`: modelo atual já treinado;
        * `last_conv_layer_name`: nome da ultima camada convolucional. Utilizar função `get_last_conv_layer_name`;
        * `pred_index`: indice da probabilidade de uma determinada classe.

    Return:
        Heatmap como um array Numpy.
    '''

    # Primeiro, criamos um modelo que mapeia a imagem de entrada para as ativações
    # da última camada de convolução, bem como as previsões de saída.
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )


    # Em seguida, calculamos o gradiente da classe predita (com maior probabildiade)
    # para nossa imagem de entrada com relação às ativações da última camada de convolução
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # Gradiente do output do neurônio de saida em relação ao mapa de feature da ultima
    #camada de convolução
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # Vetor para cada entrada é a média da intensidade do gradiente
    # em relação ao mapa de feature
    pooled_grads = tf.reduce_mean(grads, axis = (0, 1, 2))


    # Multiplicamos cada canal na matriz do mapa de features
    # pelo gradiente (importância do canal) em relação à classe predita com maior probabilidade
    # então soma-se os canais para obter a ativação da classe do mapa de calor
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalizando entre 0 e 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    return heatmap.numpy()


def display_gradcam(img_path: str, heatmap: np.ndarray, cam_path: str = "cam.jpg", alpha: float = 0.4) -> None:
    '''
    Mostra a imagem usada como teste no modelo treinado com o mapa de calor sobreposto a ela.

    Args:
        * `img_path`: caminho (path) da imagem de teste;
        * `heatmap`: mapa de calor retornado pela função `make_gradcam_heatmap`;
        * `cam_path`: caminho (path) para a imagem de saída;
        * `alpha`: valor da trasnparência do mapa de calor sobre a imagem testada.
    '''

    # imagem original
    img = tf.keras.preprocessing.image.load_img(img_path)
    img = tf.keras.preprocessing.image.img_to_array(img)

    # rescalando a imagem entre até 255
    heatmap = np.uint8(255 * heatmap)

    # jet para colorizar o "gradiente"
    jet = cm.get_cmap("jet")

    # usando os valores RGB do heatmap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Criando uma imagem com RGB colorido com o heatmap
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    # Sobreposição das imagens
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

    # Salvando a imagem para realizar o display
    superimposed_img.save(cam_path)

    # Display
    display(display_image(cam_path))
    # Excluí a imagem do caminho salvo
    os.remove(cam_path)