import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

src_folder_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(src_folder_path)


def get_images_path(path: str) -> list[str]:
    '''
    Lista todos os arquivos presentes na pasta especificada e seleciona todos os arquivos `.jpg`.

    Args:
        * `path`: caminho até a pasta com as imagens.

    Return:
        Lista com o caminho de cada imagem presente em path.
    '''

    images_path_list = os.listdir(path)

    images_path_list = [path + image_path for image_path in images_path_list if image_path.endswith('.jpg')]

    return images_path_list


def get_images_name(images_path_list: list[str]) -> list[str]:
    '''
    A partir de uma lista de caminhos de imagem, seleciona apenas o nome de cada imagem.

    Args:
        * `images_path_list`: lista com o caminho de cada imagem da base de dados.

    Return:
        Lista contendo o nome de todas as imagens da base de dados.
    '''

    images_path = [image_path.split('/')[-1] for image_path in images_path_list]

    images_name_list = [image_name[:image_name.rfind('_')] for image_name in images_path]

    return images_name_list


def defining_identifiers_for_classes(images_name_list: list[str]) -> list[str]:
    '''
    A partir de uma lista de nomes das imagens, define a qual classe pertence, respeitando a seguinte regra:
        * Se a primeira letra for `maiúscula` -> `cat`
        * caso contrário -> `dog`

    Args:
        * `images_name_list`: lista com os nomes das imagens.

    Return:
        Lista com o identificador de classe de cada imagem.
    '''

    return ['Cat' if image_name[0].isupper() else 'Dog' for image_name in images_name_list]


def create_dataframe(path: str) -> pd.DataFrame:
    '''
    Cria um DataFrame a partir da lista de `paths`, `nome` das imagens e lista de `classes` (espécie).

    Args:
        * `path`: caminho até a pasta com as imagens.

    Return:
        Pandas DataFrame com as 3 colunas juntas.
    '''

    images_path_list = get_images_path(path)
    images_name_list = get_images_name(images_path_list)
    species_id_list = defining_identifiers_for_classes(images_name_list)

    return pd.DataFrame({'image_path': images_path_list, 'name': images_name_list, 'specie': species_id_list})


def plot_samples(dataset: pd.DataFrame) -> None:
    '''
    A partir do DataFrame gerado, mostra algumas amostras de imagens graficamente.

    Args:
        * `dataset`: base de dados já processada como um DataFrame.
    '''

    image_list = dataset.tail(4)['image_path'].values
    names_list = dataset.tail(4)['name'].values

    rows = 1
    columns = 4

    fig = plt.figure(figsize = (16, 8))

    for i in range(1, columns * rows + 1):
        fig.add_subplot(rows, columns, i)

        pil_im = Image.open(image_list[i-1])
        im_array = np.asarray(pil_im)

        plt.imshow(im_array)

        plt.title(names_list[i-1])

    plt.show()