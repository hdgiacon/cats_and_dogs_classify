import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from types import FunctionType

src_folder_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(src_folder_path)


def split_data_into_classes(dataset: pd.DataFrame) -> pd.DataFrame | pd.DataFrame:
    '''
    Divide o DataFrame de dados em dois: DataFrame de `gato` e DataFrame de `cachorro`.

    Args:
        * `dataset`: DataFrame com os dados de gato e cachorro misturados.

    Returns:
        `cat_dataframe`, `dog_dataframe`
    '''

    cat_dataframe = dataset.loc[dataset['specie'] == 'Cat']

    dog_dataframe = dataset.loc[dataset['specie'] == 'Dog']

    return cat_dataframe, dog_dataframe


def train_test_valid_sep(dataset: pd.DataFrame) -> pd.DataFrame | pd.DataFrame | pd.DataFrame:
    '''
    Divide o dataset atual em treino, teste e validação respeitando a porporção das raças na coluna `name`.

    Args:
        * `dataset`: base de dados de `gato` ou `cachorro`.

    Returns:
        DataFrame `treino`, DataFrame `teste`, DataFrame `validação`
    '''

    train_df, temp_df = train_test_split(dataset, train_size = 0.7, stratify = dataset['name'])

    valid_df, test_df = train_test_split(temp_df, test_size = 0.3, stratify = temp_df['name'])

    return train_df, test_df, valid_df


def split_data_for_classes(dataset: pd.DataFrame) -> pd.DataFrame | pd.DataFrame | pd.DataFrame | pd.DataFrame | pd.DataFrame | pd.DataFrame:
    '''
    Separa os dados em bases de treino, teste e validação para gato e cachorro separadamente.

    Args:
        * `dataset`: DataFrame com os dados de gato e cachorro misturados.

    Returns:
        `cat_train`, `cat_test`, `cat_valid`, `dog_train`, `dog_test`, `dog_valid`
    '''

    cat_dataframe, dog_dataframe = split_data_into_classes(dataset)

    cat_train, cat_test, cat_valid = train_test_valid_sep(cat_dataframe)

    dog_train, dog_test, dog_valid = train_test_valid_sep(dog_dataframe)

    return cat_train, cat_test, cat_valid, dog_train, dog_test, dog_valid


def split_dataframe(dataset: pd.DataFrame) -> pd.DataFrame | pd.DataFrame | pd.DataFrame:
    '''
    Divide os dados do DataFrame em `treino`, `teste` e `validação`, respeitando as devidas proporções de cada classe para evitar
    desbalanceamento.

    Args:
        * `dataset`: DataFrame com os dados de gato e cachorro misturados.

    Returns:
        `train_df`, `test_df`, `valid_df`
    '''

    cat_train, cat_test, cat_valid, dog_train, dog_test, dog_valid = split_data_for_classes(dataset)

    train_df = pd.concat([cat_train, dog_train]).drop(columns = ['name']).reset_index(drop = True)

    test_df = pd.concat([cat_test, dog_test]).drop(columns = ['name']).reset_index(drop = True)

    valid_df = pd.concat([cat_valid, dog_valid]).drop(columns = ['name']).reset_index(drop = True)

    return train_df, test_df, valid_df


def create_train_generator(train_df: pd.DataFrame, preprocess_input: FunctionType, fig_size: int = 300, batch_size: int = 8):
    '''
    Cria o ImageDataGenerator para os dados de treino.

    Args:
        * `train_df`: DataFrame de treino;
        * `preprocess_input`: método de pré-processamento dos dados. EfficientNet, ResNet e MobileNet_v2 tem o seu próprio `preprocess_input`;
        * `fig_size`: tamanho das imagens. Por padrão é `300`;
        * `batch_size`: número de amostras de dados em cada lote. Por padrão é `8`.

    Args used on Datagen:
        * `rotation_range`: 
        * `width_shift_range`:
        * `height_shift_range`:
        * `shear_range`:
        * `zoom_range`:
        * `horizontal_flip`:
        * `fill_mode`:
        * `preprocessing_function`:

    flow_from_dataframe_Args:
        * `dataframe`:
        * `x_col`:
        * `y_col`:
        * `target_size`:
        * `batch_size`:
        * `class_mode`:

    Return:
        Generator para os dados de treino.
    '''

    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range = 90,
        width_shift_range = 0.2,
        height_shift_range = 0.2,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True,
        fill_mode = "nearest",
        preprocessing_function = preprocess_input
    )

    return train_datagen.flow_from_dataframe(
        dataframe = train_df,
        x_col = 'image_path',
        y_col = 'specie',
        target_size = (fig_size, fig_size),
        batch_size = batch_size,
        class_mode = 'categorical'
    )


def create_generator(df: pd.DataFrame, preprocess_input: FunctionType, fig_size: int = 300, batch_size: int = 8):
    '''
    Cria um ImageDataGenerator para os dados de teste ou validação.

    Args:
        * `df`: DataFrame com dados de teste ou validação;
        * `preprocess_input`: método de pré-processamento dos dados. EfficientNet, ResNet e MobileNet_v2 tem o seu próprio `preprocess_input`;
        * `fig_size`: tamanho das imagens. Por padrão é `300`;
        * `batch_size`: número de amostras de dados em cada lote. Por padrão é `8`.

    Return:
        Generator para os dados presentes em df.
    '''
    
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function = preprocess_input
    )

    return datagen.flow_from_dataframe(
        dataframe = df,
        x_col = 'image_path',
        y_col = 'specie',
        target_size = (fig_size, fig_size),
        batch_size = batch_size,
        class_mode = 'categorical'
    )


def create_test_valid_generators(test_df: pd.DataFrame, valid_df: pd.DataFrame, preprocess_input: FunctionType):
    '''
    Cria um ImageDataGenerator para os dados de teste e um ImageDataGenerator para os dados de validação.

    Args:
        * `test_df`: DataFrame com dados de teste;
        * `valid_df`: DataFrame com dados de validação;
        * `preprocess_input`: método de pré-processamento dos dados. EfficientNet, ResNet e MobileNet_v2 tem o seu próprio `preprocess_input`.

    Returns:
        `test_generator`, `valid_generator`
    '''

    test_generator = create_generator(test_df, preprocess_input)
    valid_generator = create_generator(valid_df, preprocess_input)

    return test_generator, valid_generator