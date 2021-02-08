import warnings
warnings.filterwarnings('ignore')

# importar os pacotes necessários
import os
import argparse
from imutils import paths
import progressbar
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import numpy as np
import h5py
import cv2

class HDF5Dataset(object):
    """Escreve o conjunto de dados fornecidos como input (em formato numpy array)
    para um conjunto de dados HDF5.

    :param dims_data: dimensões dos dados a serem armazenados no dataset
    :param output_path: arquivo onde o arquivo hd5f será salvo
    :param data_key: nome do dataset que armazenará os dados
    :param buffer_size: buffer de memória
    """
    def __init__(self, dims_data, dims_label, output_path, data_key='images', buffer_size=500):

        # # verifica se o arquivo já existe, evitando seu overwrite.
        if os.path.isfile(output_path):
            raise ValueError("O arquivo '{}' já existe e não pode "
                             "ser apagado.".format(output_path))

        # abrir um database HDF5 e criar 02 datasets:
        #   1. Para armazenar as imagens; e
        #   2. Para armazenar os labels.
        self.db = h5py.File(output_path, 'w')
        self.data = self.db.create_dataset(data_key, dims_data, dtype='float', compression="gzip", compression_opts=9)
        self.labels = self.db.create_dataset('labels', dims_label, dtype='int', compression="gzip", compression_opts=9)

        # Definir o buffer e índice da próxima linha disponível
        self.buffer_size = buffer_size
        self.buffer = {"data": [], "labels": []}
        self.idx = 0
        
    def store_class_labels(self, class_labels):
        """
        Cria um dataset para armazenar as classes dos labels.

        :param class_labels: lista com todos nomes das classes dos labels
        """
        dt = h5py.special_dtype(vlen=str)
        label_set = self.db.create_dataset("label_names",
                                           shape=class_labels.shape,
                                           dtype=dt)
        label_set[:] = class_labels
    
    def add(self, rows, labels):
        """
        Adiciona as linhas e labels ao buffer.

        :param rows: linhas a serem adicionadas ao dataset
        :param labels: labels correspondentes às linhas adicionadas
        """
        self.buffer["data"].extend(rows)
        self.buffer["labels"].extend(labels)

        # verificar se o buffer precisa ser limpo (escrever
        # os dados informados no disco)
        if len(self.buffer["data"]) >= self.buffer_size: #&gt
            self.flush()
    
    def flush(self):
        """Reseta o buffer e escreve os dados no disco."""
        i = self.idx + len(self.buffer["data"])
        self.data[self.idx:i] = self.buffer["data"]
        self.labels[self.idx:i] = self.buffer["labels"]
        self.idx = i
        self.buffer = {"data": [], "labels": []}
    
    def close(self):
        """
        Fecha o dataset após verificar se ainda há algum dado
        remanescente no buffer.
        """
        # if len(self.buffer["data"]) == 0: #&gt
        if len(self.buffer["data"]) > 0: #&gt
            self.flush()

        # fechar o dataset
        self.db.close()


# argumentos de entrada do script
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True,
                help='caminho do dataset')
ap.add_argument('-o', '--output', required=True, default="./tempo.hdf5",
                help='caminho para salvar o arquivo HDF5')
ap.add_argument('-s', '--buffer-size', type=int, default=500,
                help='buffer para escrever no arquivo HDF5')
ap.add_argument('-r', '--resize', type=int, default=64,
                help='dimensoes para redimensionar as imagens')
args = vars(ap.parse_args())

# armazenar o buffer size
bs = args["buffer_size"]
rs = args["resize"]

# importar os nomes dos arquivos das imagens
print("[INFO] carregando imagens...")
imagePaths = list(paths.list_images(args["dataset"]))

# extrair e codificar os labels das classes de cada imagem do dataset
labels = [imagePath.split(os.path.sep)[-2] for imagePath in imagePaths]
# print (labels)
le = LabelEncoder()
labels = le.fit_transform(labels)
# perform one-hot encoding on the labels
# lb = LabelBinarizer()
# labels = lb.fit_transform(labels)

dims_data = (len(imagePaths), rs, rs, 3)
dims_label = labels.shape
# iniciar o HDF5 e armazenar os nomes dos labels
dataset = HDF5Dataset(dims_data, dims_label, args["output"], buffer_size=bs)
dataset.store_class_labels(le.classes_)
print (le.classes_)
# dataset.store_class_labels(lb.classes_)

# Barra de progresso para acopanhar
widgets = ["[imagens] -&gt; [hdf5]: ", progressbar.Percentage(), " ",
           progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(imagePaths),
                               widgets=widgets).start()

#  processar e importar as imagens e labels
for (i, (path, label)) in enumerate(zip(imagePaths, labels)):
    # image = cv2.imread(path)
    # image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_LANCZOS4) / 255.0
    
    # load the input image (dm_resize x dm_resize) and preprocess it
    image = load_img(path, target_size=(rs, rs), interpolation='lanczos')
    image = img_to_array(image)
    image = np.array(image, dtype="float32") / 255.0

    dataset.add([image], [label])
    print (i, path, label)
    # dataset.add(image, label)

    pbar.update(i)

# finalizar e fechar o arquivo HDF5
pbar.finish()
dataset.close()


# db = h5py.File("./output/train_max_compress.hdf5", "r")

# print (db["imagens"][0])
# print (db["imagens"].shape)

# print (db["labels"][0])
# print (db["labels"].shape)

# carlos$ python data_hdf5.py --dataset dataset/train/ --output ./train_dataset.hdf5
# [INFO] carregando imagens…
# [imagens] -&gt; [hdf5]: 100% |#####################################| Time: 0:02:19

"""
Para abrir o arquivo e acessar suas fotos, é só importar o pacote h5py e abrir o arquivo em formato de leitura usando
db = h5py.File(“./nome_do_arquivo.hdf5”, “r”)

Para acessar a primeira foto do dataset, é só usar 
db[“imagens”][0]

Para ver o shape do dataset com as images, é só usar
db[“imagens”].shape

E é isso mesmo, exatamente como manipular um array do NumPy.

python data_hdf5.py -d ../../databases/Soja_splited/test/ -o ./output/test_64.hdf5 -r 64; python data_hdf5.py -d ../../databases/Soja_splited/train/ -o ./output/train_128.hdf5 -r 128; python data_hdf5.py -d ../../databases/Soja_splited/val/ -o ./output/val_128.hdf5 -r 128; python data_hdf5.py -d ../../databases/Soja_splited/test/ -o ./output/test_128.hdf5 -r 128; python data_hdf5.py -d ../../databases/Soja_splited/train/ -o ./output/train_224.hdf5 -r 224 -s 250; python data_hdf5.py -d ../../databases/Soja_splited/val/ -o ./output/val_224.hdf5 -r 224 -s 250; python data_hdf5.py -d ../../databases/Soja_splited/test/ -o ./output/test_224.hdf5 -r 224 -s 250; python data_hdf5.py -d ../../databases/Soja_splited/train/ -o ./output/train_256.hdf5 -r 256 -s 200; python data_hdf5.py -d ../../databases/Soja_splited/val/ -o ./output/val_256.hdf5 -r 256 -s 200; python data_hdf5.py -d ../../databases/Soja_splited/test/ -o ./output/test_256.hdf5 -r 256 -s 200
"""
