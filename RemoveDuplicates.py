import os
import cv2
import numpy as np
from argparse import ArgumentParser

def manager(path, remove=True, show_images=False):
    path = os.path.abspath(path)
    image_paths = os.listdir(path)
    image_paths = map(lambda x: os.path.join(path, x), image_paths)
    # image_paths = map(lambda x: fr"{x}", image_paths)
    hashes = get_hashes(image_paths)
    remove_images(hashes, remove, show_images)
    
def remove_images(hashes, remove, show_images):
    # Iteramos sobre los hashes.
    for hash, hashed_paths in hashes.items():
        # Si el hash en cuestión está asociado a más de una ruta, entonces tenemos una imagen repetida.
        if len(hashed_paths) > 1:
            # Si no vamos a remover los duplicados, entonces construiremos un montaje para mostrarlos.
            if not remove:

                # Construimos un mosaico de todos los duplicados.
                if show_images:
                    montage = None
                    for p in hashed_paths:
                        # Leemos y redimensionamos la imagen.
                        image = cv2.imread(p)
                        image = cv2.resize(image, (150, 150))

                        if montage is None:
                            montage = image
                        else:
                            montage = np.hstack([montage, image])

                # Imprimimos información sobre los duplicados y mostramos el montaje en pantalla.
                show_data(hash, hashed_paths, show_images)
            else:
                # Removemos todos los duplicados salvo el primero.
                for p in hashed_paths[1:]:
                    os.remove(p)
                    
def show_data(hash, hashed_paths, montage=False):
    print(f'Hash: {hash}')
    print(f'# duplicados: {len(hashed_paths)}')
    print('---')
    if montage:
        cv2.imshow('Montaje', montage)
        cv2.waitKey(0)
    

def get_hashes(images):
    hashes = {}
    for image_path in images:
        image = cv2.imread(image_path)
        if image is None: continue
        hash = dhash(image)
        
        p = hashes.get(hash, [])
        p.append(image_path)
        hashes[hash] = p
    return hashes
    
def dhash(image, hash_size=8):
    """
    Calcula el dhash de la imagen de entrada.
    :param image: Imagen a la cuaal le calcularemos el dhash.
    :param hash_size: Número de bytes en el hash resultante.
    """
    # Resdimensionamos la imagen con base al tamaño del hash.
    resized = cv2.resize(image, (hash_size + 1, hash_size))

    # Generamos la imagen de diferencias de píxeles adyacentes.
    diff = resized[:, 1:] > resized[:, :-1]

    # Calculamos el hash.
    return sum([2 ** i for i, v in enumerate(diff.flatten()) if v])

def def_args():
    argument_parser = ArgumentParser()
    argument_parser.add_argument('-p', '--path', required=True , help='Path to images')
    argument_parser.add_argument('-r', '--remove', type=bool, default=True,
                                help='Determine if remove the duplicates images')
    argument_parser.add_argument('-s', '--show_images', required=False, type=bool, default=False, help="Determine if show images")
    arguments = vars(argument_parser.parse_args())
    manager(**arguments)
    
if "__main__" == __name__:
    # manager(r'C:\Users\juanb\OneDrive\Imágenes\NGNL', show_images=True)
    def_args()