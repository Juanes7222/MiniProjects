from ebooklib import epub
from PIL import Image
from io import BytesIO
import os
import ebooklib
import argparse

def manager(path, destiny, isFile=False, start=1):
    path = os.path.abspath(path)
    if isFile:
        validate_path(path, "path", 2)
    else:
        validate_path(path, "path")
        
    validate_path(destiny, "destiny")
    
    files = map(lambda x: os.path.join(path, x), os.listdir(path))
    
    epubs = get_epubs(files)
    for book in epubs:
        images = extract_images(book)
        start = save_images(images, destiny, start)

def save_images(images, destiny, start=1):
    for i, image in enumerate(images, start=start):
        path = os.path.join(destiny, f"{i}.{image.format.lower()}")
        image.save(path)
    return i+1

def extract_images(epub_path: str):
    images = []
    book = epub.read_epub(epub_path)
    for image in book.get_items_of_type(ebooklib.ITEM_IMAGE):
        images.append(Image.open((BytesIO(image.get_content()))))
    return images
    
    
def validate_path(path, name, __case=1):
    if not os.path.exists(path):
        raise ValueError("No such file or directory")
    if __case == 1:
        if os.path.isfile(path):
            raise ValueError(f"The {name} most by a directory, not file")
    return True
    
def get_epubs(files: list[str]):
    epub_files = []
    for file in files:
        if os.path.isdir(file):
            inner_files = map(lambda x: os.path.join(file, x), os.listdir(file))
            epub_files += get_epubs(inner_files)
            
        if ".epub" in file:
            epub_files.append(file)
        
    return epub_files

def check_params():
    parser = argparse.ArgumentParser(description='Revisar parámetros de línea de comandos')
    
    parser.add_argument("--path", type=str, required=True, help="Determine the epub books path")
    parser.add_argument("--destiny", type=str, required=True, help="Determine the path where the images will be saved")
    parser.add_argument("--isFile", type=bool, default=False, help="Determine if the path is a file or directory")
    parser.add_argument("--start", type=int, required=False, default=1, help="Determine the number start of the enumerate")
    
    args = parser.parse_args()
    manager(**vars(args))
    
if "__main__" == __name__:
    check_params()