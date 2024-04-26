from ebooklib import epub
from PIL import Image
from io import BytesIO
import os
import ebooklib
import argparse
import fitz # PyMuPDF

def manager(path, destiny, isFile=False, start=1):
    path = os.path.abspath(path)
    if isFile:
        validate_path(path, "path", 2)
    else:
        validate_path(path, "path")
        
    validate_path(destiny, "destiny")
    
    files = map(lambda x: os.path.join(path, x), os.listdir(path))
    
    extract_files = get_all_files(files)
    
    for book in extract_files:
        type_file = os.path.splitext(book)[1][1:]
        images = eval(f"{type_file}_extract_images(book)")
        start = save_images(images, destiny, start)
        
def save_images(images, destiny, start=1):
    for i, image in enumerate(images, start=start):
        path = os.path.join(destiny, f"{i}.{image.format.lower()}")
        image.save(path)
        del image
    return i+1
        
def pdf_extract_images(pdf):
    images = []
    doc = fitz.open(pdf)  # open document
    for page in doc:  # iterate through the pages
        image_list = page.get_images(full=True)
        for img in image_list:
            img_base = doc.extract_image(img[0])
            img_bytes = img_base["image"]
            img_bytes_object = BytesIO(img_bytes)
            if validate_bytes(img_bytes_object.tell()):
                object_image = Image.open((img_bytes_object))
                if validate_size(object_image.size):
                    images.append(object_image)
    return images

def epub_extract_images(epub_path: str):
    images = []
    book = epub.read_epub(epub_path)
    for image in book.get_items_of_type(ebooklib.ITEM_IMAGE):
        image_bytes_object = BytesIO(image.get_content())
        if validate_bytes(image_bytes_object):
            object_image = Image.open((image_bytes_object))
            if validate_size(object_image.size):
                images.append(object_image)
    return images

def validate_bytes(size):
    if size <= 100000:
        return False
    return True

def validate_size(size: tuple):
    if size[1] <= 300:
        return False
    return True
    
def validate_path(path, name, __case=1):
    if not os.path.exists(path):
        if __case==1:
            os.makedirs(path)
        else:
            raise ValueError("No such file or directory")
    if __case == 1:
        if os.path.isfile(path):
            raise ValueError(f"The {name} most be a directory, not file")
    return True
    
def get_all_files(files: list[str]):
    all_files = []
    for file in files:
        if os.path.isdir(file):
            inner_files = map(lambda x: os.path.join(file, x), os.listdir(file))
            all_files += get_all_files(inner_files)
            
        if ".pdf" in file or ".epub" in file:
            all_files.append(file)
        
    return all_files

def check_params():
    parser = argparse.ArgumentParser(description='Revisar parámetros de línea de comandos')
    
    parser.add_argument("-p", "--path", type=str, required=True, help="Determine the epub books path")
    parser.add_argument("-d", "--destiny", type=str, required=True, help="Determine the path where the images will be saved")
    parser.add_argument("-f", "--isFile", type=bool, default=False, help="Determine if the path is a file or directory")
    parser.add_argument("-s", "--start", type=int, required=False, default=1, help="Determine the number start of the enumerate")
    
    args = parser.parse_args()
    manager(**vars(args))
    
if "__main__" == __name__:
    check_params()