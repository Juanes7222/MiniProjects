from torchvision.io import read_image
from torchvision.utils import make_grid, draw_bounding_boxes
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from matplotlib import pyplot as plt
from pathlib import Path
import torchvision.transforms.functional as F
import cv2
import numpy as np
import os
import argparse

plt.rcParams["savefig.bbox"] = 'tight'

def list_dir_images(path):
    files = os.listdir(path)
    types = ["jpg", "png"]
    images = [file for file in files if file.split(".")[-1] in types]
    return images

def prepare_images(images):
    images_list = [read_image(image) for image in images]
    return images_list

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

# grid = make_grid(images_list)
# show(grid)


# persons_with_boxes = [
#     draw_bounding_boxes(image_int, boxes=output['boxes'][output['scores'] > score_threshold], width=8)
#     for image_int, output in zip(images_list, outputs)
# ]
# print(persons)
# show(persons_with_boxes)

def cut_with_aspect(image, aspect_relation, box):
    # Relación de aspecto es una tupla (ancho, alto)
    original_width, original_height = image.shape[1], image.shape[0]
    desired_width, desired_height = aspect_relation

    # Calcular el nuevo tamaño de la imagen manteniendo la relación de aspecto
    if (original_width / original_height) > (desired_width / desired_height):
        new_width = original_height * (desired_width / desired_height)
        new_height = original_height
    else:
        new_width = original_width
        new_height = original_width * (desired_height / desired_width)

    # Calcular la posición de recorte
    left = (original_width - new_width) / 2
    up = (original_height - new_height) / 2
    right = (original_width + new_width) / 2
    down = (original_height + new_height) / 2
    left, up, right, down = validate_size((left, up, right, down), box)
    # Recortar la imagen

    new_image = image[int(up):int(down), int(left):int(right)]

    return new_image

def select_box(persons):
    box = []
    name_index = ["min", "min", "max", "max"]
    for j in range(4):
        value = 0
        if name_index[j] == "min":
            for i in range(4):
                person_value = persons[i][j]
                if value > person_value:
                    value = person_value
        else:
            for i in range(4):
                person_value = persons[i][j]
                if value < person_value:
                    value = person_value
        box.append(value)
    print(box)
    return box

def validate_size(sizes: tuple, box):
    left, up, right, down = sizes
    xmin, ymin, xmax, ymax = box
    if xmin > right or xmax < left or ymin > down or ymax < up:
        # Si la persona está fuera del área de recorte, no se hace nada
        return sizes
    if xmin < left:
        left = xmin - 300
    if xmax > right:
        right = xmax + 300
    if ymin < up:  
        up = ymin - 300
    if ymax > down:
        down = ymax + 300
    sizes = (left, up, right, down)
    return sizes

def prepare_model(weights, progress=False):
    model = fasterrcnn_resnet50_fpn(weights=weights, progress=progress)
    model = model.eval()
    return model

def predict(images, model):

    outputs = model(images)
    
    return outputs

def transform_images(images_list, weights):
    transforms = weights.transforms()

    images = [transforms(d) for d in images_list]
    
    return images

def select_persons(outputs):
    score_threshold = .9
    persons = [output['boxes'][output['scores'] > score_threshold].detach().numpy() for output in outputs]
    
    return persons

def save_image(image, name):
    cv2.imwrite(name, image)
    
def select_relation(image):
    original_width, original_height = image.shape[1], image.shape[0]
    if original_height < original_width:
        return (16, 9)
    return (1, 1)

def create_name(path: Path):
    base_dir = path.parent
    name = path.name
    name = name.replace(".jpg", "_cut.jpg")
    return base_dir/name

def main(path):
    list_dir = list_dir_images(path)
    images_list = prepare_images(list_dir)
    weight = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    transforms_images = transform_images(images_list, weight)
    model = prepare_model(weight)
    outputs = predict(transforms_images, model)
    persons_selected = select_persons(outputs)
    for i, original_image in enumerate(list_dir):
        opened_image = cv2.imread(original_image)
        box = select_box(persons_selected[i])
        new_image = cut_with_aspect(opened_image, box)
        path = Path(original_image)
        name_image = create_name(path)
        save_image(new_image, name_image.as_posix())
        
def check_params():
    parser = argparse.ArgumentParser(description='Revisar parámetros de línea de comandos')
    
    parser.add_argument("-p", "--path", type=str, default="./", required=True, help="Determine the path of images")
    
    args = parser.parse_args()
    main(**vars(args))
    
if "__main__" == __name__:
    # check_params()
    main("./")
        
    