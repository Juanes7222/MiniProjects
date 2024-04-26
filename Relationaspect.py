from PIL import Image
from torchvision import models, transforms
from matplotlib import pyplot as plt
import cv2
import numpy as np
import torch

# Cargar el modelo preentrenado
model = models.detection.fasterrcnn_resnet50_fpn(weights=models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
model.eval() # Cambiar el modelo a modo de evaluación

def preprocesar_imagen(imagen_path):
    # Definir las transformaciones
    transform = transforms.Compose([
        transforms.Resize((800, 800)), # Cambiar el tamaño de la imagen
        transforms.ToTensor(), # Convertir la imagen a tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Normalizar
    ])
    
    # Cargar y preprocesar la imagen
    imagen = Image.open(imagen_path).convert("RGB")
    imagen_tensor = transform(imagen).unsqueeze(0) # Añadir una dimensión extra para el batch
    
    return imagen_tensor

# Uso de la función
imagen_tensor = preprocesar_imagen('./DSC_0471.jpg')

def detectar_personas(imagen_tensor, modelo):
    # Realizar la detección
    detecciones = modelo(imagen_tensor)
    
    # Extraer las cajas de detección y las etiquetas
    cajas = detecciones[0]['boxes']
    etiquetas = detecciones[0]['labels']
    
    # Filtrar solo las detecciones de personas
    personas = [caja for caja, etiqueta in zip(cajas, etiquetas) if etiqueta == 10] # Asumiendo que la etiqueta 0 es "persona"
    
    return personas

# Uso de la función
# personas = detectar_personas(imagen_tensor, model)


# def visualizar_detecciones(imagen_path, personas):
#     imagen = Image.open(imagen_path)
#     fig, ax = plt.subplots(1)
#     ax.imshow(imagen)
    
#     for caja in personas:
#          np_box = caja.detach().numpy()
#          x1, y1, x2, y2 = np_box
#          rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color='red')
#          ax.add_patch(rect)
    
#     plt.show()

def visualizar_detecciones(imagen_path, detecciones):
    # Cargar la imagen
    imagen = Image.open(imagen_path)
    fig, ax = plt.subplots(1)
    ax.imshow(imagen)
    
    # Extraer las cajas de detección y las etiquetas
    cajas = detecciones[0]['boxes'].detach().numpy()
    etiquetas = detecciones[0]['labels'].detach().numpy()
    
    # Definir un diccionario para mapear las etiquetas a nombres de clases
    # Este paso es opcional y depende de cómo estén etiquetadas tus clases
    # En este ejemplo, asumimos que las etiquetas son índices numéricos
    # que corresponden a clases predefinidas.
    # Ajusta este diccionario según tus necesidades.
    
    for caja, etiqueta in zip(cajas, etiquetas):
        x1, y1, x2, y2 = caja
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color='red')
        ax.add_patch(rect)
        
        # Añadir la etiqueta de la clase en la esquina superior izquierda de la caja
        ax.text(x1, y1, etiqueta, fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
    plt.show()

# Uso de la función
visualizar_detecciones('./DSC_0471.jpg', model(imagen_tensor))


def change_relation(image: Image, relation: str):
   new_width, new_height = relation.split(":")
   width, height = image.size
   
   factor_scale = min(width/new_width, height/new_height)
   
   width_cut = width/factor_scale
   height_cut = height/factor_scale
   
   