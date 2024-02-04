import os
import shutil
import argparse
import pathlib

def manager(path, destiny, rename=True, isFile=False, removeZip=False):
   path = os.path.abspath(path)
   if isFile:
      validate_path(path, "path", 2)
   else:
      validate_path(path, "path")
      
      
   validate_path(destiny, "destiny")
   files = normalize_path(path, os.listdir(path))
    
   extract_files = get_all_files(files, ".zip")
   extract_files += get_all_files(files, ".rar")
   
   temp = os.path.abspath(f"{path}/temp")
   if not os.path.exists(temp):
      os.makedirs(temp)
   
   for zip_file in extract_files:
      shutil.unpack_archive(zip_file, temp)
      if rename:
         files = normalize_path(temp, os.listdir(temp))
         zip_name = pathlib.Path(zip_file).name.split(".")[0]
         rename_files(files, zip_name)
      
      files = normalize_path(temp, os.listdir(temp))
      move_files(files, destiny)
      shutil.rmtree(temp)
      if removeZip:
         os.remove(zip_file)
   # os.remove(temp)

def normalize_path(root, files):
   paths = map(lambda x: os.path.join(root, x), files)
   return paths
   
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
    
def get_all_files(files: list[str], ext):
   all_files = []
   for file in files:
      if os.path.isdir(file):
         inner_files = map(lambda x: os.path.join(file, x), os.listdir(file))
         all_files += get_all_files(inner_files, ext)
         
      if ext in file:
         all_files.append(file)
      
   return all_files

def rename_files(files, pre):
   for i, file in enumerate(files):
      os.rename(file, f"{pathlib.Path(file).parent}/{pre}_{i}.png")
      
def move_files(files, destiny):
   for file in files:
      shutil.move(file, destiny)
      
def check_params():
   parser = argparse.ArgumentParser(description='Revisar parámetros de línea de comandos')
   
   parser.add_argument("--path", type=str, required=True, help="Determine the zip files path")
   parser.add_argument("--destiny", type=str, required=True, help="Determine the path where the extracted files will be saved")
   parser.add_argument("--isFile", type=bool, default=False, help="Determine if the path is a file or directory")
   parser.add_argument("--rename", type=bool, default=True, help="Determine if the extracted files will be renamed")
   parser.add_argument("--removeZip", type=bool, default=False, help="Determine if the zip file will be removed")
   
   args = parser.parse_args()
   manager(**vars(args))
    
if "__main__" == __name__:
   # manager(r'C:\Users\juanb\OneDrive\Imágenes', r'C:\Users\juanb\OneDrive\Imágenes\Ultimatia')
    check_params()