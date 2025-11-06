import argparse
import os

def rename_files(path: str, names):
    if os.path.isfile(path):
        raise ValueError("The path must be a directory, not file")
    files: list[str] = os.listdir(os.path.abspath(path))
    
    if names is None:
        names = range(1, len(files), 1)
    else:
        if len(names) != len(files):
            raise ValueError("The lenght of names must be equal that files")
        
    files.sort(key=lambda x: int(os.path.splitext(x)[0]))
    files = map(lambda x: os.path.join(path, x), files)
    
    for file, name in zip(files, names):
        file_type = os.path.splitext(file)[1]
        new_name = os.path.join(os.path.dirname(file), f"{name}{file_type}")
        os.rename(file, new_name)
        
def check_params():
    parser = argparse.ArgumentParser(description='Revisar parámetros de línea de comandos')
    
    parser.add_argument("--path", type=str, required=True, help="Determine the files path")
    parser.add_argument("--names", type=list, required=False, help="Determine the new names for the files")
    
    args = parser.parse_args()
    rename_files(**vars(args))
    
if "__main__" == __name__:
    check_params()