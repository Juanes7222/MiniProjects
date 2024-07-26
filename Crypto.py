from cryptography.fernet import Fernet

# Generar una clave y guardarla en un archivo
def generate_password():
    password = Fernet.generate_key()
    with open("sources/password.key", "wb") as file_password:
        file_password.write(password)

# Cargar la clave
def load_password():
    return open("sources/password.key", "rb").read()

def create_Fernet():
    return Fernet(load_password())

def crypt_data(data):
    f = create_Fernet()
    crypt_data = f.encrypt(data)
    return crypt_data

def save_crypt(crypt_data, file_name):
    with open(file_name + ".enc", "wb") as crypt_file:
        crypt_file.write(crypt_data)

# Encriptar el archivo
def encrypt_file(file_name):
    
    with open(file_name, "rb") as file:
        data_file = file.read()
    
    crypt_data = crypt_data(data_file)
    save_crypt(crypt_data, file_name)

def decrypt_file(crypt_file_name):
    f = create_Fernet()
    
    with open(crypt_file_name, "rb") as crypt_file:
        crypt_data = crypt_file.read()
    
    decrypt_data = f.decrypt(crypt_data)
    return decrypt_data
