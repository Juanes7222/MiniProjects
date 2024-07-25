import argparse
import os
import os.path
from tqdm import tqdm
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

SCOPES = [
          'https://www.googleapis.com/auth/drive', "https://www.googleapis.com/auth/drive.file"]

service = None
types = {
   "all": None,
   "images": "'image/'",
   "videos": "'video/'",
   "pdf": "'application/pdf'",
   "audio": "'audio/'",
   "document": "application/"
}

def get_credentials():
   creds = None
   token = "sources/token.json"
   if os.path.exists(token):
      creds = Credentials.from_authorized_user_file(token, SCOPES)
   if not creds or not creds.valid:
      if creds and creds.expired and creds.refresh_token:
         creds.refresh(Request())
      else:
         flow = InstalledAppFlow.from_client_secrets_file(
            "sources/client_secret.json", SCOPES
         )
         creds = flow.run_local_server(port=8000)
      with open("sources/token.json", "w") as token:
         token.write(creds.to_json())
   return creds
   
def search_file(q=None, folder=None, mimeType="mimeType contains 'image/'", **kwargs):

  try:
    files = []
    page_token = None
    qwery = f"{mimeType if mimeType != None else ""} {f'and "{folder}" in parents' if folder != None else ""} {f'and {q}' if q != None else ""}"
    while True:
        response = (
            service.files()
            .list(
                q=qwery,
                spaces="drive",
                fields="nextPageToken, files(id, name, parents)",
                pageToken=page_token,
                
            )
            .execute()
        )
        
        files.extend(response.get("files", []))
        page_token = response.get("nextPageToken", None)
        if page_token is None:
            break

  except HttpError as error:
    print(f"An error occurred: {error}")
    files = None

  return files
    
def move_to(file, folder):
    try:
        file_id = file.get("id")
        old_folder = file.get("parents")[0]
        response = (service.files()
        .update(fileId=file_id, addParents=folder, removeParents=old_folder)
        .execute()
        )
        return response
    except HttpError as error:
        print(f"An error occurred: {error}")

def move_to_all(files, destiny):
   for file in tqdm(files):
      move_to(file, destiny)

def create_service():
   try:
      global service
      creds = get_credentials()
      service = build("drive", "v3", credentials=creds)
   except HttpError as error:
      print(f"An error occurred: {error}")
      service = None
      
def manager(**params):
   file_type = types.get(params.get("mimeType"))
   if not file_type:
      raise TypeError("File type is not avalaible")
   params["mimeType"] = f"mimeType contains {file_type}"
   create_service()
   files = search_file(**params)
   move_to_all(files, params["new_folder"])
      
def check_params():
    parser = argparse.ArgumentParser(description='Revisar parámetros de línea de comandos')
    
    parser.add_argument("--folder", "-F", type=str, required=False, default=None, help="Determine the folder that contain the files")
    parser.add_argument("--new_folder", "-N", type=str, required=True, help="Determine the new folder for the files")
    parser.add_argument("--mimeType", "-T", type=str, default="images", required=False, help="Determine the mimetype of files (write ''all' if you want to move all type of files)")
    parser.add_argument("--qwery", "-Q", type=str, required=False, default=None, help="Specific qwery")
    
    args = parser.parse_args()
    manager(**vars(args))
    
check_params()
   
