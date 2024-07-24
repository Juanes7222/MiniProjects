import logging
import os
import os.path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

SCOPES = [
          'https://www.googleapis.com/auth/drive', "https://www.googleapis.com/auth/drive.file"]

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

try:
   creds = get_credentials()
   service = build("drive", "v3", credentials=creds)
except HttpError as error:
   logging.error(f"An error occurred: {error}")
   service = None
   
def search_file(q=None, folder=None, mimeTypes="mimeType contains 'image/'"):

  try:
    files = []
    page_token = None
    while True:
        response = (
            service.files()
            .list(
                q=f"{mimeTypes} {f'and {folder} in parents' if folder != None else ""} {f'and {q}' if q != None else ""}",
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
    logging.error(f"An error occurred: {error}")
    files = None

  return files

# files = search_file("not '18EIhV757gPud087Ab34kGgK6eHy7iXQp' in parents and not '16PtMFo_-Hz7IBOfeweMxCv5ouiQ7jWTb' in parents and not '1pLtmpCfX0i7apS8TbrFCyiYdvvr0L3Uj' in parents and not '1DB9lPLyPHN8uNijL70vL4zeENCEZk6HT' in parents and not '11GQoCymdmYmLb2VW-V5Hvpe1p4t9k7f1'  in parents and 'juanblandon975@gmail.com' in owners", None)
files = search_file("'0ADoAQ_oLjlm-Uk9PVA' in parents")
print(files)
    
def move_to(file, folder):
    try:
        file_id = file.get("id")
        old_folder = file.get("parents")[0]
        response = (service.files()
        .update(fileId=file_id, addParents=folder, removeParents=old_folder)
        .execute()
        )
    except HttpError as error:
        print(f"An error occurred: {error}")
        
        
for file in files:
    move_to(file, "1NtnlrX5qkYWDe2O5ytoA6WpDVEqHtbQe")
# move_to("1NyKJlWY4R97hXosRBwMqhSAKPdRmZKVM")