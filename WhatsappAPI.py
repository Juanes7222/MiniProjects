from requests import request
import os

class WP:
    def __init__(self) -> None:
        self.token = os.getenv("TOKEN")
        self.id = os.getenv("WP_ID")

    def send_message(self, number):
        url = 'https://graph.facebook.com/v19.0/324109830776301/messages'
        message = {
            "messaging_product": "whatsapp",
            "preview_url": False,
            "recipient_type": "individual",
            "to": number,
            "type": "template",
            "template": {
                "name": "hello_world",
                "language": {
                    "code": "en_US"
                },
            }
        }
        
        headers = {
          "Autorization": f"Bearer {self.token}",
          "Conttent-Type": "aplication/json"
        }
        return self.__post(url=url, json=message, headers=headers)