# test_youtube_service.py
import unittest
from unittest.mock import patch
from sender import search_youtube_video
from constants import YOUTUBE_API_KEY, YOUTUBE_CHANNEL_ID

class TestYouTubeService(unittest.TestCase):

    # El string dentro de patch debe apuntar a donde se USA requests, no de donde proviene.
    @patch('sender.requests.get')
    def test_buscar_video_youtube_exitosa(self, mock_get):
        # 1. Configurar el Mock (la respuesta simulada)
        mock_response = mock_get.return_value
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "items": [
                {"snippet": {"title": "Sermón de prueba"}}
            ]
        }

        # 2. Ejecutar la función
        title = "CUANDO SE OBEDECE AL MENSAJE"
        preacher = "Rev Javier Carrascal"
        resultado = search_youtube_video(title, preacher, 90)
        print("Resultado de búsqueda:", resultado)  # Debug: Ver el resultado retornado por la función   

        # 3. Afirmaciones (Asserts)
        # Verificar que la función retorna la data mockeada
        self.assertEqual(resultado["items"][0]["snippet"]["title"], "Sermón de prueba")

        # Verificar que requests.get fue llamado con los parámetros exactos
        mock_get.assert_called_once_with(
            "https://www.googleapis.com/youtube/v3/search",
            params={
                "key":        YOUTUBE_API_KEY,
                "channelId":  YOUTUBE_CHANNEL_ID,
                "q":          "La fe Juan Perez",
                "part":       "snippet",
                "type":       "video",
                "maxResults": 5,
            },
            timeout=10,
        )

if __name__ == '__main__':
    unittest.main()