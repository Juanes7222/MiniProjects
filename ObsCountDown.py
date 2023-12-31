# from obswebsocket import obsws, requests
# import time
# import os
from dotenv import load_dotenv
load_dotenv()

# def actualizar_contador(tiempo_restante):
#     # Actualizar el texto de la fuente en OBS
#     ws.call(requests.SetTextGDIPlusProperties(
#         source=OBS_SOURCE_NAME, text=f"Tiempo restante: {tiempo_restante}", visible=True))

# def iniciar_cronometro(segundos):
#     for segundo_actual in range(segundos, 0, -1):
#         actualizar_contador(segundo_actual)
#         time.sleep(1)

#     # Al llegar a cero, realizar acciones adicionales
#     ws.call(requests.SetTextGDIPlusProperties(
#         source=OBS_SOURCE_NAME, text="¡Tiempo terminado!", visible=True))
#     # Aquí puedes agregar otros eventos o acciones que desees realizar al llegar a cero



# if __name__ == "__main__":
#    host = os.getenv("OBSIP") #"localhost"
#    port = int(os.getenv("OBSPORT"))
#    password = os.getenv("OBSPASSWORD")
#    # print(obsport, obspass)
#    OBS_SOURCE_NAME = "Hola"
#    ws = obsws(host, port, password)
#    ws.connect()
#    tiempo_inicial = 10

#     # Iniciar el cronómetro
#    iniciar_cronometro(tiempo_inicial)
# #    if ws.connected:
# #     print("Conexión exitosa a OBS")
# #    else:
# #      print("Error de conexión a OBS")

# #    root = tk.Tk()
# #    app = CronometroApp(root)
# #    root.mainloop()

#    ws.disconnect()

import sys
import time
import os

import logging
logging.basicConfig(level=logging.INFO)

sys.path.append('../')
from obswebsocket import obsws, requests, events    # noqa: E402


host = "localhost"
port = int(os.getenv("OBSPORT"))
password = os.getenv("OBSPASSWORD")

def on_event(message):
    print("Got message: {}".format(message))


def on_switch(message):
    print("You changed the scene to {}".format(message.getSceneName()))
    


ws = obsws(host, port, password)
ws.register(on_event)
ws.register(on_switch, events.SwitchScenes)
ws.register(on_switch, events.CurrentProgramSceneChanged)
ws.connect()

try:
    print("OK")
    source = ws.call(requests.GetSourceSettings(sourceName="Hola"))
    print(source)
    time.sleep(10)
    print("END")

except KeyboardInterrupt:
    pass

ws.disconnect()

#https://github.com/obsproject/obs-websocket/blob/4.x-compat/docs/generated/protocol.md#setsourcesettings
#https://github.com/yingshaoxo-lab/use-python-to-control-obs/blob/main/main.py