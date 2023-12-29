import tkinter as tk
from obswebsocket import obsws, requests
import time
import os
from dotenv import load_dotenv
load_dotenv()

class CronometroApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Cronómetro")

        self.tiempo_inicial = None
        self.label = tk.Label(master, text="00:00", font=("Helvetica", 48))
        self.label.pack()

        self.btn_iniciar = tk.Button(
            master, text="Iniciar", command=self.iniciar_cronometro)
        self.btn_iniciar.pack()

        self.btn_detener = tk.Button(
            master, text="Detener", command=self.detener_cronometro)
        self.btn_detener.pack()

    def iniciar_cronometro(self):
        self.tiempo_inicial = time.time()
        self.actualizar_cronometro()

    def detener_cronometro(self):
        self.tiempo_inicial = None
        self.label.config(text="00:00")

    def actualizar_cronometro(self):
        if self.tiempo_inicial is not None:
            tiempo_transcurrido = time.time() - self.tiempo_inicial
            minutos, segundos = divmod(tiempo_transcurrido, 60)
            tiempo_formateado = "{:02d}:{:02d}".format(
                int(minutos), int(segundos))
            self.label.config(text=tiempo_formateado)
            self.master.after(1000, self.actualizar_cronometro)

            # Enviar el tiempo transcurrido a OBS como texto para mostrar en una fuente de texto
            ws.call(requests.SetTextGDIPlusProperties(
                "NombreDeTuFuente", text=tiempo_formateado))


if __name__ == "__main__":
   host = os.getenv("OBSIP") #"localhost"
   port = int(os.getenv("OBSPORT"))
   password = os.getenv("OBSPASSWORD")
   # print(obsport, obspass)

   ws = obsws(host, port, password)
   ws.connect()

   root = tk.Tk()
   app = CronometroApp(root)
   root.mainloop()

   ws.disconnect()
