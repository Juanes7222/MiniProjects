import subprocess
import os
import json
import math
import time
import logging
import multiprocessing
import argparse
import sys
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    # Dummy tqdm if not installed
    def tqdm(iterable, **kwargs):
        return iterable

DEFAULT_CONFIG = {
    "INPUT_DIR": "./",
    "OUTPUT_DIR": "audios_whatsapp",
    "MAX_MB": 16,
    "WATCH_MODE": False,
    "WATCH_INTERVAL": 5,
    "ENABLE_PARALLEL": False,
    "CPU_THREADS": multiprocessing.cpu_count(),
    "OUTPUT_FORMAT": "m4a", 
}

LOG_FILE = "audio_processor.log"
PROCESSED_FILE = "processed_files.json"

# Extensiones soportadas
SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".aac", ".wma", ".m4b"}

# Límites de bitrate para voz
MIN_BITRATE = 32
MAX_BITRATE = 128

# Configuración de audio
NORMALIZE_AUDIO = True
TARGET_LUFS = -16
VOICE_FILTERS = True
HIGHPASS_FREQ = 80       
LOWPASS_FREQ = 12000     
COMPRESSION_RATIO = 3    
NOISE_REDUCTION = True   

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

def load_processed_files():
    """Carga el registro de archivos ya procesados"""
    if os.path.exists(PROCESSED_FILE):
        try:
            with open(PROCESSED_FILE, 'r') as f:
                return json.load(f)
        except:
            logging.warning("No se pudo cargar el registro de archivos procesados")
    return {}

def save_processed_files(processed):
    """Guarda el registro de archivos procesados"""
    try:
        with open(PROCESSED_FILE, 'w') as f:
            json.dump(processed, f, indent=2)
    except Exception as e:
        logging.error(f"Error al guardar registro: {e}")

def get_file_hash(filepath):
    """Calcula hash rápido del archivo"""
    try:
        stat = os.stat(filepath)
        return f"{stat.st_size}_{stat.st_mtime}"
    except:
        return None

def check_ffmpeg():
    """Verifica que FFmpeg y FFprobe estén instalados"""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        subprocess.run(["ffprobe", "-version"], capture_output=True, check=True)
        return True
    except:
        return False

def get_duration_seconds(audio_path):
    """Obtiene la duración del audio en segundos"""
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "json", audio_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        return float(data["format"]["duration"])
    except Exception as e:
        logging.error(f"Error al obtener duración de {os.path.basename(audio_path)}: {e}")
        return None

def calculate_bitrate(duration_sec, target_mb, channels=2, safety_margin=0.90):
    """Calcula el bitrate óptimo"""
    max_bits = target_mb * 8 * 1024 * 1024
    bitrate = (max_bits / duration_sec) * safety_margin
    bitrate_kbps = math.floor(bitrate / 1000)

    # Ajustar límites según canales
    max_limit = MAX_BITRATE if channels == 2 else 96
    min_limit = MIN_BITRATE if channels == 2 else 24 # Permitir bitrate más bajo en mono

    return max(min_limit, min(bitrate_kbps, max_limit))

def build_audio_filters():
    """Construye la cadena de filtros de audio"""
    filters = []
    
    if VOICE_FILTERS:
        if NOISE_REDUCTION:
            filters.append("afftdn=nf=-25")
        filters.append(f"highpass=f={HIGHPASS_FREQ}")
        filters.append(f"lowpass=f={LOWPASS_FREQ}")
        filters.append(f"acompressor=threshold=-20dB:ratio={COMPRESSION_RATIO}:attack=5:release=50:makeup=2")
        filters.append("deesser")
    
    if NORMALIZE_AUDIO:
        filters.append(f"loudnorm=I={TARGET_LUFS}:TP=-1.5:LRA=11")
    
    # Limitador final
    filters.append("alimiter=limit=0.95:attack=5:release=50")
    
    return ",".join(filters) if filters else None

def get_optimal_sample_rate(bitrate_kbps):
    """Calcula la frecuencia de muestreo óptima según el bitrate"""
    # A bajos bitrates, reducir la frecuencia de muestreo mejora la fidelidad
    # en las frecuencias que sí se escuchan, evitando el sonido "metálico".
    if bitrate_kbps < 32:
        return "22050"
    elif bitrate_kbps < 48:
        return "32000"
    elif bitrate_kbps < 96:
        return "44100"
    else:
        return "48000"


def process_single_file(args):
    """
    Procesa un archivo individual.
    Esta función está diseñada para ser 'pure' en cuanto a escritura de JSON para soportar multithreading.
    Retorna un diccionario con el resultado.
    """
    input_path, config, file_hash, ffmpeg_threads = args
    input_filename = os.path.basename(input_path)
    
    result = {
        "file": input_filename,
        "hash": file_hash,
        "success": False,
        "message": "",
        "skipped": False
    }

    try:
        # Obtener duración
        duration = get_duration_seconds(input_path)
        if duration is None:
            result["message"] = "No se pudo obtener duración"
            return result

        # Configuración inicial
        use_stereo = duration < 3300 # < 55 min
        channels = 2 if use_stereo else 1
        suffix = "_wp_stereo" if use_stereo else "_wp_mono"
        
        base_name = os.path.splitext(input_filename)[0]
        # Determinar extensión y codec según configuración
        out_fmt = config.get("OUTPUT_FORMAT", "m4a")
        suffix = f"_wp_stereo.{out_fmt}" if use_stereo else f"_wp_mono.{out_fmt}"
        output_file = f"{base_name}{suffix}"
        output_path = os.path.join(config["OUTPUT_DIR"], output_file)
        
        audio_filters = build_audio_filters()

        # Bucle de intentos para ajustar tamaño
        current_bitrate = calculate_bitrate(duration, config["MAX_MB"], channels)
        attempts = 0
        max_attempts = 3
        
        while attempts < max_attempts:
            attempts += 1
            
            # Ajustar sample rate dinámicamente para maximizar calidad a bajo bitrate
            sample_rate = get_optimal_sample_rate(current_bitrate)
            
            cmd = [
                "ffmpeg", "-y",
                "-threads", str(ffmpeg_threads),
                "-i", input_path,
                "-vn",
                "-ac", str(channels),
                "-ar", sample_rate,
            ]
            
            # Selector de codec
            if out_fmt == "m4a":
                cmd.extend(["-c:a", "aac", "-movflags", "+faststart"])
            elif out_fmt == "mp3":
                cmd.extend(["-c:a", "libmp3lame"])
            elif out_fmt == "ogg":
                cmd.extend(["-c:a", "libopus", "-vbr", "on"])

            if audio_filters:
                cmd.extend([
                    "-filter_threads", str(ffmpeg_threads),
                    "-filter:a", audio_filters
                ])

            cmd.extend([
                "-b:a", f"{current_bitrate}k",
                "-map_metadata", "-1",
                output_path
            ])

            # Ejecutar FFmpeg
            process = subprocess.run(cmd, capture_output=True, text=True)

            if process.returncode != 0:
                result["message"] = f"Error FFmpeg: {process.stderr[:200]}"
                return result

            # Verificar tamaño
            if not os.path.exists(output_path):
                result["message"] = "Archivo de salida no creado"
                return result

            output_size_mb = os.path.getsize(output_path) / (1024 * 1024)

            if output_size_mb <= config["MAX_MB"]:
                # Éxito
                result["success"] = True
                result["message"] = f"OK ({current_bitrate}kbps, {output_size_mb:.2f}MB)"
                return result
            
            # Si falló, reducir bitrate y reintentar
            if attempts < max_attempts:
                # Reducir un 15% el bitrate
                current_bitrate = int(current_bitrate * 0.85)
                # Si bajamos del mínimo, forzar mono si era estéreo
                if current_bitrate < MIN_BITRATE and channels == 2:
                    channels = 1
                    current_bitrate = 64 # Reiniciar bitrate para mono
                    
                result["message"] = f"Intento {attempts} excedió tamaño ({output_size_mb:.2f}MB). Reintentando a {current_bitrate}kbps..."
                # No logueamos aquí para evitar desorden en multiproceso, el mensaje final se maneja afuera

        # Si llegamos aquí, fallamos después de max_attempts
        result["message"] = f"No se pudo reducir a {config['MAX_MB']}MB después de {max_attempts} intentos"
        return result

    except Exception as e:
        result["message"] = f"Excepción: {str(e)}"
        return result

def scan_and_process(config):
    """Escanea y orquesta el procesamiento"""
    processed_files = load_processed_files()
    os.makedirs(config["OUTPUT_DIR"], exist_ok=True)
    
    # Buscar archivos
    files_to_process = []
    try:
        candidates = [f for f in os.listdir(config["INPUT_DIR"]) 
                     if os.path.splitext(f.lower())[1] in SUPPORTED_EXTENSIONS]
    except FileNotFoundError:
        logging.error(f"Directorio de entrada no encontrado: {config['INPUT_DIR']}")
        return 0

    for file in candidates:
        input_path = os.path.join(config["INPUT_DIR"], file)
        
        # Saltarse archivos de salida generados
        if "_wp_" in file:
            continue
            
        file_hash = get_file_hash(input_path)
        
        if file not in processed_files or processed_files[file] != file_hash:
            files_to_process.append((input_path, config, file_hash))
        # Si ya está procesado, lo ignoramos silenciosamente en logs (o debug)

    count = len(files_to_process)
    if count == 0:
        return 0

    logging.info(f"Encontrados {count} archivos para procesar")
    if config["ENABLE_PARALLEL"]:
        logging.info(f"Modo Paralelo: {config['CPU_THREADS']} procesos")
    
    new_processed_count = 0
    
    # Determinar hilos por trabajo de FFmpeg
    # Si procesamos en paralelo, FFmpeg debe usar 1 hilo para no saturar CPU (que ya tiene N procesos)
    # Si procesamos secuencial, FFmpeg puede usar todos los núcleos
    ffmpeg_threads = 1 if config["ENABLE_PARALLEL"] else config["CPU_THREADS"]

    # Preparar tareas agregando el argumento de threads
    tasks = [t + (ffmpeg_threads,) for t in files_to_process]

    results = []
    
    if config["ENABLE_PARALLEL"] and count > 1:
        # Multiprocessing
        num_processes = min(config["CPU_THREADS"], count)
        with multiprocessing.Pool(processes=num_processes) as pool:
            # Usar tqdm para barra de progreso
            for res in tqdm(pool.imap_unordered(process_single_file, tasks), total=count, unit="audio"):
                results.append(res)
    else:
        # Secuencial
        for task in tqdm(tasks, unit="audio"):
            results.append(process_single_file(task))

    # Procesar resultados y guardar JSON (solo el hilo principal escribe)
    for res in results:
        if res["success"]:
            processed_files[res["file"]] = res["hash"]
            new_processed_count += 1
            logging.info(f" {res['file']}: {res['message']}")
        elif not res["skipped"]:
            logging.error(f" {res['file']}: {res['message']}")

    if new_processed_count > 0:
        save_processed_files(processed_files)
    
    return new_processed_count

def watch_folder(config):
    """Modo vigilancia"""
    logging.info(f"Modo vigilancia activo en: {os.path.abspath(config['INPUT_DIR'])}")
    logging.info(f"Salida: {os.path.abspath(config['OUTPUT_DIR'])}")
    logging.info("Presiona Ctrl+C para detener")

    try:
        while True:
            scan_and_process(config)
            time.sleep(config["WATCH_INTERVAL"])
    except KeyboardInterrupt:
        logging.info("\n Detenido por usuario")


def main():
    parser = argparse.ArgumentParser(description="Reductor de Audio para WhatsApp v2.0")
    
    parser.add_argument("--input", "-i", default=DEFAULT_CONFIG["INPUT_DIR"], help="Directorio de entrada")
    parser.add_argument("--output", "-o", default=DEFAULT_CONFIG["OUTPUT_DIR"], help="Directorio de salida")
    parser.add_argument("--limit", "-l", type=float, default=DEFAULT_CONFIG["MAX_MB"], help="Límite en MB (Default: 16)")
    parser.add_argument("--watch", "-w", action="store_true", help="Activar modo vigilancia continuo")
    parser.add_argument("--parallel", "-p", action="store_true", help="Activar procesamiento paralelo múltiple")
    parser.add_argument("--threads", "-t", type=int, default=DEFAULT_CONFIG["CPU_THREADS"], help="Número de hilos/procesos a usar")
    parser.add_argument("--format", "-f", default=DEFAULT_CONFIG["OUTPUT_FORMAT"], choices=["mp3", "m4a", "ogg"], help="Formato de salida (mejor calidad: m4a/ogg)")
    parser.add_argument("--no-filter", action="store_true", help="Desactivar filtros de mejora de voz")
    
    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    config["INPUT_DIR"] = args.input
    config["OUTPUT_DIR"] = args.output
    config["MAX_MB"] = args.limit
    config["WATCH_MODE"] = args.watch
    config["ENABLE_PARALLEL"] = args.parallel
    config["CPU_THREADS"] = args.threads
    config["OUTPUT_FORMAT"] = args.format
    
    global VOICE_FILTERS
    if args.no_filter:
        VOICE_FILTERS = False

    print(" PROCESADOR DE AUDIO PARA WHATSAPP (MEJORADO)")
    print("=" * 50)
    print(f" Entrada: {config['INPUT_DIR']}")
    print(f" Salida:  {config['OUTPUT_DIR']}")
    print(f" Límite:  {config['MAX_MB']} MB")
    print(f" Paralelo: {'SÍ' if config['ENABLE_PARALLEL'] else 'NO'} ({config['CPU_THREADS']} hilos)")
    print(f"  Filtros:  {'SÍ' if VOICE_FILTERS else 'NO'}")
    print("=" * 50)

    if not check_ffmpeg():
        logging.error(" FFmpeg no encontrado. Instálalo y agrégalo al PATH.")
        return

    if config["WATCH_MODE"]:
        watch_folder(config)
    else:
        start_time = time.time()
        processed = scan_and_process(config)
        elapsed = time.time() - start_time
        
        if processed > 0:
            logging.info(f"\n Completado en {elapsed:.2f}s ({processed} archivos)")
        else:
            logging.info("\n No se encontraron archivos nuevos.")

if __name__ == "__main__":
    # Soporte para ejecución en Windows sin abrir ventana extra en multiprocessing
    multiprocessing.freeze_support()
    main()
