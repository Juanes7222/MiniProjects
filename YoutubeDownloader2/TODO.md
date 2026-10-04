# TODO — Auditoría de rendimiento y aprovechamiento de hardware

Objetivo: exprimir CPU, GPU, RAM y disco; eliminar tiempo muerto; eliminar
degradaciones silenciosas.

**Estado:** completo. `547 tests` · `ruff clean` · `mypy: no issues found`.

## Hardware de referencia

| | |
|---|---|
| CPU | Ryzen 5 7600X — 6 núcleos / 12 hilos |
| RAM | 32 GB |
| GPU | RTX 5060 Ti — 16 GB VRAM, compute cap **12.0** (Blackwell) |
| Driver | 617.14 · torch 2.8.0+cu128, `sm_120` presente ✅ |

---

# RESULTADOS MEDIDOS

Todo lo de abajo se midió en esta máquina salvo indicación contraria. Las
cifras que **no** se pudieron medir están marcadas como tales, y las
estimaciones que resultaron equivocadas se corrigieron.

## GPU / Kev — A/B real, misma carga (31 preguntas, 5 candidatos)

> ### ⚠️ Todas las cifras absolutas de esta sección dependen del estado de la GPU
>
> La tarjeta estaba **compartida** durante las mediciones: Wallpaper Engine
> renderizando en segundo plano y 22 procesos con contexto GPU. El reloj SM iba a
> **652 MHz en reposo sobre 3090 MHz de máximo**, y subía solo a ~1177 MHz bajo
> carga — un 38% de lo que debería.
>
> Una medición posterior de concurrencia dio **18.957 ms** por evaluación donde
> antes daba 2.470 ms, con el mismo código. La diferencia es la GPU, no el
> software.
>
> **Qué sigue siendo válido:** la comparación *relativa* entre las dos filas de
> la tabla, medida back-to-back en la misma ventana de minutos. **Qué no:** los
> números absolutos. Re-medir con la GPU libre antes de sacar conclusiones de
> capacidad.

| Configuración | CUDA graphs | fused kernels | Latencia |
|---|---|---|---|
| Antes (ambas apagadas a propósito) | OFF | OFF (sin aviso) | **3338 ms** |
| fused ON, graphs OFF | OFF | ON | **no respondió** (>90 s) |
| fused OFF, graphs ON | ON | OFF | **2451–2492 ms** |

**Ganancia: −26%** (3338 → ~2470 ms). Cuadros CUDA activos: sí, y medido.

### El hallazgo importante: `flash-linear-attention` no funciona aquí

Se instala correctamente (`fla.__version__ == 0.5.2`, exactamente lo que Kev
exige), `triton-windows==3.4.0.post21` compila y ejecuta un kernel real en
sm_120 sin error… y aun así **el primer evaluation real nunca termina**. El
mismo request con fused declinado responde en 2.5 s.

Es decir: `kev.fused_available()` (un chequeo de import) dice que sí, y miente.
Por eso la decisión ahora se toma con un **probe de latencia real bajo timeout
duro**, no con un import. Un batch que colgaba en su primera canción, con nada
en el log que lo explicara.

### `KEV_FUSED=0` resultaba ser correcto aquí — por accidente

El código original tenía `KEV_CUDA_GRAPHS=0` **y** `KEV_FUSED=0`. El segundo era
la decisión correcta para esta máquina, pero por el motivo equivocado: no
buscaba "los kernels no funcionan", y además **suprimía el aviso** de kev.serve
sobre ellos apagados. Ahora la decisión es medida, argumentada y reversible con
`--kev-fused` / `--no-kev-fused`.

### Payload de decisión: −21,4%

| | chars |
|---|---|
| Antes | 31.389 |
| Después | 24.674 |

De donde:
- allowlist de campos por dimensión (A6): el `view_count` no lo lee ninguna
  dimensión; `reference` no necesita la descripción; el `key` nunca se recorta.
- `DESCRIPTION_LIMIT` 1200 → 500: la cabecera de una descripción de YouTube
  lleva las señales ("Official Audio", "Lyrics", la línea de copyright de la
  discográfica) en los primeros ~200 caracteres. El resto era boilerplate.

**El límite real sigue siendo el número de preguntas**, no los bytes: 31
preguntas × ~250 tokens. `--decision-questions` es el dial explícito de
calidad/velocidad; bajarlo de 32 a 20 quita un 35% de preguntas de un solo
golpe. No se cambió el default porque es un compromiso de calidad, no una
optimización gratuita.

## Descargas / red

| | |
|---|---|
| Fragmentos DASH | 1 (default de yt-dlp) → **4** |
| aria2c | inalcanzable → **detectado y opcional** (`--use-aria2c`) |
| Transcode de audio | todos los cores → **`-threads 1`** |
| `concurrent_fragment_downloads` | se ignoraba en la descarga real → **se pasa el `config`** |

## CPU / decodificación

`silencedetect` a −50 dB / 3000 ms no necesita 44,1 kHz estéreo. Con
`-ac 1 -ar 8000` se midió **0,34 s → 0,28 s, un 16%**.

> **Corrección:** la estimación inicial era "5× menos decode". Se medió y es
> **16%**. El decode de mp3 ya es barato y single-thread; la ganancia real es
> modesta. El cambio se queda porque es correcto y gratis, no porque sea
> grande.

Se verificó que **no cambia el resultado**: sobre un mp3 sintetizado con
silencio conocido, ambas rutas dan idéntico ratio, idéntico número de
segmentos, idéntica duración y el mismo veredicto del gate.

## Memoria / footprint

| | |
|---|---|
| venv de la app | 750 MB → **240 MB** |

torch 2.14.1+cpu (509 MB) estaba instalado y **nada en `ytdl_core` lo
importaba** — las dos únicas menciones son strings que se ejecutan con el
intérprete del venv de Kev. Además avisaba al importar porque faltaba numpy.

---

# FASE 1 — Kev / GPU

- [x] **A1. `KEV_CUDA_GRAPHS=0` apagaba los CUDA graphs.** Medido −26%.
- [x] **A2. `KEV_FUSED=0` apagaba los kernels fusionados *y su aviso*.** Ahora la
      decisión es medida por un probe de latencia con timeout duro, y es
      reversible con `--kev-fused` / `--no-kev-fused`.
- [x] **A3. `flash-linear-attention` + Triton.** Se instalan
      `flash-linear-attention==0.5.2` y el Triton que corresponde al torch
      (Windows: `triton-windows~=3.4.0` para torch 2.8, mapeado por versión;
      Linux: el que ya viene con el build cu128). Idempotente y no fatal.
- [x] **A4. `DECISION_MAX_IN_FLIGHT` 1 → 6**, expuesto como
      `--decision-in-flight`. Kev drena hasta `MAX_BATCH=64` requests en un
      solo model pass; con 1 el techo del lote era `1/latencia`.
- [x] **A5. `requests.Session` con keep-alive.** Antes cada request abría conexión
      TCP+TLS nueva; se veía en el log del servidor (puerto origen distinto en
      cada línea). Ahora el pool se dimensiona al in-flight.
- [x] **A6. Allowlist de campos por dimensión** + `DESCRIPTION_LIMIT` a 500.
      −21,4% de payload. Bloqueado con 8 tests.
- [x] **A7. Telemetría de Kev capturada**: `latency_ms`,
      `usage.input_tokens`/`output_tokens`, `x-typesafe-request-id`. Se
      descartaban; ahora se reportan.
- [x] **A8. Verificación post-arranque** de lo que el servidor *hace*, no de lo
      que el entorno pide: `cuda_graphs` de `/v1/models`, el banner de fused en el
      log del servidor, y las estadísticas de batching y de state cache.
- [x] **A9. `--kev-runs > 1` documentado como trabajo duplicado** para un modelo
      determinista (prefill-only; la temperatura nunca cambia qué gana).
- [x] **A10. Modelo pineado** a `jaredpalmer/kev-4b@v1.0`. El nombre sin pin
      seguía `main`, que ya había resuelto a otro snapshot.

# FASE 2 — Tiempo muerto y concurrencia

- [x] **B1. `fpcalc` y el backend de pyacoustid.** `fpcalc.exe` se encuentra ahora
      por ruta absoluta (PATH primero, luego raíz del paquete/proyecto), y
      `acoustid.FPCALC_COMMAND` se apunta a él.
      **El problema real era otro:** `have_chromaprint` es `False` en este
      entorno, así que pyacoustid usaba su backend `audioread` — un FFT en Python
      puro — por cada clip de 90 s, aunque `fpcalc` estuviera en el disco. Ahora
      se pasa `force_fpcalc=True`. También esto rearmaba `TRUST_VERIFIED_MULTIPLIER`
      del modelo de confianza de canales, que nunca se disparaba.
- [x] **B2. `_prime_catalogs` ya no es una barrera.** Antes enviaba todos los
      artistas y **esperaba a todos** antes de arrancar el pipeline: con
      MusicBrainz a 1 req/s, N artistas = N segundos sin hacer nada. Ahora dispara
      las llamadas y sigue; `_resolve_catalog` ya sabe caer al path por canción.
- [x] **B3. Transcode de audio limitado a 1 hilo** vía `postprocessor_args`
      top-level. ffmpeg tomaba todos los cores por defecto en cada descarga
      concurrente.
- [x] **B4. `concurrent_fragment_downloads` = 4**, no el 1 de yt-dlp.
- [x] **B5. `aria2c` alcanzable**: `find_aria2c()` lo localiza (el `.exe` de 5,6 MB
      llevaba ahí sin usarse) y `--use-aria2c` lo activa. Opt-in, y avisa si se
      pide y no está.
- [x] **B6. `audio.py` eliminado.** 298 líneas de código muerto: `download_audio`
      no tenía callers, y era el único sitio con fragmentos concurrentes, aria2c y
      el hook de progreso. Sus tres piezas útiles están ahora en el path real.
- [x] **B7. `search_all_sources` es determinista.** Recolectaba con
      `as_completed`, así que el orden dependía del timing de red → "primera
      ocurrencia gana" en el dedup era no determinista → el desempate por stable
      sort de `rank_results` también. Contradecía el docstring de
      `search_with_variants`, que promete reproducibilidad byte a byte. Ahora
      recolecta en orden de submission.
- [x] **B8. `ytmusicapi` declarado** en `pyproject.toml` y `requirements.txt`. Se
      importaba en try/except sin estar declarado: un `pip install .` limpio
      perdía en silencio `ytmusic_api` y `search_ytmusic_album`, el match exacto
      por label. Los tests lo mockean, así que CI nunca lo vio.
- [x] **B9. Telemetría de producción.** `stage_stats()`, `write_count`,
      `DecisionGate.successes/failures` existían y solo los ejercitaban los
      tests. Ahora se reportan al final del lote.
- [x] **B10. `build_ytdlp_base_opts` recibía el `config`.** No lo recibía: sin él
      el builder caía a un `Config()` de módulo, así que *todo* override —
      concurrencia de fragmentos, aria2c, retries — se ignoraba en la única
      llamada que realmente descarga el archivo.

# FASE 3 — CPU / disco

- [x] **C1. `silencedetect` a `-ac 1 -ar 8000`.** Medido −16% (no 5×; corregido).
      Verificado que no cambia el resultado.
- [x] **C2. blake2b con etiqueta de algoritmo.** MD5 no tiene aceleración por
      hardware en ningún x86 actual. El tag (`blake2b:<hex>`) es lo que hace el
      cambio seguro: una entrada legacy sin etiqueta se re-verifica con MD5, así
      que una biblioteca existente no se marca como cambiada ni se re-descarga.
      Chunk de 64 KB → 1 MB.
- [x] **C3. `verifier.py` hasheaba el mismo archivo dos veces.** Ahora reutiliza
      el digest que ya se comprobó en la restauración desde el estado.
- [x] **C4. torch fuera del venv de la app** (509 MB, sin uso).
- [x] **C5. `mypy` útil.** `no_site_packages = true` hacía que toda llamada a
      terceros resolviera a `Any`. Ahora la primera vez: **sin errores**.

---

# Bugs encontrados AL CORRER (no leyendo)

Dos los encontró el E2E, y por eso ahora hay tests que los fijan:

1. **`postprocessor_args` dentro del dict del postprocesador rompía *todas* las
   descargas.** yt-dlp pasa ese dict directo al constructor del postprocesador,
   así que la clave desconocida daba
   `FFmpegExtractAudioPP.__init__() got an unexpected keyword argument` en cada
   canción del lote. Va top-level, con clave por nombre de postprocesador.
   Fijado en `tests/test_ytdlp_options.py`.
2. **`file_matches_hash` comparaba el string entero** con la etiqueta, así que
   ninguna entrada legacy podía coincidir nunca. Se compara el hex.
3. **`write_count` es una property que devuelve un int**, y se consultaba con
   `callable()` — así que la línea de telemetría nunca salía.

---

# Descartado con motivo

- **Reescribir en Rust/Go.** El hot path en Python es I/O-bound de principio a
  fin (HTTP de yt-dlp, AcoustID, MusicBrainz, iTunes, Kev). El único trabajo
  CPU-nativo real es ffmpeg (ya C) y RapidFuzz (ya C++). El GIL no es el cuello
  de botella. La única jugada nativa que paga es **aria2c**, que ya estaba en el
  repo sin usarse (ver B5).
- **Insistir en los kernels fusionados.** Descartado por medición en esta
  máquina: cuelgan. Queda disponible por flag para hardware donde sí funcionen.
- **Subir `KEV_PREFIX_CACHE`.** Cada canción manda un state distinto en un solo
  request, así que la caché de prefijos solo ayuda en reintentos. Con 16 GB de
  VRAM y ~14 GB residentes, bajarla o dejarla en 4.
- **`KEV_DTYPE=fp32`.** El default de Kev en GPU ya es bf16, y la doc dice que
  bf16 es 2–4,5× más rápido con las mismas respuestas.
- **Bajar `DECISION_MAX_QUESTIONS` por defecto.** Funciona, pero es un
  compromiso de calidad y debe ser decisión explícita, no un default cambiado a
  escondidas.

---

# Verificación

```bash
python -m pytest -q          # 547 passed
python -m ruff check ytdl_core tests
python -m mypy ytdl_core     # no issues found

# estado real del servidor de Kev
curl -s localhost:8009/v1/models | python -m json.tool
#   -> cuda_graphs: {...}   (null = graphs apagados)
#   -> batches: {count, requests, queued}
#   -> prefix_cache: {hits, misses, cached_states, oom_retries}
grep "fused Qwen3.5 kernels off" ~/.cache/ytdl/kev/ytdl-kev.log
```

## Lo que queda abierto

Ordenado por lo que más cambia una decisión.

### 1. Re-medir con la GPU libre  ← lo más importante
Todo lo absoluto de la sección GPU es dudoso, y `DECISION_MAX_IN_FLIGHT = 6` se
eligió **por razonamiento, no por medición**: el throughput a 6 concurrentes
nunca se midió en esta máquina. Con la GPU-clock correcta se puede:
- confirmar o ajustar `--decision-in-flight` (el número correcto depende de VRAM
  libre, y 14,3 GB residentes en 16 GB deja poco margen para batches grandes)
- obtener la latencia real, y con ella saber si compensa bajar
  `--decision-questions`
- volver a intentar los kernels fusionados con la GPU sin competencia

### 2. Los kernels fusionados siguen sin funcionar en sm_120 / Windows
~1/3 del tiempo de GPU no se está aprovechando. El probe ya los detecta y
degrada con un aviso honesto, así que no hay riesgo, pero la ganancia existe.
Vale la pena solo si aparece un build de Triton que los ejecute, o si el
problema fuera de la fricción de `triton-windows`. `--kev-fused` fuerza el camino.

### 3. `aria2c` y `--fragment-concurrency` — MEDIDOS, ninguno es una ganancia

Descargas reales, 4 configuraciones × 2 canciones × 3 repeticiones
intercaladas (para que unaauce de red no caiga siempre en la misma), misma
canción de 5,97 MB:

| Configuración | mediana | vs baseline |
|---|---|---|
| baseline (1 fragmento) | **12,7 s** | — |
| `aria2c`, 1 fragmento | 22,3 s | **+75%** |
| sin aria2c, 8 fragmentos | 12,6 s | **−1%** |
| aria2c, 8 fragmentos | 22,5 s | **+77%** |

**`--use-aria2c` es una pessimización en esta red** (~450 KB/s). Repartir el
ancho de banda de una conexión entre ocho cuesta más en arranque de proceso,
ocho setups de conexión y un fichero `.aria2` de lo que recupera. La herramienta
adecuada es cuando una conexión individual está limitada o el RTT es alto, no
cuando el que limita es el enlace. Sigue siendo opt-in y apagado.

**`--fragment-concurrency` es inerte con la configuración actual.** Se
seleccionó el formato **251 (webm/opus), `protocol=https`, sin fragmentos**: con
`YOUTUBE_PLAYER_CLIENTS = android/mweb/web_embedded` yt-dlp recibe el audio como un
único fichero progresivo, así que no hay nada que traer en paralelo. Pasaría a
importar si un formato llegara segmentado (otro player client, o un directo).

> **Dos afirmaciones mías anteriores resultaron incorrectas**, y la medición es
> lo que las refuta:
> 1. *"aria2c es la jugada nativa que paga, la diferencia entre saturar la línea y no"* → en esta línea, es 75% más lento.
> 2. *"fragmentos concurrentes: el cambio de red más barato que hay"* → no hay fragmentos que traer.

### 4. AcoustID end-to-end — SE ENCONTRÓ UN BUG REAL 🔴

Con key real. El pipeline corría (partial, fpcalc, sin errores, sin ficheros
huérfanos) y devolvía **`no AcoustID match` en 4/4 canciones**. No era la
configuración: era un bug que hacía la etapa **incapaz de verificar nada**.

### La causa: se declaraba la duración del clip, no la de la canción

AcoustID acota su conjunto de candidatos por la duración que le declaran.
`acoustid.match` lee esa duración **del fichero que le pasan** — y el pipeline le
pasa un excerpt de 90 s, así que declaraba 90 s. Pero el audio de esos 90 s no
viene de una grabación de 90 s.

Medido sobre una pista que AcoustID reconoce a 0.95 desde el fichero entero,
usando **los mismos 90 s de audio** en todos los casos:

| Duración declarada | Resultado |
|---|---|
| 90 s (lo que hacía el pipeline) | sin match |
| 120 / 180 / 240 / 300 s | sin match |
| **340 s (la real)** | **MATCH 0.951** |
| 400 / 600 s | sin match |

La ventana es **estrecha** alrededor de la cifra real: 300 y 400 fallan. Por eso
hay que pasar la duración verdadera, no una generosa.

Descartado por el camino:
- **No es la re-codificación**: el fichero entero re-codificado a 128 kbps sigue
  dando 0.950. Un clip de 120 s con `-c copy` tampoco da.
- **No es la longitud mínima**: con `maxlength=120` el fingerprint saturado da
  `fp_len` 3350 (clip) contra 3342 (fichero completo) — prácticamente el mismo
  input, y sin embargo uno coincide y el otro no.
- **No son los acentos**: `fpcalc` abre sin problema los 38 ficheros con
  caracteres no-ASCII de la biblioteca.

### El arreglo

`_lookup()` hace el fingerprint igual que `match` pero declara la duración de la
canción, que el pipeline ya conoce. Se hilado desde tres sitios, y en
`_try_next_fp` **cada alterno declara la suya** porque cada uno es otra
grabación — reutilizar la del ganador buscaría la pista equivocada.

Verificado sobre el pipeline real (partial de 90 s → fpcalc → lookup → veredicto
→ cache), con la URL que el propio state file tenía registrada:

```
RESULT : verified=True confidence=0.95 title='Dios háblame'
label  : 'verified 95% conf.'
cached : (True, 0.95022166, 'Dios háblame')
```

Antes de ese arrangement, esa misma canción tenía `fingerprint_label: "no
AcoustID match"` en el state file.

**Falta**: re-verificar la biblioteca con `--verify` para republicar los sellos,
que ahora serán correctos. Antes salieron `no AcoustID match` o `disabled`.

### Un aviso sobre mi propio diagnóstico

Durante esta investigation reporté casi un bug que no existía: `fpcalc exited
with status 2` sobre un fichero con apóstrofe. La causa era **mi ruta mal
escrita** — el fichero real se llama `Dios Háb'lame.mp3` con apóstrofo
tipográfico (U+2019) y `sanitize_filename` se lo había quitado, así que la ruta
que construí a mano no existía. `fpcalc` estaba perfectamente bien. Un test que
no verifica su propia premisa no vale como diagnóstico.

### Corrección a B1

B1 ("fpcalc + force_fpcalc") era **mucho menos de lo que parecía**. El valor real
era solo la **ruta absoluta**: `shutil.which` en Windows devuelve un nombre
relativo cuando el binario está en el directorio actual, así que la etapa
dependía del cwd. El `force_fpcalc=True` es una garantía, no una optimización:
pyacoustid solo prefiere su FFT en Python cuando la extensión C de chromaprint
**también** está presente, y aquí no lo está — medido, ambos caminos dan tiempos
indistinguibles. Lo que de verdad arreglaba la verificación era lo de este punto.

### 5. Módulos sin auditar
`reports.py` (115), `retry_queue.py` (164), `json_io.py` (24), `cli_entry.py`
(11), y del CLI: `review.py` (428), `interactive.py` (365), `rich_ui.py` (626).
Los últimos tres son código de interacción, no de rendimiento, pero
`rich_ui` recibe un evento por cada progreso de descarga desde 36 hilos — ahí
puede haber contención que no miré.

### 6. Config muerta — RESUELTO ✅

Seis campos declarados y nunca leídos. **No era un bug de selección.** La
evidencia del historial:

| Campo | Introducido | Por qué estaba muerto |
|---|---|---|
| `COVER_KARAOKE_PENALTY` (−50) | 2026-05-03 | Superado un mes después por el **hard-reject** de forbidden terms: "cover"/"karaoke" ahora se descartan a −9999 |
| `REACTION_REMIX_PENALTY` (−50) | 2026-05-03 | Ídem, con "reaction"/"remix"/"mashup" |
| `HIGH_FUZZY_BONUS` (+20) | 2026-05-03 | Discriminador que el scorer dejó de usar; el ranking ya usa `song_match` |
| `DECIDE_WORKERS` | 2026-10-03 (ayer) | Etapa que no existe: la decisión corre dentro del stage de búsqueda y la limita `DECISION_MAX_IN_FLIGHT` |
| `MUSICBRAINZ_APP` | 2026-05-03 | Cuatro copias hardcodeadas del mismo par de strings |
| `DECISION_PROBE_BUDGET_SECONDS` | 2026-10-03 (ayer) | El valor estaba duplicado como literal en el default del método |

Los hard-reject son **más estrictos** que las penalizaciones que dejaron atrás,
así que la lógica de selección nunca estuvo más blanda de lo que se creía: está
más fuerte. `HIGH_FUZZY_BONUS` sí sería un discriminador perdido, pero
reintroducirlo cambiaría qué canción se descarga en toda la biblioteca, y eso no
se decide de pasada.

Resolución:
- Los cuatro realmente muertos → **eliminados**, con el motivo escrito en
  `config.py` para que la decisión no se pierda.
- `MUSICBRAINZ_APP` → **cableado** en los 4 sitios (pair para la librería,
  string unido para los headers HTTP), con un helper que tolera un valor mal
  configurado en vez de mandar un User-Agent vacío.
- `DECISION_PROBE_BUDGET_SECONDS` → **cableado** al probe de latencia.
- **Guard permanente**: `tests/test_config_health.py` falla si un campo de
  `Config` queda sin leerse, si se declara dos veces, o si una copia hardcodeada
  del User-Agent reaparece. Verificado que **atrapa** un campo muerto inyectado
  y pasa tras revertirlo.

**Ahora: 92 campos, 0 sin uso.**

### 6. Sin benchmark del pipeline completo
Todas las cifras de red/CPU son de componentes por separado. La telemetría nueva
—etapas, coalescing del state, batches del servidor, latencia por evaluación—
ya existe, así que un A/B de 500 canciones es ahora posible. No es urgente:
requiere una biblioteca grande.