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

## GPU / Kev — medido en esta máquina, GPU sin contention

> ### ⚠️ La sección anterior daba números no fiables, y sus conclusiones eran falsas
>
> Las mediciones de la primera pasada se hicieron con la tarjeta **compartida**
> (Wallpaper Engine y 22 procesos con contexto GPU; reloj SM a 652 MHz de 3090).
> Todo lo absoluto de aquella tabla está retirado. Lo que se mantiene son las
> cifras de abajo, tomadas con la GPU libre, y las tres correcciones que
> salieron de ellas.

### Ley de escalado: el coste es superlineal en el número de preguntas

Un request, con CUDA graphs activo, `kev-4b@v1.0` en RTX 5060 Ti 16 GB:

| candidatos | preguntas | latencia |
|---|---|---|
| 1 | 6 | 1,3 s |
| 2 | 13 | 2,2 s |
| 3 | 19 | 3,1 s |
| 4 | 25 | 10,1 s |
| 5 | **31** | **29,8 s** |
| 6 | 37 | 82,4 s |
| 12 | 73 | 221,7 s |

12× las preguntas → **440× el tiempo**. El codo está entre 31 y 37 preguntas. Por
eso `DECISION_MAX_QUESTIONS = 32` es el número correcto: con 6 dimensiones deja
5 candidatos y 31 preguntas. El recorte a presupuesto ya existía
(`JevClassifier._trim_to_question_budget`) y sí funciona; lo que no funcionaba era
el probe, que medía 19 preguntas y daba un veredicto optimista.

### CUDA graphs: load-bearing, no una optimisation del 26%

Mismo payload real de 31 preguntas, un request en vuelo:

| | graphs ON | graphs OFF |
|---|---|---|
| 31 preguntas | **3,0 s** | **124,1 s** |

La cifra de "−26%" era de una forma de batch estrecha. La realidad es que sin
graphs un request normal **no cabe ni de lejos** en `DECISION_TIMEOUT_SECONDS`, y
el probe de arranque no lo detectaba porque medía 19 preguntas (la parte plana de
la curva). El probe ahora se dimensiona con el mismo presupuesto que el pipeline.

### `DECISION_MAX_IN_FLIGHT`: 6 → 1 (A4 estaba equivocado)

Payload real de 31 preguntas, CUDA graphs activo:

| en vuelo | resultado | throughput |
|---|---|---|
| 1 | 3,0 s | 0,33 req/s |
| 2 | 5,8 s (ambos OK) | 0,34 req/s |
| 4 | **4/4 timeout >150 s** | — |
| 6 | **6/6 timeout >150 s** | — |

Subirlo a 6 se hizo con un argumento sobre *batching* (`MAX_BATCH=64`) sin
medir. Medido: 2 en vuelo **no dasthroughput** (0,34 contra 0,33) y duplica los
estados residentes; a partir de 4 el batch crece, la captura de CUDA graphs
empieza a churnar sobre formas nuevas (`captured` 0→1→2, `pending` 11→16) y
**todas** las peticiones encoladas detrás pierden el plazo del cliente. La GPU
luego se queda al 100% terminando trabajo que nadie espera — exactamente el fallo
que subir el límite pretendía evitar. De vuelta a 1, con los números en el
comentario de `Config`.

### `flash-linear-attention` sí funciona aquí

Dos errores de mi parte se combinaban para conclusionarlo al revés:

1. `flash-linear-attention` importa como **`fla`**, no como `flash_linear_attention`.
   Mi comprobación manual usó el nombre equivocado y "probó" la ausencia del paquete.
2. Los avisos de transformers (`causal_conv1d_fn` / `chunk_gated_delta_rule`
   "falling back to its reference PyTorch implementation") **no son evidencia**:
   transformers busca el módulo `flash_linear_attention`, que ese paquete no
   expone; kev usa `fla` directamente en su `fused_qwen35`.

El probe de latencia real bajo timeout duro se queda —es buena defensa—, pero el
veredicto actual es que **los kernels funcionan** (1,8 s para las 31 preguntas del
payload real). El "no respondió >90 s" anterior era la GPU compartida y la
captura de graphs, no FLA.

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

- [x] **A1. `KEV_CUDA_GRAPHS=0` apagaba los CUDA graphs.** Medido: 3,0 s → 124,1 s
      para el payload real de 31 preguntas. No era una optimisation del 26%, era
      la condición de posibilidad de que un request acabe.
- [x] **A2. `KEV_FUSED=0` apagaba los kernels fusionados *y su aviso*.** Ahora la
      decisión es medida por un probe de latencia con timeout duro, y es
      reversible con `--kev-fused` / `--no-kev-fused`. Veredicto actual en esta
      máquina: **los kernels funcionan**.
- [x] **A3. `flash-linear-attention` + Triton.** Se instalan
      `flash-linear-attention==0.5.2` (importa como `fla`) y el Triton que
      corresponde al torch (Windows: `triton-windows~=3.4.0` para torch 2.8,
      mapeado por versión; Linux: el que ya viene con el build cu128).
      Idempotente y no fatal.
- [x] **A4. `DECISION_MAX_IN_FLIGHT` 6 → 1.** Se subió a 6 por un argumento sobre
      batching, sin medir, y **fue la causa del fallo de un lote real de 1554
      canciones**. Medido con el payload real: 1 → 3,0 s; 2 → 5,8 s sin ganancia
      de throughput; 4 → 4/4 timeout >150 s; 6 → 6/6 timeout >150 s. Sigue
      expuesto como `--decision-in-flight`.
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
- [x] **A9. `runs > 1` documentado como trabajo duplicado** para un modelo
      determinista (prefill-only; la temperatura nunca cambia qué gana). Los flags
      `--jev-runs` y `--kev-runs` se fundieron en **`--decision-runs`** (A11).
- [x] **A10. Modelo pineado** a `jaredpalmer/kev-4b@v1.0`. El nombre sin pin
      seguía `main`, que ya había resuelto a otro snapshot.
- [x] **A11. Los tres flags con prefijo de proveedor son uno.** `--jev-runs`,
      `--kev-runs`, `--jev-threshold`, `--kev-threshold` y `--kev-timeout`
      vivían en pares que sólo se diferenciaban en qué flag se escribiera, y
      leían el mismo `Config.JEV_*`. Ahora son `--decision-runs`,
      `--decision-threshold` y `--decision-timeout`. Los nombres viejos se
      aceptan como alias sobre el mismo `dest` y avisan una vez por stderr; un
      perfil `profiles.toml` que los use se mapea y avisa, y poner las dos
      together es un error en vez de un "gana el último" que dependía del orden
      del fichero.
- [x] **A12. `--jev` recibía la mitad del presupuesto.** Se construía el
      `JevClassifier` sin `timeout_seconds`, `max_in_flight`,
      `failure_threshold`, `max_questions` ni `eval_headroom`, así que
      `--decision-questions` y `--decision-in-flight` no le aplicaban mientras
      parecían aplicarle. Todos los providers se puntúan con las mismas
      preguntas, así que todos reciben los mismos límites.

# FASE 1b — Decider

- [x] **D1. `decider.serve` habla el mismo endpoint que `kev.serve`.**
      `POST /v1/systemone`, dialecto TypeSafe, tipo `noul`. Por eso el
      transporte (pool, gate, reintentos, telemetría) se extrajo a
      `ytdl_core/systemone.py` y `DeciderClassifier` es una subclase fina igual
      que `KevClassifier`. No es un provider de otro tipo: es otra URL.
- [x] **D2. `decider` como provider "sin GPU".** `decider.serve` resuelve
      CUDA → MPS → CPU por su cuenta, y publica pesos GGUF para llama.cpp. Es el
      caso que Kev no cubre: su supervisor se detiene en "no NVIDIA device was
      found". Su supervisor reporta el device que realmente cargó, porque un
      servidor en CPU responde bien y es ordinales de magnitud más lento.
- [x] **D3. El nombre del checkpoint lo pregunta el servidor.**
      `decider.serve` nombra el checkpoint según la `version` de su propio
      `decider_config.json` (`decider-v2.1`), así que no hay cadena estable que
      pinear como `kev-latest`. `--decider-model` sin valor lo lee de
      `/v1/models` una vez y lo recuerda; si el servidor no contesta, el campo se
      omite, que es opcional en el cable. Pinear un nombre fijo habría dado 404
      contra todo checkpoint menos uno, y el error habría parecido "falta el
      endpoint".
- [x] **D4. Sin repositorio que clonar.** `decider-ai` es un paquete normal: venv
      + install pineado (`decider-ai==1.9.0`), con marcador para no reinstalar.
      Kev pineaba los *pesos* a `@v1.0` pero seguía `main` en el *código*; aquí
      no hay código que persiga.
- [x] **D5. No se sobreescriben los defaults de `decider.serve`.** Ya calienta
      sus CUDA graphs al arrancar, ya ajusta una temperatura por tipo de
      pregunta desde la config del modelo, y ya elige device. Tocar eso desde aquí
      sería volver a decidir, en otra máquina, preguntas que sus autores midieron
      en la suya.
- [ ] **D6. Falta medirlo en estas preguntas.** Los números publicados de Decider
      y de Kev vienen de suites distintas y no son comparables entre sí: Decider
      está por detrás de Jev en conocimiento (MMLU/GPQA/GSM8K) y por delante en
      calibración. Nuestro trabajo no es knowledge, pero eso es una hipótesis
      sobre *nuestras* preguntas, no un hecho. `--kev` sigue siendo el default
      hasta que haya un A/B de 500 canciones medido con
      `--decision-threshold` fijo.
- [ ] **D7. El schema cache de `decider.serve` no aplica aquí.**
      `DECIDER_SCHEMAS` cachea un prefijo que no dependa del state, pero
      nuestras `instructions` llevan el `candidate` adentro, así que el "schema"
      cambia en cada canción. Descartado, no implementado.

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

### 1. ~~Re-medir con la GPU libre~~ — hecho
Rehecho con la tarjeta libre. `--decision-in-flight` medido y **vuelto a 1**; la
latencia real es 3,0 s para 31 preguntas; y los kernels fusionados **sí
funcionan**, así que el punto 2 de esta lista era una conclusión falsa. Lo que
queda por debajo es el resto de la lista.

### 2. ~~Los kernels fusionados siguen sin funcionar~~ — falso
`flash-linear-attention==0.5.2` está instalado, importa como `fla` y responde un
payload real de 31 preguntas en 1,8 s. El "no responde >90 s" de la primera pasada
era la GPU compartida. El probe de latencia real se mantiene porque es buena
defensa, pero hoy reporta éxito.

Queda un residuo: los avisos de transformers sobre
`causal_conv1d_fn` / `chunk_gated_delta_rule` son ruido en esta configuración
(transformers busca `flash_linear_attention`, que el paquete no expone; kev usa
`fla`). Ruido que se lee como avería y ya costó una conclusión equivocada.

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

### 4b. `--verify` vaciaba el estado que el downloader había construido 🔴

Encontrado al **ejecutar** `--verify` de verdad, no leyendo. El run escribió
`status: "verified"` en las 88 entradas… y dos sitios comparaban contra el
literal `"downloaded"`.

Medido sobre la biblioteca real, antes y después de un solo `--verify`:

| | antes | después |
|---|---|---|
| `ChannelTrust.from_state` → canales | **10** | **0** |
| `--skip-existing` → canciones saltadas | **88 de 91** | **0 de 91** |

Es decir: el modelo de confianza de canales aprendía se borraba entero, y la
siguiente tanda con `--skip-existing` se **re-descargaba la biblioteca completa**.
El `channel` seguía en las 88 entradas — los datos estaban, solo dejaban de
leerse.

Arreglo: `state.has_local_file()` / `PRESENT_STATUSES = {"downloaded",
"verified"}`, y los dos sitios pasan por ahí para que la regla no vuelva a
separarse. Verificado: los 10 canales sobreviven.

**Efecto secundario que sí queríamos**: al re-verificar, 28 descargas pasaron a
`fingerprint_verified: True`, y el peso de "Abel Zavala" en el modelo de
confianza subió de **41 a 95** — `TRUST_VERIFIED_MULTIPLIER=3` por fin se
dispara, que era justo lo que B1 predijo y por fin ocurre. Eso es la
cascada de la corrección de la duración.

### Nota sobre la biblioteca real

- **88/88 verificados** en 36 s (ficheros intactos).
- **28/88 confirmados por AcoustID**; los otros 60 no están en la base. Comprobé
  cuatro directamente, fuera del pipeline y con los ficheros completos: es una
  ausencia real, no un fallo de fontanería. Material regional ADCColombia nord
 tiene poca cobertura en AcoustID.
- Los hashes **round-trip OK**: 88/88 coinciden (84 MD5 legacy + 4 blake2b). El
  tag de compatibilidad hacia atrás funciona sobre datos reales, no solo en
  tests.

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