# Diario di sviluppo

Note sul contesto e le decisioni prese durante lo sviluppo di `bike_analysis.py`, per chi riprende il progetto in futuro (umano o AI).

## Obiettivo

Script CLI per l'analisi biomeccanica di video di pedalata (bike fitting): a partire da un video laterale del ciclista, misurare l'angolo del ginocchio, del busto e della caviglia frame per frame, stimare l'escursione angolare (ROM) del ginocchio per valutare l'altezza sella (range ideale di flessione a fondo corsa: 25–35°), rilevare i cicli di pedalata ed esportare video annotato, CSV e grafico.

## Decisioni tecniche e perché

- **MediaPipe Pose, API legacy (`mp.solutions.pose`) invece delle Tasks API (`mediapipe.tasks.python.vision`)**: il primo tentativo con le Tasks API e il modello `pose_landmarker_lite.task` andava in crash su macOS (Apple Silicon) con `F0000 ... Check failed: service_ Service is unavailable` (problema di inizializzazione GPU/Metal in `TensorsToDetectionsCalculator`). Risolto scaricando `mediapipe` a `0.10.21` e passando alla API legacy, che è stabile su questo ambiente. Il modello `.task` scaricato per il tentativo abbandonato è stato rimosso dal repo.
- **Rotazione video verticali (smartphone)**: OpenCV 4.11 non applica di default la rotazione da metadati EXIF/matrix. Serve `cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)`. Inoltre `CAP_PROP_FRAME_WIDTH/HEIGHT` riportano le dimensioni del flusso codificato ignorando la rotazione, mentre i frame restituiti da `cap.read()` sono già ruotati: le dimensioni reali per il `VideoWriter` vanno lette dalla `.shape` del primo frame, non dalle `CAP_PROP`.
- **Niente simbolo "°" nell'overlay video**: `cv2.putText` usa i font Hershey, senza supporto Unicode («°» diventava «??»). Nell'overlay video si usa la stringa `"deg"`; nei grafici matplotlib e nei `print()` a console il simbolo `°` funziona regolarmente e viene mantenuto.
- **Jitter di tracking sui keypoint**: analizzando il video di test, il jitter frame-per-frame (soprattutto su caviglia e punta del piede) è risultato in parte dovuto a rumore reale del tracking, non solo a movimento veloce reale. Soluzione: interpolazione dei frame con keypoint mancanti/poco affidabili + filtro Savitzky-Golay (finestra di default 0.3s, `polyorder=2`). La finestra è stata scelta empiricamente confrontando 0.15s/0.3s/0.45s su un segmento di dati grezzi: 0.15s riduceva poco il rumore, 0.45s appiattiva i picchi reali di 10-15° (inaccettabile per la misura del ROM), 0.3s è il miglior compromesso.
- **Rilevamento cicli di pedalata**: `scipy.signal.find_peaks` su `angolo_ginocchio` (e sul suo opposto per le valli), con vincoli di `prominence` e `distance` minima per evitare falsi positivi generati da micro-oscillazioni. Viene inoltre calcolato il numero medio di frame per mezzo ciclo: se troppo basso (< `MIN_HALF_CYCLE_FRAMES_RELIABLE = 8`), lo script avvisa che il framerate del video è probabilmente insufficiente per catturare con affidabilità i punti morti (PMS/PMI).
- **Output in cartella dedicata**: tutti i file generati (video annotato, CSV, grafico) vengono salvati di default in `output/`, configurabile con `--output-dir`. La cartella `output/` non è versionata (vedi `.gitignore`).

## Registrazione del video: raccomandazioni

Discusse a partire dai metadati del video di test (girato con Pixel 10, 29.99 fps, 1080x1920 verticale, HEVC):
- **framerate**: consigliati almeno 60 fps (i telefoni moderni, incluso Pixel 10, lo supportano in modalità standard) per catturare con affidabilità i punti morti della pedalata; 30 fps è al limite/insufficiente secondo la soglia usata dallo script.
- **risoluzione**: 1080p è sufficiente per il tracking dei keypoint ed è più leggero da elaborare del 4K, che non porta benefici significativi per questo caso d'uso.

## Cosa non è nel repository

- I video di input/output (`*.mp4`, cartella `output/`) non sono versionati: sono file pesanti e contengono, nel caso dei video usati per i test, riprese personali dell'autore mentre pedala.
- Il modello `pose_landmarker_lite.task` (scaricato per il tentativo con le Tasks API, poi abbandonato) è stato eliminato perché non più referenziato dal codice.

## Stato attuale

Script funzionante e testato end-to-end su un video reale (video completo di ~45s e una clip di 5s), con output verificati (video annotato, CSV, grafico ROM). Nessuna issue nota aperta.
