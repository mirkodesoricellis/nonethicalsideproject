# bike-fitter

Analisi biomeccanica di video di pedalata per il bike fitting, basata su [MediaPipe Pose](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker).

Da un video laterale del ciclista mentre pedala, lo script:

- rileva i keypoint del corpo frame per frame (spalla, anca, ginocchio, caviglia, punta del piede);
- calcola l'angolo del ginocchio, l'inclinazione del busto e l'angolo della caviglia;
- traccia scheletro e angoli sovrimpressi su un video annotato;
- misura l'escursione angolare (ROM) del ginocchio e la flessione stimata a fondo corsa, per valutare l'altezza sella (range ideale 25–35°);
- rileva i cicli di pedalata e segnala se il frame rate del video è troppo basso per catturare in modo affidabile i punti morti (PMS/PMI);
- esporta un CSV con gli angoli per ogni frame e un grafico dell'escursione del ginocchio.

## Requisiti

- Python 3.9+
- macOS/Linux/Windows (testato su macOS, Apple Silicon)

```bash
pip install -r requirements.txt
```

## Uso

```bash
python bike_analysis.py --input video.mp4 --side right
```

Per default i risultati vengono salvati in `output/`:

- `output/<video>_annotated.mp4` — video con scheletro e angoli sovrimpressi
- `output/<video>_angles.csv` — angoli per frame (`frame`, `timestamp`, `angolo_ginocchio`, `angolo_busto`, `angolo_caviglia`)
- `output/<video>_knee_rom.png` — grafico dell'escursione angolare del ginocchio con massimi/minimi evidenziati

### Opzioni CLI

| Flag | Default | Descrizione |
|---|---|---|
| `--input` | *(obbligatorio)* | Percorso del video di input |
| `--output-dir` | `output` | Cartella in cui salvare video, CSV e grafico |
| `--output` | `<output-dir>/<video>_annotated.mp4` | Percorso del video annotato |
| `--csv` | `<output-dir>/<video>_angles.csv` | Percorso del CSV |
| `--plot` | `<output-dir>/<video>_knee_rom.png` | Percorso del grafico |
| `--side` | `right` | Lato del ciclista ripreso (`left` o `right`) |
| `--min-detection-confidence` | `0.5` | Soglia minima di confidenza/visibilità dei keypoint |
| `--smoothing-window` | `0.3` | Finestra (in secondi) del filtro Savitzky-Golay per attenuare il jitter di tracking |
| `--no-smoothing` | *(disattivato)* | Usa gli angoli grezzi (solo interpolati), senza smoothing |

## Come registrare il video

Per una misura affidabile dei punti morti della pedalata:

- **framerate**: almeno 60 fps (uno smartphone moderno in modalità "60 fps" va bene; 30 fps rischia di perdere il fondo corsa esatto)
- **risoluzione**: 1080p è sufficiente e più leggera da elaborare del 4K
- **inquadratura**: piano laterale fisso, ciclista interamente nel frame, buona luce, sfondo poco affollato per aiutare il tracking dei keypoint

Lo script stampa un avviso se il frame rate rilevato è troppo basso per campionare i cicli di pedalata in modo affidabile.

## Note tecniche

- Il tracking di MediaPipe può avere piccoli errori/jitter frame per frame (specialmente su caviglia e punta del piede); per questo i dati vengono interpolati (per i frame senza keypoint rilevati) e attenuati con un filtro Savitzky-Golay prima del calcolo di ROM e cicli.
- I cicli di pedalata (massimi/minimi dell'angolo del ginocchio) sono rilevati con `scipy.signal.find_peaks`, con vincoli di prominenza e distanza minima per evitare falsi positivi.
- Video, CSV e grafici generati (cartella `output/`) non sono versionati in questo repository.

## Licenza

MIT — vedi [LICENSE](LICENSE).
