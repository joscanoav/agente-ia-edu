# Agente IA-Edu

Recolector de noticias sobre **inteligencia artificial y educación**. Consulta
varias fuentes, filtra por relevancia, puntúa con Groq, extrae el texto real de
cada ítem y genera dos informes Markdown. Opcionalmente los envía por correo.

## Qué hace

1. **Recogida de datos** (`main.py`):
   - **arXiv** — API de arXiv para `cs.CY` y `cs.AI`, filtrando por los términos
     *education*, *learning* y *teaching*.
   - **Google News RSS** — 2 consultas en español y 2 en inglés.
   - **Bing News RSS** — las mismas 4 consultas; el enlace real del artículo se
     extrae del parámetro `url=` y se descarga con trafilatura.
   - **Feeds de `sources.yaml`** — medios de educación y tecnología educativa,
     fuentes de IA y canales de YouTube.
2. **Ventana temporal** de los últimos 7 días y deduplicación por enlace.
3. **Filtro estricto IA + educación**: cada ítem debe contener al menos un término
   de IA y otro de educación (en el título o en los primeros 500 caracteres del
   resumen). Las coincidencias en el título pesan triple.
4. **Deduplicación por título normalizado** (se quita el sufijo de medio, espacios
   y mayúsculas) priorizando fuentes directas y Bing sobre Google News.
5. **Revaluación con Groq**: cada ítem recibe una puntuación de 1 a 5 con un
   motivo; se descartan los menores de 4. Si Groq no está configurado o falla, se
   usa el ranking por palabras clave.
6. **Selección**: máx. 10 ítems por fuente y 60 en total, priorizando las fuentes
   directas frente a Google News.
7. **Extracción del texto real** de cada ítem (artículo con trafilatura, abstract
   completo de arXiv o descripción de YouTube), con limpieza de líneas repetitivas.
8. **Generación de dos informes** en `informes/` (ver más abajo).

## Requisitos

- Python 3.12
- Dependencias en `requirements.txt`: `requests`, `feedparser`, `python-dotenv`,
  `pyyaml`, `groq`, `trafilatura`, `googlenewsdecoder`, `selectolax<1.0`, `tzdata`.

## Instalación

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt   # Windows
.venv/bin/pip install -r requirements.txt       # Linux/macOS
cp .env.example .env                            # y rellena las variables
```

## Ejecución en local

```bash
.venv/Scripts/python main.py     # Windows
.venv/bin/python main.py         # Linux/macOS
```

Al terminar imprime por consola las tablas de ítems por fuente (tras el filtro
estricto y tras la selección), cuántos textos reales se extrajeron por fuente y
un resumen con los ítems con texto real.

Para enviar el informe más reciente por correo:

```bash
.venv/Scripts/python send_email.py
```

## Variables de entorno (`.env`)

Copia `.env.example` a `.env` y rellénalo:

| Variable             | Descripción                                                        |
| -------------------- | ------------------------------------------------------------------ |
| `GROQ_API_KEY`       | Clave de API de Groq para el paso de revaluación de relevancia.    |
| `GROQ_MODEL`         | Modelo de Groq (opcional; por defecto `openai/gpt-oss-120b`).      |
| `GMAIL_USER`         | Cuenta de Gmail remitente (solo para `send_email.py`).             |
| `GMAIL_APP_PASSWORD` | Contraseña de aplicación de Gmail (no la contraseña normal).       |
| `MAIL_TO`            | Destinatarios, separados por comas.                                |

Si falta `GROQ_API_KEY` o el servicio falla, el pipeline continúa usando el
ranking por palabras clave.

## Informes de salida

Se escriben dos archivos en `informes/` con la fecha de Madrid (`Europe/Madrid`):

- **`AAAA-MM-DD.md`** — informe principal. Incluye **solo los ítems con texto
  real**, ordenados por puntuación de Groq, con secciones de Noticias,
  Investigación (arXiv) y Vídeos, más una lista de **Fuentes para NotebookLM**
  (solo URLs reales). Es el archivo que se envía por correo.
- **`AAAA-MM-DD-titulares.md`** — ítems de los que solo se obtuvo el titular
  (`[solo titular]`). **No debe subirse a NotebookLM.**

## Envío por correo (`send_email.py`)

Usa solo la librería estándar (`smtplib`, `ssl`, `email.message`) y
`python-dotenv`. Busca el informe principal más reciente, lo envía por SMTP SSL
(`smtp.gmail.com:465`) con el asunto `Informe IA y Educación - AAAA-MM-DD`, un
cuerpo breve con los ítems por sección (leídos del índice del informe) y el
`.md` como adjunto. Termina con código de error si falta una variable o falla el
envío.

## Workflow de GitHub Actions

`.github/workflows/informe.yml` ejecuta el pipeline automáticamente:

- **Disparadores**: manual (`workflow_dispatch`) y programado (`cron`), todos los
  **sábados a las 05:00 UTC** (`"0 5 * * 6"`).
- Corre en `ubuntu-24.04` con Python 3.12 y caché de pip; `timeout-minutes: 20`.
- Pasos: instala dependencias → `python main.py` → sube `informes/` como
  artefacto (retención 14 días, `if: always()`) → `python send_email.py`.

**Secrets** que hay que configurar en el repositorio:

- `GROQ_API_KEY`
- `GMAIL_USER`
- `GMAIL_APP_PASSWORD`
- `MAIL_TO`

## Añadir fuentes y canales de YouTube (`sources.yaml`)

Lista de feeds con `name`, `url` y `categoria` (`educacion`, `ia` o `youtube`):

```yaml
feeds:
  - name: EdSurge
    url: https://www.edsurge.com/feed
    categoria: educacion
```

Para YouTube se usa el feed Atom con el `channel_id`:

```yaml
  - name: YouTube - Platzi
    url: https://www.youtube.com/feeds/videos.xml?channel_id=UC55-mxUj5Nj3niXFReG44OQ
    categoria: youtube
```

Para obtener el `channel_id` de un canal concreto se descarga
`https://www.youtube.com/@handle` con `requests` (User-Agent de navegador,
`Accept-Language: es`, cookie `SOCS=CAI`) y se busca `channel_id=UC...` o
`"externalId":"UC..."` en el HTML; después conviene comprobar que el feed
`https://www.youtube.com/feeds/videos.xml?channel_id=UC...` devuelve entradas.

## Estructura

```
main.py                        recogida, filtrado, puntuación y generación de informes
send_email.py                  envío del informe por correo (SMTP SSL)
sources.yaml                   feeds RSS/Atom y canales de YouTube
requirements.txt               dependencias
.env.example                   plantilla de variables de entorno
.github/workflows/informe.yml  workflow semanal (GitHub Actions)
informes/                      informes generados
```

## Limitaciones

- **Google News sin texto**: en muchas redes el enlace de Google News redirige a
  `consent.google.com` y no se puede resolver la URL real. Esos ítems quedan como
  `[solo titular]` y van al informe de titulares.
- **Páginas con JavaScript**: el texto se extrae con trafilatura, que solo lee el
  HTML servido; los sitios que cargan el contenido por JavaScript pueden no
  aportar texto y acabar como `[solo titular]`.
- El filtro estricto puede descartar noticias relevantes si no incluyen un término
  explícito de IA o de educación.
