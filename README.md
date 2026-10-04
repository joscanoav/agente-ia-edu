# Agente IA-Edu

Recolector de noticias sobre inteligencia artificial y educación.
Esta primera versión **solo recoge datos**: consulta fuentes, filtra los
últimos 7 días, elimina duplicados y muestra cuántos ítems salió de cada
fuente. La generación del informe con LLM se añadirá después.

## Requisitos

- Python 3.10+
- Dependencias: `requests`, `feedparser`, `python-dotenv`, `pyyaml`, `groq`

## Instalación

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt   # Windows
.venv/bin/pip install -r requirements.txt       # Linux/macOS
cp .env.example .env                           # y rellena GROQ_API_KEY
```

## Uso

```bash
.venv/Scripts/python main.py
```

## Qué hace

1. **arXiv** — API de arXiv para las categorías `cs.CY` y `cs.AI` filtrando
   los términos *education*, *learning* y *teaching*, ordenados por fecha de
   envío.
2. **Google News RSS** — 2 consultas en español (`inteligencia artificial educación`,
   `inteligencia artificial docentes`) y 2 en inglés (`artificial intelligence education`,
   `AI teaching learning`).
3. **Feeds de `sources.yaml`** — todos los feeds RSS/Atom listados en el archivo
   (educación, IA y un canal de YouTube).

Después:

- Filtra los items de los **últimos 7 días**.
- Elimina **duplicados** por enlace.
- Imprime por pantalla **cuántos ítems salieron de cada fuente**.
- Si una fuente falla, la registra y **continúa** con el resto.

## Configuración

### `.env`

```
GROQ_MODEL=openai/gpt-oss-120b
GROQ_API_KEY=
```

`GROQ_MODEL` indica el modelo a usar cuando se añada el paso de LLM.

### `sources.yaml`

Lista de feeds con `name`, `url` y `categoria`:

```yaml
feeds:
  - name: EdSurge
    url: https://www.edsurge.com/feed
    categoria: educacion
```

Para añadir un canal de YouTube se usa el feed Atom con su `channel_id`:

```yaml
  - name: YouTube - TED Talks
    url: https://www.youtube.com/feeds/videos.xml?channel_id=UCAuUUnT6oDeKwE6v1NGQxug
    categoria: youtube
```

## Salida

Los informes se guardarán en `informes/` con el nombre `AAAA-MM-DD.md`
(p. ej. `informes/2026-10-05.md`). Todavía no se generan.

## Estructura

```
main.py         recogida de datos
sources.yaml    feeds RSS
requirements.txt
.env.example
```