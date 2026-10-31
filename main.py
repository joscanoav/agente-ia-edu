"""Recogida de datos sobre IA y educación.

Fuentes: arXiv (cs.CY, cs.AI), Google News RSS y Bing News RSS
(2 consultas ES + 2 EN en cada uno) y todos los feeds definidos en sources.yaml.

Filtra los últimos 7 días, elimina duplicados, aplica un filtro estricto de
relevancia (IA + educación), reevalúa con Groq (descarta los menores de 3),
selecciona los 60 mejores (máx. 10 por fuente, priorizando las fuentes directas)
y descarga el contenido real de cada uno.

Genera dos informes Markdown en `informes/`: `AAAA-MM-DD.md` con los ítems que
tienen texto real (artículo extraído, abstract de arXiv o descripción de YouTube),
ordenados por puntuación de Groq y destinados a NotebookLM; y
`AAAA-MM-DD-titulares.md` con los ítems de los que solo se obtuvo el titular
(no debe subirse a NotebookLM).

Si Groq no está configurado o falla por completo, se usa el ranking por
palabras clave para que los informes se generen igualmente.
"""

from __future__ import annotations

import difflib
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from zoneinfo import ZoneInfo

import feedparser
import requests
import trafilatura
import yaml
from dotenv import load_dotenv

try:  # Se aísla el import: si falla, se usa el resumen del feed.
    from googlenewsdecoder import gnewsdecoder
except Exception:  # noqa: BLE001
    gnewsdecoder = None

try:  # Se aísla el import: si falla, se usa el ranking por palabras clave.
    from groq import Groq, RateLimitError
except Exception:  # noqa: BLE001
    Groq = None
    RateLimitError = Exception

BASE_DIR = Path(__file__).resolve().parent
SOURCES_FILE = BASE_DIR / "sources.yaml"
OUTPUT_DIR = BASE_DIR / "informes"

DAYS_WINDOW = 7
MAX_ITEMS = 60
MAX_POR_FUENTE = 10
MAX_HILOS = 5
MAX_CARACTERES = 2000
MAX_RESUMEN_PUNTUACION = 500
TIMEOUT = 15
ZONA = ZoneInfo("Europe/Madrid")

# Reintentos de las peticiones HTTP ante 429 y 5xx.
PEDIR_MAX_INTENTOS = 3
PEDIR_ESPERA_BASE = 2.0

# Paso de relevancia con Groq
GROQ_MODEL_POR_DEFECTO = "openai/gpt-oss-120b"
GROQ_LOTE = 15
GROQ_MINIMO = 4
GROQ_MAX_REINTENTOS = 3
GROQ_ESPERA_ENTRE_LOTES = 2.0
GROQ_ESPERA_BASE = 2.0
MAX_ESPERA_RATE_LIMIT = 120.0

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

ARXIV_API = "https://export.arxiv.org/api/query"
ARXIV_CATEGORIES = ["cs.CY", "cs.AI"]
ARXIV_TERMS = ["education", "learning", "teaching"]
ARXIV_MAX_RESULTS = 40

BING_NEWS_URL = "https://www.bing.com/news/search"

GOOGLE_NEWS_QUERIES = [
    ("es", "inteligencia artificial educación"),
    ("es", "inteligencia artificial docentes"),
    ("en", "artificial intelligence education"),
    ("en", "AI teaching learning"),
]

# Pon a False para desactivar por completo la fuente Google News.
USAR_GOOGLE_NEWS = True

# ---------------------------------------------------------------------------
# Filtro estricto de relevancia (expresiones regulares con límites de palabra)
# ---------------------------------------------------------------------------
# "AI" e "IA" solo en mayúsculas y como palabra completa.
RE_IA_MAYUSCULAS = re.compile(r"\b(?:AI|IA)\b")
# Resto de términos de IA, sin distinguir mayúsculas.
RE_IA = re.compile(
    r"\b(?:artificial intelligence|inteligencia artificial|chatgpt|llm|"
    r"generative ai|ia generativa|machine learning)\b",
    re.IGNORECASE,
)
# Términos de educación, sin distinguir mayúsculas. No se usa "learning" suelto.
RE_EDUCACION = re.compile(
    r"\b(?:education|educación|teachers?|teaching|students?|classroom|"
    r"schools?|k-12|university|higher ed|docentes?|profesores?|alumnos?|"
    r"estudiantes?|escuela|aula|universidad)\b",
    re.IGNORECASE,
)

CATEGORIA_ARXIV = "arxiv"
CATEGORIA_GOOGLE_NEWS = "google_news"
CATEGORIA_BING_NEWS = "bing_news"
CATEGORIA_YOUTUBE = "youtube"


@dataclass
class Item:
    titulo: str
    enlace: str
    fecha: datetime | None
    fuente: str
    categoria: str
    resumen: str = ""
    texto: str = ""
    enlace_real: str = ""
    puntuacion: int = 0
    puntuacion_groq: int | None = None
    motivo: str = ""
    solo_titular: bool = False
    verificacion: dict | None = None


@dataclass
class Resultado:
    fuente: str
    items: list[Item]
    error: str | None = None

    @property
    def total(self) -> int:
        return len(self.items)


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------


def cargar_env() -> None:
    load_dotenv(BASE_DIR / ".env")


def sesion_http() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def a_datetime(estructura) -> datetime | None:
    """Convierte un time.struct_time de feedparser a datetime con timezone."""
    if not estructura:
        return None
    try:
        return datetime(*estructura[:6], tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def limpiar_html(texto: str) -> str:
    """Elimina etiquetas HTML y normaliza espacios."""
    if not texto:
        return ""
    texto = re.sub(r"<[^>]+>", " ", texto)
    texto = texto.replace("&nbsp;", " ")
    return re.sub(r"\s+", " ", texto).strip()


def recortar_texto(texto: str, max_len: int = MAX_CARACTERES) -> str:
    texto = (texto or "").strip()
    if len(texto) <= max_len:
        return texto
    return texto[:max_len].rstrip()


_ABREVIATURAS_FIN_FRASE = {
    "abr",
    "ago",
    "aprox",
    "art",
    "av",
    "cap",
    "cf",
    "d",
    "dic",
    "dr",
    "dra",
    "ej",
    "ene",
    "etc",
    "feb",
    "fig",
    "figs",
    "jul",
    "jun",
    "mar",
    "may",
    "mr",
    "mrs",
    "ms",
    "no",
    "nov",
    "num",
    "oct",
    "p",
    "pp",
    "sept",
    "sep",
    "sr",
    "sra",
    "srta",
    "st",
    "tel",
    "ud",
    "uds",
    "vds",
    "vol",
    "vs",
}


def _termina_en_abreviatura(fragmento: str) -> bool:
    texto = (fragmento or "").rstrip().strip("\"'”’»")
    if not texto.endswith("."):
        return False
    if re.search(r"(?:\b[a-záéíóúüñ]\.){2,}$", texto, re.IGNORECASE):
        return True
    coincidencia = re.search(r"([a-záéíóúüñ]+)\.$", texto, re.IGNORECASE)
    if coincidencia:
        return coincidencia.group(1).lower() in _ABREVIATURAS_FIN_FRASE
    return bool(re.search(r"\b[a-záéíóúüñ]\.$", texto, re.IGNORECASE))


def dividir_frases(texto: str) -> list[str]:
    """Divide en frases sin cortar tras abreviaturas como Oct., U.S. o Dr."""
    if not texto:
        return []
    partes = re.split(r"(?<=[.!?])\s+", texto)
    frases: list[str] = []
    for parte in partes:
        if frases and _termina_en_abreviatura(frases[-1]):
            frases[-1] = frases[-1].rstrip() + " " + parte.lstrip()
        else:
            frases.append(parte)
    return frases


def _ultimo_fin_frase(texto: str) -> int | None:
    ultimo: int | None = None
    for coincidencia in re.finditer(r"[.!?](?=\s|$)", texto):
        if coincidencia.group() == "." and _termina_en_abreviatura(
            texto[: coincidencia.end()]
        ):
            continue
        ultimo = coincidencia.end()
    return ultimo


def recortar_en_frase(texto: str, max_len: int = MAX_CARACTERES) -> str:
    """Recorta en el último punto final anterior al límite (no a mitad de frase)."""
    texto = (texto or "").strip()
    if len(texto) <= max_len:
        return texto
    fragmento = texto[:max_len]
    ultimo = _ultimo_fin_frase(fragmento)
    if ultimo:
        return fragmento[:ultimo].strip()
    return fragmento.strip()


def limpiar_youtube(texto: str) -> str:
    """Quita las URLs y deja las 3 primeras frases de la descripción."""
    texto = re.sub(r"https?://\S+", " ", limpiar_html(texto))
    texto = re.sub(r"\s+", " ", texto).strip()
    frases = dividir_frases(texto)
    return " ".join(frases[:3]).strip()


# Líneas de cabecera o pie que se repiten y no aportan contenido.
PREFIJOS_BASURA = (
    "did you know search engines",
    "key points:",
    "get stories like this",
    "sign up for",
    "lectura",
    "escuchar esta nota",
)


def normalizar_titulo(titulo: str) -> str:
    """Normaliza un titular: quita el sufijo de medio, espacios y mayúsculas.

    "IA en el aula - El País" y "IA EN EL AULA | El País" comparten clave.
    """
    texto = limpiar_html(titulo or "")
    partes = re.split(r"\s+[-–—|]\s+", texto)
    if len(partes) > 1:
        texto = partes[0]
    texto = texto.lower()
    texto = re.sub(r"[^0-9a-záéíóúüñ]+", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()


def limpiar_texto(texto: str, titulo: str) -> str:
    """Elimina líneas repetitivas de cabecera/pie y las que repiten el título."""
    if not texto:
        return ""
    titulo_norm = normalizar_titulo(titulo)
    limpias: list[str] = []
    for linea in texto.splitlines():
        desnuda = linea.strip()
        if not desnuda:
            limpias.append("")
            continue
        bajo = desnuda.lower()
        if any(bajo.startswith(prefijo) for prefijo in PREFIJOS_BASURA):
            continue
        if titulo_norm and normalizar_titulo(desnuda) == titulo_norm:
            continue
        limpias.append(desnuda)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(limpias)).strip()


def _prioridad_fuente(item: Item) -> int:
    """Mayor prioridad para fuentes directas y Bing; menor para Google News."""
    return 0 if item.categoria == CATEGORIA_GOOGLE_NEWS else 1


def deduplicar_por_titulo(items: list[Item]) -> list[Item]:
    """Une ítems con titulares equivalentes, priorizando la fuente directa/Bing."""
    unicos: list[Item] = []
    claves: list[str] = []
    for item in items:
        clave = normalizar_titulo(item.titulo) or (item.enlace or item.titulo).lower()
        reemplazado = False
        for indice, existente in enumerate(unicos):
            if clave == claves[indice] or difflib.SequenceMatcher(
                None, clave, claves[indice]
            ).ratio() > 0.9:
                if _prioridad_fuente(item) > _prioridad_fuente(existente):
                    unicos[indice] = item
                    claves[indice] = clave
                reemplazado = True
                break
        if not reemplazado:
            unicos.append(item)
            claves.append(clave)
    return unicos


def deduplicar_videos(items: list[Item]) -> list[Item]:
    """Si dos vídeos tienen la misma descripción, conserva solo uno."""
    vistos: set[str] = set()
    resultado: list[Item] = []
    for item in items:
        if item.categoria != CATEGORIA_YOUTUBE:
            resultado.append(item)
            continue
        clave = limpiar_html(item.texto or item.resumen).strip().lower()
        if clave and clave in vistos:
            continue
        if clave:
            vistos.add(clave)
        resultado.append(item)
    return resultado


def parsear_feed(contenido: bytes, fuente: str, categoria: str) -> list[Item]:
    """Convierte un XML de feed en Items con fecha timezone-aware."""
    datos = feedparser.parse(contenido)
    items: list[Item] = []
    for entrada in datos.entries:
        fecha = a_datetime(
            getattr(entrada, "published_parsed", None)
            or getattr(entrada, "updated_parsed", None)
        )
        resumen = (
            getattr(entrada, "summary", "")
            or getattr(entrada, "media_description", "")
            or ""
        ).strip()
        items.append(
            Item(
                titulo=(getattr(entrada, "title", "") or "").strip(),
                enlace=(getattr(entrada, "link", "") or "").strip(),
                fecha=fecha,
                fuente=fuente,
                categoria=categoria,
                resumen=resumen,
            )
        )
    return items


def dentro_de_ventana(item: Item, limite: datetime) -> bool:
    """True si el item no tiene fecha o es posterior al límite."""
    if item.fecha is None:
        return True
    return item.fecha >= limite


def _espera_indicada_respuesta(respuesta: requests.Response) -> float | None:
    valor = respuesta.headers.get("Retry-After")
    if not valor:
        return None
    try:
        return max(0.0, min(MAX_ESPERA_RATE_LIMIT, float(str(valor).strip())))
    except ValueError:
        return None


def _espera_indicada_error(error: Exception) -> float | None:
    respuesta = getattr(error, "response", None)
    if respuesta is not None:
        espera = _espera_indicada_respuesta(respuesta)
        if espera is not None:
            return espera
    texto = str(error)
    patrones = (
        r"retry[- ]after[^\d]{0,20}(\d+(?:[.,]\d+)?)\s*(ms|s|m)?",
        r"(?:try again|intenta(?:r)? de nuevo|reintenta)[^\d]{0,40}"
        r"(\d+(?:[.,]\d+)?)\s*(ms|s|m)?",
        r"(?:in|en)\s+(\d+(?:[.,]\d+)?)\s*(ms|s|segundos?|m|minutos?)",
    )
    for patron in patrones:
        coincidencia = re.search(patron, texto, re.IGNORECASE)
        if not coincidencia:
            continue
        try:
            cantidad = float(coincidencia.group(1).replace(",", "."))
        except ValueError:
            continue
        unidad = (coincidencia.group(2) or "s").lower()
        if unidad == "ms":
            cantidad /= 1000.0
        elif unidad in ("m", "minuto", "minutos"):
            cantidad *= 60.0
        return max(0.0, min(MAX_ESPERA_RATE_LIMIT, cantidad))
    return None


def pedir(session: requests.Session, url: str, params: dict | None = None) -> bytes:
    """GET con reintentos y espera creciente ante errores 429 y 5xx."""
    demora = PEDIR_ESPERA_BASE
    for intento in range(1, PEDIR_MAX_INTENTOS + 1):
        respuesta = session.get(url, params=params, timeout=TIMEOUT)
        if (respuesta.status_code == 429 or respuesta.status_code >= 500) and (
            intento < PEDIR_MAX_INTENTOS
        ):
            espera = demora
            if respuesta.status_code == 429:
                indicada = _espera_indicada_respuesta(respuesta)
                if indicada is not None:
                    espera = indicada
            time.sleep(espera)
            demora *= 2
            continue
        respuesta.raise_for_status()
        return respuesta.content
    raise RuntimeError("No se pudo completar la petición")  # pragma: no cover


def contar_coincidencias(texto: str) -> tuple[int, int]:
    """Devuelve (coincidencias de IA, coincidencias de educación)."""
    ia = len(RE_IA_MAYUSCULAS.findall(texto)) + len(RE_IA.findall(texto))
    edu = len(RE_EDUCACION.findall(texto))
    return ia, edu


def evaluar_relevancia(item: Item) -> int:
    """Puntúa solo con el título y los primeros 500 caracteres del resumen.

    Las coincidencias en el título pesan x3. Devuelve 0 si en ese fragmento
    no hay al menos un término de IA y otro de educación.
    """
    titulo = limpiar_html(item.titulo)
    resumen = recortar_texto(limpiar_html(item.resumen), MAX_RESUMEN_PUNTUACION)
    ia_titulo, edu_titulo = contar_coincidencias(titulo)
    ia_resumen, edu_resumen = contar_coincidencias(resumen)
    coincidencias_ia = ia_titulo * 3 + ia_resumen
    coincidencias_edu = edu_titulo * 3 + edu_resumen
    if coincidencias_ia > 0 and coincidencias_edu > 0:
        return coincidencias_ia + coincidencias_edu
    return 0


# ---------------------------------------------------------------------------
# Paso de relevancia con Groq
# ---------------------------------------------------------------------------


def clave_puntuacion(item: Item) -> int:
    """Puntuación usada para ordenar: la de Groq si existe, si no la de palabras."""
    if item.puntuacion_groq is not None:
        return item.puntuacion_groq
    return item.puntuacion


def config_groq() -> tuple[str, str]:
    clave = os.environ.get("GROQ_API_KEY", "").strip()
    modelo = os.environ.get("GROQ_MODEL", "").strip() or GROQ_MODEL_POR_DEFECTO
    return clave, modelo


def _prompt_lote(lote: list[tuple[int, Item]]) -> str:
    cuerpo = [
        {
            "id": indice,
            "titulo": item.titulo,
            "resumen": recortar_texto(limpiar_html(item.resumen), 300),
        }
        for indice, item in lote
    ]
    return (
        'Evalúa de 1 a 5 la relevancia de cada ítem respecto a '
        '"IA aplicada a la educación" (5 = muy relevante, 1 = nada relevante).\n'
        "Devuelve SOLO un JSON con la clave \"resultados\": una lista de objetos "
        "con las claves \"id\", \"puntuacion\" (entero de 1 a 5) y \"motivo\" "
        "(una sola línea).\n"
        "Ítems:\n" + json.dumps(cuerpo, ensure_ascii=False)
    )


def _evaluar_lote(cliente, modelo: str, lote: list[tuple[int, Item]]) -> dict[int, tuple[int, str]]:
    respuesta = cliente.chat.completions.create(
        model=modelo,
        messages=[
            {
                "role": "system",
                "content": (
                    "Eres un evaluador experto en inteligencia artificial y "
                    "educación. Respondes únicamente con JSON válido."
                ),
            },
            {"role": "user", "content": _prompt_lote(lote)},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    contenido = respuesta.choices[0].message.content or "{}"
    datos = json.loads(contenido)

    pares: dict[int, tuple[int, str]] = {}
    for fila in datos.get("resultados", []) or []:
        try:
            indice = int(fila["id"])
            puntuacion = int(fila["puntuacion"])
        except (KeyError, TypeError, ValueError):
            continue
        motivo = str(fila.get("motivo", "")).strip()
        pares[indice] = (max(1, min(5, puntuacion)), motivo)
    return pares


def _evaluar_lote_con_reintentos(cliente, modelo: str, lote: list[tuple[int, Item]]):
    espera = GROQ_ESPERA_BASE
    for intento in range(GROQ_MAX_REINTENTOS + 1):
        try:
            return _evaluar_lote(cliente, modelo, lote)
        except RateLimitError as exc:
            if intento >= GROQ_MAX_REINTENTOS:
                print("     429: se agotaron los reintentos")
                return None
            indicada = _espera_indicada_error(exc)
            if indicada is not None:
                espera = indicada
            print(
                f"     429, reintento {intento + 1}/{GROQ_MAX_REINTENTOS} "
                f"en {espera:.1f}s"
            )
        except Exception as exc:  # noqa: BLE001
            if intento >= GROQ_MAX_REINTENTOS:
                print(f"     {type(exc).__name__}: se agotaron los reintentos")
                return None
            print(
                f"     {type(exc).__name__}, reintento "
                f"{intento + 1}/{GROQ_MAX_REINTENTOS} en {espera:.1f}s"
            )
        time.sleep(espera)
        espera *= 2
    return None


def evaluar_con_groq(items: list[Item]) -> dict[int, tuple[int, str]] | None:
    """Puntúa los ítems con Groq en lotes. None si Groq no está disponible."""
    if Groq is None:
        return None
    clave, modelo = config_groq()
    if not clave:
        return None

    indexados = list(enumerate(items))
    lotes = [
        indexados[i : i + GROQ_LOTE] for i in range(0, len(indexados), GROQ_LOTE)
    ]
    cliente = Groq(api_key=clave)
    resultados: dict[int, tuple[int, str]] = {}
    for numero, lote in enumerate(lotes, 1):
        print(f"  Groq: lote {numero}/{len(lotes)} ({len(lote)} ítems)")
        pares = _evaluar_lote_con_reintentos(cliente, modelo, lote)
        if pares:
            resultados.update(pares)
        if numero < len(lotes):
            time.sleep(GROQ_ESPERA_ENTRE_LOTES)
    return resultados or None


def aplicar_groq(items: list[Item]) -> list[Item]:
    """Reevalúa con Groq y descarta los ítems con puntuación menor que 3.

    Si Groq falla por completo, devuelve los ítems sin cambios (ranking por
    palabras clave).
    """
    if not items:
        return items
    clave, _ = config_groq()
    if Groq is None or not clave:
        print("  Groq no configurado: ranking por palabras clave.")
        return items

    resultados = evaluar_con_groq(items)
    if resultados is None:
        print("  Groq no disponible: ranking por palabras clave.")
        return items

    descartados = 0
    conservados: list[Item] = []
    for indice, item in enumerate(items):
        par = resultados.get(indice)
        if par is None:
            # Lote sin respuesta: se conserva con puntuación neutra.
            item.puntuacion_groq = 3
            item.motivo = "(sin evaluar)"
            conservados.append(item)
            continue
        puntuacion, motivo = par
        item.puntuacion_groq = puntuacion
        item.motivo = motivo
        if puntuacion < GROQ_MINIMO:
            descartados += 1
            continue
        conservados.append(item)

    print(f"  Groq: {descartados} ítems descartados (< {GROQ_MINIMO})")
    return conservados


# ---------------------------------------------------------------------------
# Fuentes concretas
# ---------------------------------------------------------------------------


def obtener_arxiv(session: requests.Session, categoria: str) -> Resultado:
    nombre = f"arXiv ({categoria})"
    try:
        terminos = " OR ".join(f"all:{t}" for t in ARXIV_TERMS)
        contenido = pedir(
            session,
            ARXIV_API,
            {
                "search_query": f"cat:{categoria} AND ({terminos})",
                "start": 0,
                "max_results": ARXIV_MAX_RESULTS,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            },
        )
        items = parsear_feed(contenido, nombre, CATEGORIA_ARXIV)
        return Resultado(nombre, items, None if items else "sin resultados")
    except Exception as exc:  # noqa: BLE001 - se continúa con el resto
        return Resultado(nombre, [], f"{type(exc).__name__}: {exc}")


def obtener_google_news(session: requests.Session, idioma: str, consulta: str) -> Resultado:
    nombre = f"Google News [{idioma}] {consulta}"
    try:
        if idioma == "es":
            params = {"hl": "es", "gl": "ES", "ceid": "ES:es"}
        else:
            params = {"hl": "en-US", "gl": "US", "ceid": "US:en"}
        contenido = pedir(
            session,
            "https://news.google.com/rss/search",
            params | {"q": consulta},
        )
        items = parsear_feed(contenido, nombre, CATEGORIA_GOOGLE_NEWS)
        return Resultado(nombre, items, None if items else "sin resultados")
    except Exception as exc:  # noqa: BLE001
        return Resultado(nombre, [], f"{type(exc).__name__}: {exc}")


def obtener_bing_news(session: requests.Session, idioma: str, consulta: str) -> Resultado:
    nombre = f"Bing News [{idioma}] {consulta}"
    try:
        if idioma == "es":
            params = {"q": consulta, "format": "RSS", "setmkt": "es-ES", "setlang": "es"}
        else:
            params = {"q": consulta, "format": "RSS", "setmkt": "en-US", "setlang": "en"}
        contenido = pedir(session, BING_NEWS_URL, params)
        items = parsear_feed(contenido, nombre, CATEGORIA_BING_NEWS)
        for item in items:
            item.enlace = extraer_url_real_bing(item.enlace)
        return Resultado(nombre, items, None if items else "sin resultados")
    except Exception as exc:  # noqa: BLE001
        return Resultado(nombre, [], f"{type(exc).__name__}: {exc}")


def cargar_sources() -> list[dict]:
    if not SOURCES_FILE.exists():
        raise FileNotFoundError(f"No existe {SOURCES_FILE}")
    with open(SOURCES_FILE, "r", encoding="utf-8") as fh:
        datos = yaml.safe_load(fh) or {}
    feeds = datos.get("feeds") or []
    return [f for f in feeds if f.get("url")]


def obtener_feed(session: requests.Session, feed: dict) -> Resultado:
    nombre = feed.get("name") or feed["url"]
    categoria = feed.get("categoria") or "noticias"
    try:
        contenido = pedir(session, feed["url"])
        items = parsear_feed(contenido, nombre, categoria)
        return Resultado(nombre, items, None if items else "sin resultados")
    except Exception as exc:  # noqa: BLE001
        return Resultado(nombre, [], f"{type(exc).__name__}: {exc}")


def _informar(resultado: Resultado) -> Resultado:
    sufijo = f"  ({resultado.error})" if resultado.error else ""
    print(f"  {resultado.fuente:<44} -> {resultado.total} items{sufijo}")
    return resultado


# ---------------------------------------------------------------------------
# Contenido real
# ---------------------------------------------------------------------------


def resolver_enlace_google_news(enlace: str) -> str | None:
    """Resuelve el enlace real de Google News con googlenewsdecoder."""
    if gnewsdecoder is None:
        return None
    try:
        resultado = gnewsdecoder(enlace, interval=1)
    except Exception:  # noqa: BLE001
        return None
    if isinstance(resultado, dict) and resultado.get("success"):
        return resultado.get("decoded_url")
    return None


def extraer_url_real_bing(enlace: str) -> str:
    """Extrae la URL real del parámetro `url=` de un enlace de Bing News."""
    if not enlace:
        return enlace
    valores = parse_qs(urlparse(enlace).query).get("url")
    if valores:
        return valores[0]
    return enlace


def descargar_texto(enlace: str) -> str | None:
    """Descarga la página y extrae el texto principal con trafilatura."""
    respuesta = requests.get(
        enlace, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT
    )
    respuesta.raise_for_status()
    return trafilatura.extract(
        respuesta.text, include_comments=False, include_tables=False
    )


def procesar_item(item: Item) -> bool:
    """Rellena `enlace_real`, `texto` y `solo_titular`.

    Devuelve True si obtuvo contenido real (no solo el titular).
    """
    item.enlace_real = item.enlace
    item.texto = ""
    item.solo_titular = True

    if item.categoria == CATEGORIA_ARXIV:
        # arXiv: abstract completo.
        item.texto = limpiar_html(item.resumen)
    elif item.categoria == CATEGORIA_YOUTUBE:
        # YouTube: descripción sin URLs, 3 primeras frases.
        item.texto = limpiar_youtube(item.resumen)
    else:
        enlace = item.enlace
        if item.categoria == CATEGORIA_GOOGLE_NEWS:
            real = resolver_enlace_google_news(enlace)
            if real:
                item.enlace_real = real
                enlace = real
            else:
                enlace = ""
        if enlace:
            try:
                texto = descargar_texto(enlace)
            except Exception:  # noqa: BLE001
                texto = None
            if texto:
                item.texto = recortar_en_frase(texto)

    if item.texto:
        item.texto = limpiar_texto(item.texto, item.titulo)

    if item.texto:
        item.solo_titular = False
        return True

    # Respaldo: el resumen del feed. El ítem queda marcado como solo titular.
    item.texto = recortar_texto(limpiar_html(item.resumen))
    return False


def enriquecer(items: list[Item]) -> int:
    """Descarga el contenido real de cada ítem (máx. 5 hilos)."""
    if not items:
        return 0
    exitos = 0
    with ThreadPoolExecutor(max_workers=MAX_HILOS) as pool:
        futuros = [pool.submit(procesar_item, item) for item in items]
        for futuro in as_completed(futuros):
            try:
                if futuro.result():
                    exitos += 1
            except Exception:  # noqa: BLE001
                continue
    return exitos


# ---------------------------------------------------------------------------
# Orquestación
# ---------------------------------------------------------------------------


def imprimir_tabla_fuentes(items: list[Item], etiqueta: str) -> None:
    """Imprime una tabla con el número de ítems por fuente."""
    conteo: dict[str, int] = {}
    for item in items:
        conteo[item.fuente] = conteo.get(item.fuente, 0) + 1
    print(f"\n  == {etiqueta} ==")
    if not conteo:
        print("  (sin ítems)")
        return
    ancho = max(len(fuente) for fuente in conteo)
    for fuente, numero in sorted(conteo.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {fuente:<{ancho}} {numero:>3}")


def imprimir_tabla_textos(items: list[Item], etiqueta: str) -> None:
    """Imprime, por fuente, cuántos textos reales se extrajeron del total."""
    conteo: dict[str, list[int]] = {}
    for item in items:
        real, total = conteo.get(item.fuente, [0, 0])
        conteo[item.fuente] = [real + (0 if item.solo_titular else 1), total + 1]
    print(f"\n  == {etiqueta} ==")
    if not conteo:
        print("  (sin ítems)")
        return
    ancho = max(len(fuente) for fuente in conteo)
    for fuente, (real, total) in sorted(
        conteo.items(), key=lambda kv: (-kv[1][0], kv[0])
    ):
        print(f"  {fuente:<{ancho}} {real:>3}/{total:<3} con texto real")


def seleccionar(unicos: list[Item]) -> list[Item]:
    """Ordena por puntuación y fecha y aplica los topes por fuente y total.

    Prioriza las fuentes directas (no Google News) frente a Google News.
    """
    sin_fecha = datetime.min.replace(tzinfo=timezone.utc)
    unicos.sort(key=lambda i: (clave_puntuacion(i), i.fecha or sin_fecha), reverse=True)
    # Reordenación estable: las fuentes directas pasan delante de Google News.
    unicos.sort(key=lambda i: i.categoria == CATEGORIA_GOOGLE_NEWS)

    seleccionados: list[Item] = []
    conteo: dict[str, int] = {}
    for item in unicos:
        if len(seleccionados) >= MAX_ITEMS:
            break
        if conteo.get(item.fuente, 0) >= MAX_POR_FUENTE:
            continue
        conteo[item.fuente] = conteo.get(item.fuente, 0) + 1
        seleccionados.append(item)

    # Orden final del informe: puntuación y fecha (descendente).
    seleccionados.sort(
        key=lambda i: (clave_puntuacion(i), i.fecha or sin_fecha), reverse=True
    )
    return seleccionados


def recoger() -> tuple[list[Item], list[Resultado]]:
    limite = datetime.now(timezone.utc) - timedelta(days=DAYS_WINDOW)
    resultados: list[Resultado] = []
    session = sesion_http()

    print("== Recogiendo datos de las ultimas 7 dias ==\n")

    print("[1/4] arXiv")
    for categoria in ARXIV_CATEGORIES:
        resultados.append(_informar(obtener_arxiv(session, categoria)))

    print("\n[2/4] Google News RSS")
    if USAR_GOOGLE_NEWS:
        for idioma, consulta in GOOGLE_NEWS_QUERIES:
            resultados.append(_informar(obtener_google_news(session, idioma, consulta)))
    else:
        print("  (desactivado con USAR_GOOGLE_NEWS = False)")

    print("\n[3/4] Bing News RSS")
    for idioma, consulta in GOOGLE_NEWS_QUERIES:
        resultados.append(_informar(obtener_bing_news(session, idioma, consulta)))

    print("\n[4/4] Feeds de sources.yaml")
    for feed in cargar_sources():
        resultados.append(_informar(obtener_feed(session, feed)))

    # Aplanar, aplicar ventana temporal y dedupe básico por enlace.
    candidatos: list[Item] = []
    vistos: set[str] = set()
    for res in resultados:
        for item in res.items:
            if not dentro_de_ventana(item, limite):
                continue
            clave = (item.enlace or item.titulo).strip().lower()
            if not clave or clave in vistos:
                continue
            vistos.add(clave)
            candidatos.append(item)
    print(f"\n  Tras ventana temporal (7d): {len(candidatos)} ítems")

    # Filtro estricto: cada ítem debe tener un término de IA y uno de educación.
    relevantes: list[Item] = []
    for item in candidatos:
        puntuacion = evaluar_relevancia(item)
        if puntuacion > 0:
            item.puntuacion = puntuacion
            relevantes.append(item)
    print(f"  Tras filtro estricto (IA + educación): {len(relevantes)} ítems")
    imprimir_tabla_fuentes(relevantes, "Ítems por fuente (tras filtro estricto)")

    # Dedupe por título normalizado (y casi idéntico), priorizando directas/Bing.
    unicos = deduplicar_por_titulo(relevantes)
    print(f"\n  Tras dedupe (título normalizado): {len(unicos)} ítems")

    # Paso de relevancia con Groq (descarta los < GROQ_MINIMO). Si falla, palabras clave.
    unicos = aplicar_groq(unicos)

    # Ordenar, aplicar topes (10 por fuente, 60 en total) y priorizar directas.
    seleccionados = seleccionar(unicos)
    print(
        f"  Seleccionados (máx. {MAX_POR_FUENTE}/fuente, {MAX_ITEMS} en total): "
        f"{len(seleccionados)} ítems"
    )
    imprimir_tabla_fuentes(seleccionados, "Ítems por fuente (selección final)")
    return seleccionados, resultados


# ---------------------------------------------------------------------------
# Informe
# ---------------------------------------------------------------------------

ETIQUETA_RESPALDADO = "✅ respaldado"
ETIQUETA_PARCIAL = "⚠️ parcial"
ETIQUETA_CONTRADICHO = "❌ contradicho"
ETIQUETA_NO_VERIFICADO = "❔ no verificado"
ETIQUETA_LIMITE_API = "⏳ no se pudo verificar (límite de la API)"

ETIQUETAS_VERIFICACION = (
    ETIQUETA_RESPALDADO,
    ETIQUETA_PARCIAL,
    ETIQUETA_CONTRADICHO,
    ETIQUETA_NO_VERIFICADO,
    ETIQUETA_LIMITE_API,
)

ETIQUETAS_POR_RESPALDO = {
    "respaldado": ETIQUETA_RESPALDADO,
    "parcial": ETIQUETA_PARCIAL,
    "contradicho": ETIQUETA_CONTRADICHO,
    "no verificado": ETIQUETA_NO_VERIFICADO,
    "limite_api": ETIQUETA_LIMITE_API,
}

PRIORIDAD_ETIQUETA = {
    ETIQUETA_RESPALDADO: 0,
    ETIQUETA_PARCIAL: 1,
    ETIQUETA_LIMITE_API: 2,
    ETIQUETA_NO_VERIFICADO: 3,
    ETIQUETA_CONTRADICHO: 4,
}

LEYENDA_ETIQUETAS = (
    (
        ETIQUETA_RESPALDADO,
        "la fuente primaria contiene la afirmación y todas sus cifras.",
    ),
    (
        ETIQUETA_PARCIAL,
        "la fuente respalda solo una parte de la afirmación.",
    ),
    (
        ETIQUETA_CONTRADICHO,
        "la fuente contradice directamente el núcleo de la afirmación.",
    ),
    (
        ETIQUETA_NO_VERIFICADO,
        "no se encontró evidencia suficiente en la fuente.",
    ),
    (
        ETIQUETA_LIMITE_API,
        "la API devolvió error 429 y se agotaron los reintentos; "
        "no es lo mismo que 'no verificado'.",
    ),
)


def formatear_fecha(item: Item) -> str:
    if item.fecha is None:
        return "Sin fecha"
    return item.fecha.astimezone(ZONA).strftime("%Y-%m-%d %H:%M")


def clasificar(items: list[Item]) -> tuple[list[Item], list[Item], list[Item]]:
    noticias: list[Item] = []
    investigacion: list[Item] = []
    videos: list[Item] = []
    for item in items:
        if item.categoria == CATEGORIA_YOUTUBE:
            videos.append(item)
        elif item.categoria == CATEGORIA_ARXIV:
            investigacion.append(item)
        else:
            noticias.append(item)
    return noticias, investigacion, videos


def cuerpo_item(item: Item) -> str:
    texto = (item.texto or "").strip()
    if item.solo_titular:
        if texto and texto.lower() != item.titulo.strip().lower():
            return f"[solo titular] {texto}"
        return "[solo titular]"
    return texto or "[solo titular]"


def formato_puntuacion(item: Item) -> str:
    if item.puntuacion_groq is not None:
        motivo = f" — {item.motivo}" if item.motivo else ""
        return f"{item.puntuacion_groq}/5 (Groq){motivo}"
    return f"{item.puntuacion} (palabras clave)"


def etiqueta_afirmacion(afirmacion: dict) -> str:
    respaldo = str(afirmacion.get("respaldo") or "").strip().lower()
    return ETIQUETAS_POR_RESPALDO.get(respaldo, ETIQUETA_NO_VERIFICADO)


def etiqueta_item(item: Item) -> str:
    verificacion = item.verificacion or {}
    afirmaciones = verificacion.get("afirmaciones") or []
    etiquetas = [
        etiqueta_afirmacion(af)
        for af in afirmaciones
        if isinstance(af, dict)
    ]
    if not etiquetas:
        if verificacion.get("limite_api"):
            return ETIQUETA_LIMITE_API
        return ETIQUETA_NO_VERIFICADO
    return max(etiquetas, key=lambda e: PRIORIDAD_ETIQUETA.get(e, 3))


def contar_etiquetas(items: list[Item]) -> dict[str, int]:
    conteo = {etiqueta: 0 for etiqueta in ETIQUETAS_VERIFICACION}
    for item in items:
        conteo[etiqueta_item(item)] += 1
    return conteo


def _urls_iguales(url_a: str, url_b: str) -> bool:
    if not url_a or not url_b:
        return False

    def normalizar(url: str) -> tuple[str, str, str]:
        parsed = urlparse(url.strip())
        dominio = (parsed.netloc or "").lower()
        if dominio.startswith("www."):
            dominio = dominio[4:]
        ruta = (parsed.path or "").rstrip("/")
        return dominio, ruta, parsed.query or ""

    return normalizar(url_a) == normalizar(url_b)


def render_leyenda_verificacion() -> list[str]:
    lineas = ["## Leyenda de etiquetas", ""]
    for etiqueta, descripcion in LEYENDA_ETIQUETAS:
        lineas.append(f"- {etiqueta}: {descripcion}")
    lineas.append("")
    return lineas


def render_verificacion(item: Item) -> list[str]:
    if item.verificacion is None:
        if item.solo_titular:
            return []
        return [
            f"- **Verificación:** {ETIQUETA_NO_VERIFICADO}",
            "- **Motivo:** No se ejecutó la verificación.",
        ]

    lineas = [f"- **Verificación:** {etiqueta_item(item)}"]
    afirmaciones = item.verificacion.get("afirmaciones") or []
    if not afirmaciones:
        errores = item.verificacion.get("errores") or []
        motivo = (
            str(errores[0]).strip()
            if errores
            else "No se pudo verificar la noticia."
        )
        lineas.append(f"- **Motivo:** {motivo}")
        return lineas

    for af in afirmaciones:
        if not isinstance(af, dict):
            continue
        etiqueta = etiqueta_afirmacion(af)
        texto = str(af.get("texto") or "").strip()
        lineas.append(f"  - {etiqueta}: {texto}" if texto else f"  - {etiqueta}")

        fuente = str(af.get("fuente_primaria_url") or "").strip()
        if fuente:
            enlace_noticia = (item.enlace_real or item.enlace or "").strip()
            if _urls_iguales(fuente, enlace_noticia):
                lineas.append(
                    "    - Fuente primaria: verificado contra el propio artículo"
                )
            else:
                lineas.append(f"    - Fuente primaria: {fuente}")

        evidencia = str(af.get("evidencia") or "").strip()
        if evidencia:
            lineas.append(f'    - Evidencia: "{evidencia}"')

        explicacion = str(af.get("explicacion_parcial") or "").strip()
        if explicacion:
            lineas.append(f"    - Explicación: {explicacion}")

        motivo = str(af.get("motivo_rechazo") or "").strip()
        if motivo:
            lineas.append(f"    - Motivo: {motivo}")

        matices = af.get("matices_omitidos") or []
        if matices:
            lineas.append("    - Matices omitidos:")
            for matiz in matices:
                if not isinstance(matiz, dict):
                    continue
                texto_matiz = str(
                    matiz.get("texto") or matiz.get("fragmento_literal") or ""
                ).strip()
                por_que = str(matiz.get("por_que_matiza") or "").strip()
                if texto_matiz and por_que:
                    lineas.append(f"      - {texto_matiz} ({por_que})")
                elif texto_matiz:
                    lineas.append(f"      - {texto_matiz}")

    return lineas


def render_resumen_verificacion(items: list[Item]) -> list[str]:
    conteo = contar_etiquetas(items)
    lineas = ["## Resumen de verificación", ""]
    for etiqueta in ETIQUETAS_VERIFICACION:
        lineas.append(f"- {etiqueta}: {conteo[etiqueta]}")
    lineas.append(f"- Total: {len(items)}")
    lineas.append("")
    return lineas


def verificar_items(items: list[Item]) -> None:
    if not items:
        return
    try:
        from verificar import verificar_item
    except Exception as exc:  # noqa: BLE001
        for item in items:
            item.verificacion = {
                "afirmaciones": [],
                "errores": [
                    f"No se pudo importar verificar: {type(exc).__name__}: {exc}"
                ],
            }
        return

    total = len(items)
    for indice, item in enumerate(items, 1):
        print(f"  Verificando {indice}/{total}: {item.titulo[:70]}")
        try:
            item.verificacion = verificar_item(item)
        except Exception as exc:  # noqa: BLE001
            item.verificacion = {
                "afirmaciones": [],
                "errores": [f"{type(exc).__name__}: {exc}"],
            }


def render_seccion(titulo: str, items: list[Item]) -> list[str]:
    lineas = [f"## {titulo}", ""]
    if not items:
        lineas += ["No hay ítems.", ""]
        return lineas
    for item in items:
        lineas.append(f"### {item.titulo}")
        lineas.append(f"- Fuente: {item.fuente}")
        lineas.append(f"- Fecha: {formatear_fecha(item)}")
        lineas.append(f"- Enlace: {item.enlace_real or item.enlace}")
        lineas.append(f"- Puntuación: {formato_puntuacion(item)}")
        lineas.append("")
        lineas.append(cuerpo_item(item))
        lineas.append("")
        lineas += render_verificacion(item)
        lineas.append("")
    return lineas


def ordenar_por_puntuacion(items: list[Item]) -> list[Item]:
    """Ordena por puntuación de Groq (o palabras clave) y fecha, descendente."""
    sin_fecha = datetime.min.replace(tzinfo=timezone.utc)
    return sorted(
        items,
        key=lambda i: (clave_puntuacion(i), i.fecha or sin_fecha),
        reverse=True,
    )


def generar_informes(items: list[Item]) -> tuple[Path, Path]:
    """Escribe dos informes: con texto real y solo titulares.

    El primero (AAAA-MM-DD.md) incluye únicamente los ítems con texto real y se
    destina a NotebookLM. El segundo (AAAA-MM-DD-titulares.md) recoge los ítems
    sin texto y no debe subirse a NotebookLM.
    """
    ahora = datetime.now(ZONA)
    con_texto = ordenar_por_puntuacion([i for i in items if not i.solo_titular])
    titulos_con_texto = {normalizar_titulo(i.titulo) for i in con_texto}
    sin_texto = ordenar_por_puntuacion(
        [
            i
            for i in items
            if i.solo_titular and normalizar_titulo(i.titulo) not in titulos_con_texto
        ]
    )

    stem = ahora.strftime("%Y-%m-%d")
    ruta_principal = OUTPUT_DIR / f"{stem}.md"
    ruta_titulares = OUTPUT_DIR / f"{stem}-titulares.md"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ruta_principal.write_text(
        _cuerpo_informe_principal(con_texto, sin_texto, ahora), encoding="utf-8"
    )
    ruta_titulares.write_text(
        _cuerpo_informe_titulares(sin_texto, ahora), encoding="utf-8"
    )
    return ruta_principal, ruta_titulares


def _cuerpo_informe_principal(
    con_texto: list[Item], sin_texto: list[Item], ahora: datetime
) -> str:
    noticias, investigacion, videos = clasificar(con_texto)

    lineas: list[str] = [f"# Informe de IA y Educación - {ahora.strftime('%Y-%m-%d')}", ""]

    lineas += render_leyenda_verificacion()
    lineas += render_resumen_verificacion(con_texto)

    lineas += ["## Índice", ""]
    lineas.append(f"- Noticias: {len(noticias)}")
    lineas.append(f"- Investigación (arXiv): {len(investigacion)}")
    lineas.append(f"- Vídeos: {len(videos)}")
    lineas.append(f"- Total con texto real: {len(con_texto)}")
    lineas.append(
        f"- Titulares sin texto (archivo aparte, no NotebookLM): {len(sin_texto)}"
    )
    lineas.append("")

    lineas += render_seccion("Noticias", noticias)
    lineas += render_seccion("Investigación (arXiv)", investigacion)
    lineas += render_seccion("Vídeos", videos)

    lineas += ["## Fuentes para NotebookLM", ""]
    fuentes_reales = [
        item
        for item in con_texto
        if "news.google.com" not in (item.enlace_real or item.enlace)
    ]
    if fuentes_reales:
        for item in fuentes_reales:
            lineas.append(f"- {(item.enlace_real or item.enlace).strip()}")
    else:
        lineas.append("- (ninguna)")
    lineas.append("")

    return "\n".join(lineas)


def _cuerpo_informe_titulares(sin_texto: list[Item], ahora: datetime) -> str:
    lineas: list[str] = [
        f"# Titulares sin texto - {ahora.strftime('%Y-%m-%d')}",
        "",
        "Ítems de los que no se pudo extraer texto real (solo titular). "
        "No subir este archivo a NotebookLM.",
        "",
    ]
    lineas += ["## Índice", "", f"- Total [solo titular]: {len(sin_texto)}", ""]
    lineas += render_seccion("Titulares sin texto", sin_texto)
    return "\n".join(lineas)


def resumen_consola_final(
    items: list[Item],
    resultados: list[Resultado],
    ruta_principal: Path | None = None,
    ruta_titulares: Path | None = None,
    exitos: int = 0,
) -> None:
    con_texto = sum(1 for item in items if not item.solo_titular)
    sin_texto = len(items) - con_texto
    print("\n== Resumen ==")
    print(f"Fuentes consultadas     : {len(resultados)}")
    fallos = [r for r in resultados if r.error]
    print(f"Fuentes con error       : {len(fallos)}")
    for r in fallos:
        print(f"  - {r.fuente}: {r.error}")
    print(f"Total bruto             : {sum(r.total for r in resultados)}")
    print(f"Ítems finales           : {len(items)}")
    print(f"  Con texto real        : {con_texto}")
    print(f"  Solo titular          : {sin_texto}")
    print(f"Textos extraídos        : {exitos}/{len(items)}")
    if ruta_principal:
        print(f"Informe principal       : {ruta_principal}")
    if ruta_titulares:
        print(f"Informe de titulares    : {ruta_titulares}")
    print(f"Ítems con texto real    : {con_texto}")


def main() -> int:
    cargar_env()
    try:
        items, resultados = recoger()
    except Exception as exc:  # noqa: BLE001
        print(f"Error fatal: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    exitos = enriquecer(items)
    items = deduplicar_videos(items)
    exitos = sum(1 for item in items if not item.solo_titular)
    imprimir_tabla_textos(items, "Textos reales extraídos por fuente")
    items_con_texto = [item for item in items if not item.solo_titular]
    try:
        verificar_items(items_con_texto)
    except Exception as exc:  # noqa: BLE001
        print(f"  Aviso: fallo global de verificación: {type(exc).__name__}: {exc}")
        for item in items_con_texto:
            if item.verificacion is None:
                item.verificacion = {
                    "afirmaciones": [],
                    "errores": [f"{type(exc).__name__}: {exc}"],
                }
    conteo_verificacion = contar_etiquetas(items_con_texto)
    print(
        "  Verificación: "
        + ", ".join(
            f"{etiqueta}: {conteo_verificacion[etiqueta]}"
            for etiqueta in ETIQUETAS_VERIFICACION
        )
    )
    ruta_principal, ruta_titulares = generar_informes(items)
    resumen_consola_final(
        items, resultados, ruta_principal, ruta_titulares, exitos
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
