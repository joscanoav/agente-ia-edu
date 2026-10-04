"""Recogida de datos sobre IA y educación.

Fuentes: arXiv (cs.CY, cs.AI), Google News RSS (2 consultas ES + 2 EN)
y todos los feeds definidos en sources.yaml.

Filtra los últimos 7 días, elimina duplicados y muestra por pantalla
cuántos ítems salieron de cada fuente. Si una fuente falla, continúa.

Este paso NO usa el LLM: solo recopila yNormaliza los datos.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote_plus

import feedparser
import requests
import yaml
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
SOURCES_FILE = BASE_DIR / "sources.yaml"
OUTPUT_DIR = BASE_DIR / "informes"

DAYS_WINDOW = 7
USER_AGENT = (
    "Mozilla/5.0 (compatible; AgenteIAEdu/1.0; +https://example.org/bot)"
)
TIMEOUT = 30

ARXIV_API = "http://export.arxiv.org/api/query"
ARXIV_CATEGORIES = ["cs.CY", "cs.AI"]
ARXIV_TERMS = ["education", "learning", "teaching"]
ARXIV_MAX_RESULTS = 40

GOOGLE_NEWS_QUERIES = [
    ("es", "inteligencia artificial educación"),
    ("es", "inteligencia artificial docentes"),
    ("en", "artificial intelligence education"),
    ("en", "AI teaching learning"),
]


@dataclass
class Item:
    titulo: str
    enlace: str
    fecha: datetime | None
    fuente: str
    resumen: str = ""
    etiquetas: list[str] = field(default_factory=list)


@dataclass
class Resultado:
    fuente: str
    items: list[Item]
    error: str | None = None

    @property
    def total(self) -> int:
        return len(self.items)


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


def parsear_feed(contenido: bytes, fuente: str) -> list[Item]:
    """Convierte un XML de feed en Items con fecha timezone-aware."""
    datos = feedparser.parse(contenido)
    items: list[Item] = []
    for entrada in datos.entries:
        fecha = a_datetime(
            getattr(entrada, "published_parsed", None)
            or getattr(entrada, "updated_parsed", None)
        )
        resumen = (getattr(entrada, "summary", "") or "").strip()
        items.append(
            Item(
                titulo=(getattr(entrada, "title", "") or "").strip(),
                enlace=(getattr(entrada, "link", "") or "").strip(),
                fecha=fecha,
                fuente=fuente,
                resumen=resumen,
                etiquetas=list(getattr(entrada, "tags", []) or []),
            )
        )
    return items


def dentro_de_ventana(item: Item, limite: datetime) -> bool:
    """True si el item no tiene fecha o es posterior al límite."""
    if item.fecha is None:
        return True
    return item.fecha >= limite


def pedir(session: requests.Session, url: str, params: dict | None = None) -> bytes:
    respuesta = session.get(url, params=params, timeout=TIMEOUT)
    respuesta.raise_for_status()
    return respuesta.content


# --------------------------------------------------------------------------
# Fuentes concretas
# --------------------------------------------------------------------------


def obtener_arxiv(session: requests.Session, categoria: str) -> Resultado:
    nombre = f"arXiv ({categoria})"
    url = ARXIV_API
    try:
        terminos = " OR ".join(f"all:{t}" for t in ARXIV_TERMS)
        consulta = f"cat:{categoria} AND ({terminos})"
        contenido = pedir(
            session,
            url,
            {
                "search_query": consulta,
                "start": 0,
                "max_results": ARXIV_MAX_RESULTS,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            },
        )
        items = parsear_feed(contenido, nombre)
        if not items:
            return Resultado(nombre, [], "sin resultados")
        return Resultado(nombre, items)
    except Exception as exc:  # noqa: BLE001 - se continúa con el resto
        return Resultado(nombre, [], f"{type(exc).__name__}: {exc}")


def obtener_google_news(session: requests.Session, idioma: str, consulta: str) -> Resultado:
    nombre = f"Google News [{idioma}] {consulta}"
    try:
        if idioma == "es":
            params = {"q": consulta, "hl": "es", "gl": "ES", "ceid": "ES:es"}
        else:
            params = {"q": consulta, "hl": "en-US", "gl": "US", "ceid": "US:en"}
        contenido = pedir(
            session,
            f"https://news.google.com/rss/search?q={quote_plus(consulta)}",
            params,
        )
        items = parsear_feed(contenido, nombre)
        if not items:
            return Resultado(nombre, [], "sin resultados")
        return Resultado(nombre, items)
    except Exception as exc:  # noqa: BLE001
        return Resultado(nombre, [], f"{type(exc).__name__}: {exc}")


def cargar_sources() -> list[dict]:
    if not SOURCES_FILE.exists():
        raise FileNotFoundError(f"No existe {SOURCES_FILE}")
    with open(SOURCES_FILE, "r", encoding="utf-8") as fh:
        datos = yaml.safe_load(fh) or {}
    feeds = datos.get("feeds") or []
    return [f for f in feeds if f.get("url")]


def obtener_feed(session: requests.Session, nombre: str, url: str) -> Resultado:
    try:
        contenido = pedir(session, url)
        items = parsear_feed(contenido, nombre)
        if not items:
            return Resultado(nombre, [], "sin resultados")
        return Resultado(nombre, items)
    except Exception as exc:  # noqa: BLE001
        return Resultado(nombre, [], f"{type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------
# Orquestación
# --------------------------------------------------------------------------


def recoger() -> tuple[list[Item], list[Resultado]]:
    limite = datetime.now(timezone.utc) - timedelta(days=DAYS_WINDOW)
    resultados: list[Resultado] = []

    session = sesion_http()

    print("== Recogiendo datos de las ultimas 7 dias ==\n")

    print("[1/3] arXiv")
    for categoria in ARXIV_CATEGORIES:
        res = obtener_arxiv(session, categoria)
        resultados.append(res)
        print(f"  {res.fuente:<24} -> {res.total} items" + (f"  ({res.error})" if res.error else ""))

    print("\n[2/3] Google News RSS")
    for idioma, consulta in GOOGLE_NEWS_QUERIES:
        res = obtener_google_news(session, idioma, consulta)
        resultados.append(res)
        print(f"  {res.fuente:<40} -> {res.total} items" + (f"  ({res.error})" if res.error else ""))

    print("\n[3/3] Feeds de sources.yaml")
    for feed in cargar_sources():
        nombre = feed.get("name") or feed["url"]
        res = obtener_feed(session, nombre, feed["url"])
        resultados.append(res)
        print(f"  {res.fuente:<40} -> {res.total} items" + (f"  ({res.error})" if res.error else ""))

    unicos: list[Item] = []
    vistos: set[str] = set()
    for res in resultados:
        for item in res.items:
            if not dentro_de_ventana(item, limite):
                continue
            clave = (item.enlace or item.titulo).strip().lower()
            if not clave or clave in vistos:
                continue
            vistos.add(clave)
            unicos.append(item)

    unicos.sort(key=lambda i: i.fecha or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
    return unicos, resultados


def resumen_consola(items: list[Item], resultados: list[Resultado]) -> None:
    print("\n== Resumen ==")
    print(f"Fuentes consultadas : {len(resultados)}")
    fallos = [r for r in resultados if r.error]
    print(f"Fuentes con error  : {len(fallos)}")
    for r in fallos:
        print(f"  - {r.fuente}: {r.error}")
    print(f"Items unicos (7d)  : {len(items)}")
    print(f"Total bruto         : {sum(r.total for r in resultados)}")
    print("\n(No se genera informe todavia: esta version solo recoge datos.)")


def main() -> int:
    cargar_env()
    try:
        items, resultados = recoger()
    except Exception as exc:  # noqa: BLE001
        print(f"Error fatal: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    resumen_consola(items, resultados)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())