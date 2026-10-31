"""Módulo de verificación de afirmaciones (fase 2).

Verifica afirmaciones comprobables de ítems ya seleccionados siguiendo
los criterios de VERIFICACION.md.
"""
from __future__ import annotations

import io
import json
import os
import re
import time
import unicodedata
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import requests
import trafilatura
from dotenv import load_dotenv
from selectolax.parser import HTMLParser

try:  # Se aísla el import: si falla, se gestiona más abajo
    from groq import Groq, RateLimitError
except Exception:  # noqa: BLE001
    Groq = None
    RateLimitError = Exception

try:  # Extractor de PDF (autorizado en requirements.txt)
    from pypdf import PdfReader
except Exception:  # noqa: BLE001
    PdfReader = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

TIMEOUT = 15
GROQ_MODEL_POR_DEFECTO = "openai/gpt-oss-120b"
MAX_CARACTERES_FUENTE = 8000  # Límite de texto de fuente enviado al modelo
MAX_LLAMADAS_HERRAMIENTA_POR_AFIRMACION = 5
UMBRAL_TEXTO_UTIL = 100  # Mínimo de caracteres para considerar útil un texto extraído

# Reintentos ante 429 (rate limit) y espera máxima aceptada del propio error.
MAX_REINTENTOS_RATE_LIMIT = 3
MAX_ESPERA_RATE_LIMIT = 120.0
ESPERA_RATE_LIMIT_POR_DEFECTO = 2.0

DOMINIOS_FUENTE_PRIMARIA_CONFIANZA = (
    "ugr.es",
    "intef.es",
    "unesco.org",
    "oecd.org",
    "arxiv.org",
)

_STOPWORDS_RELACION = {
    "a",
    "al",
    "algo",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "como",
    "con",
    "cual",
    "de",
    "del",
    "desde",
    "donde",
    "dos",
    "el",
    "en",
    "entre",
    "es",
    "esa",
    "ese",
    "eso",
    "esta",
    "este",
    "estos",
    "fue",
    "ha",
    "han",
    "hasta",
    "hay",
    "la",
    "las",
    "le",
    "les",
    "lo",
    "los",
    "mas",
    "media",
    "mediante",
    "menos",
    "mi",
    "muy",
    "no",
    "nos",
    "o",
    "of",
    "on",
    "only",
    "or",
    "para",
    "pero",
    "por",
    "que",
    "se",
    "sin",
    "single",
    "sobre",
    "sola",
    "solamente",
    "solo",
    "son",
    "su",
    "sus",
    "tambien",
    "the",
    "their",
    "this",
    "to",
    "un",
    "una",
    "uno",
    "unos",
    "unas",
    "was",
    "were",
    "with",
    "y",
    "ya",
}


@dataclass
class AfirmacionVerificada:
    texto: str
    tipo: str
    fidelidad: str  # fiel / distorsiona / no aparece en la fuente citada
    respaldo: str  # respaldado / parcial / contradicho / no verificado
    fuente_primaria_url: str
    evidencia: str
    matices_omitidos: list[dict[str, str]]
    confianza: float
    revisado_humano: bool = False
    fuente_cobertura: str = "texto_completo"  # texto_completo / solo_abstract / sin_fuente
    veredicto_modelo_sin_validar: dict[str, Any] | None = None
    motivo_rechazo: str = ""
    explicacion_parcial: str = ""


def cargar_env() -> None:
    load_dotenv(os.path.join(BASE_DIR, ".env"))


def sesion_http() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def _segundos_espera_rate_limit(error: Exception) -> float:
    """Lee el tiempo de espera que indica el error 429 (cabecera o mensaje)."""
    respuesta = getattr(error, "response", None)
    cabeceras = getattr(respuesta, "headers", None)
    if cabeceras is not None:
        try:
            valor = cabeceras.get("retry-after")
        except Exception:
            valor = None
        if valor:
            try:
                return max(
                    0.0, min(MAX_ESPERA_RATE_LIMIT, float(str(valor).strip()))
                )
            except (TypeError, ValueError):
                pass

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
    return ESPERA_RATE_LIMIT_POR_DEFECTO


def _llamar_groq_json(
    cliente,
    modelo: str,
    system_prompt: str,
    user_prompt: str,
    resultado: dict[str, Any],
    contexto: str,
) -> tuple[dict[str, Any] | None, bool]:
    """Llama a Groq y reintenta solo los 429 hasta MAX_REINTENTOS_RATE_LIMIT.

    Devuelve (datos_json, se_agoto_el_limite). No relaja ninguna validación.
    """
    for intento in range(MAX_REINTENTOS_RATE_LIMIT + 1):
        try:
            resp = cliente.chat.completions.create(
                model=modelo,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )
            contenido_resp = resp.choices[0].message.content or "{}"
            return json.loads(contenido_resp), False
        except RateLimitError as e:
            if intento >= MAX_REINTENTOS_RATE_LIMIT:
                resultado["limite_api"] = True
                resultado["errores"].append(
                    f"RateLimitError {contexto} tras "
                    f"{MAX_REINTENTOS_RATE_LIMIT + 1} intentos: {e}"
                )
                return None, True
            espera = _segundos_espera_rate_limit(e)
            resultado["errores"].append(
                f"RateLimitError {contexto}; esperando {espera:.1f}s y "
                f"reintentando ({intento + 1}/{MAX_REINTENTOS_RATE_LIMIT})"
            )
            time.sleep(espera)
        except Exception as e:
            resultado["errores"].append(f"Error {contexto}: {e}")
            return None, False
    return None, False


def config_groq_verificador() -> tuple[str, str, bool]:
    """Devuelve (clave, modelo, usado_por_defecto)."""
    clave = os.environ.get("GROQ_API_KEY", "").strip()
    modelo_verif = os.environ.get("GROQ_MODEL_VERIFICADOR", "").strip()
    usado_defecto = False
    if not modelo_verif:
        modelo_verif = os.environ.get("GROQ_MODEL", "").strip() or GROQ_MODEL_POR_DEFECTO
        usado_defecto = True
    else:
        usado_defecto = False
    return clave, modelo_verif, usado_defecto


def _normalizar_texto_para_comparacion(texto: str) -> str:
    if not texto:
        return ""
    texto = texto.replace("\u201c", '"').replace("\u201d", '"')
    texto = texto.replace("\u2018", "'").replace("\u2019", "'")
    texto = re.sub(r"\s+", " ", texto)
    texto = re.sub(r"\s*-\s*", "-", texto)
    return texto.strip().lower()


def _tokens_relacion(texto: str) -> set[str]:
    if not texto:
        return set()
    normalizado = unicodedata.normalize("NFKD", texto.lower())
    normalizado = "".join(
        c for c in normalizado if not unicodedata.combining(c)
    )
    return {
        token
        for token in re.findall(r"[a-z0-9]+", normalizado)
        if len(token) >= 3 and token not in _STOPWORDS_RELACION
    }


def _matiz_relacionado(
    texto_matiz: str,
    por_que: str,
    afirmacion: str,
    fragmento: str = "",
) -> bool:
    base = _tokens_relacion(afirmacion)
    matiz = _tokens_relacion(f"{texto_matiz} {por_que} {fragmento}")
    if not base or not matiz:
        return False
    for token_af in base:
        for token_matiz in matiz:
            if token_af == token_matiz:
                return True
            if (
                len(token_af) >= 5
                and len(token_matiz) >= 5
                and token_af[:5] == token_matiz[:5]
            ):
                return True
    return False


def _dominio_es_fuente_primaria_confiada(url: str) -> bool:
    try:
        parsed = urlparse(url)
        dominio = (parsed.netloc or "").lower()
        for d in DOMINIOS_FUENTE_PRIMARIA_CONFIANZA:
            if dominio == d or dominio.endswith("." + d):
                return True
    except Exception:
        pass
    return False


def _extraer_id_arxiv(url: str) -> str | None:
    try:
        parsed = urlparse(url)
        path = parsed.path or ""
        m = re.search(r"(\d{4}\.\d{4,5})(v\d+)?", path)
        if m:
            return m.group(1)
        m2 = re.search(r"([a-z\-]+/\d{7})(v\d+)?", path)
        if m2:
            return m2.group(1)
    except Exception:
        pass
    return None


def descargar_recurso(
    url: str, session: requests.Session | None = None
) -> tuple[bytes | None, str, str]:
    """Descarga una URL una sola vez y devuelve (bytes, tipo, content_type)."""
    try:
        if session is None:
            session = sesion_http()
        resp = session.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
        resp.raise_for_status()
        datos = resp.content or b""
        content_type = resp.headers.get("Content-Type", "").lower()
        if "application/pdf" in content_type or datos[:4] == b"%PDF":
            return datos, "pdf", content_type
        if "text/html" in content_type or not content_type:
            return datos, "html", content_type
        return datos, "text", content_type
    except Exception as e:
        return None, f"error:{e}", ""


def _bytes_a_texto(datos: bytes, content_type: str = "") -> str:
    if not datos:
        return ""
    encoding = None
    m = re.search(r"charset=([\w\-]+)", content_type or "", re.I)
    if m:
        encoding = m.group(1)
    for enc in (encoding, "utf-8", "latin-1"):
        if not enc:
            continue
        try:
            return datos.decode(enc)
        except (LookupError, UnicodeDecodeError):
            continue
    return datos.decode("utf-8", errors="replace")


def _extraer_texto_pdf(datos: bytes | None) -> str:
    if not datos or PdfReader is None:
        return ""
    try:
        lector = PdfReader(io.BytesIO(datos))
        partes: list[str] = []
        for pagina in lector.pages:
            try:
                partes.append(pagina.extract_text() or "")
            except Exception:
                continue
        return re.sub(r"\s+", " ", " ".join(partes)).strip()
    except Exception:
        return ""


def _texto_de_recurso(datos: bytes | None, tipo: str, content_type: str = "") -> str:
    if not datos:
        return ""
    if tipo == "pdf" or datos[:4] == b"%PDF":
        return _extraer_texto_pdf(datos)
    if tipo in ("html", "text"):
        return descargar_texto_con_enlaces(_bytes_a_texto(datos, content_type))
    return ""


def descargar_contenido(url: str, session: requests.Session | None = None) -> tuple[str | None, str]:
    """Descarga contenido. Devuelve (texto_extraido_o_none, tipo)."""
    datos, tipo, content_type = descargar_recurso(url, session)
    if datos is None:
        return None, tipo
    if tipo == "pdf":
        return None, "pdf"
    return _bytes_a_texto(datos, content_type), tipo


def descargar_html(url: str, session: requests.Session | None = None) -> str | None:
    texto, tipo = descargar_contenido(url, session)
    if tipo == "html" or tipo == "text":
        return texto
    return None


def _texto_util(texto: str | None, minimo: int = UMBRAL_TEXTO_UTIL) -> bool:
    if not texto:
        return False
    return len(re.sub(r"\s+", " ", texto).strip()) >= minimo


def extraer_enlaces_html(html: str, base_url: str | None = None) -> list[str]:
    """Extrae href absolutos/relativos desde HTML usando selectolax."""
    enlaces: list[str] = []
    try:
        parser = HTMLParser(html)
        for node in parser.css("a[href]"):
            href = node.attributes.get("href") or ""
            href = href.strip()
            if not href or href.startswith("#") or href.lower().startswith("javascript:"):
                continue
            if base_url and href.startswith("/"):
                try:
                    from urllib.parse import urljoin

                    href = urljoin(base_url, href)
                except Exception:
                    pass
            elif base_url and not href.lower().startswith("http"):
                try:
                    from urllib.parse import urljoin

                    href = urljoin(base_url, href)
                except Exception:
                    pass
            if href.lower().startswith("http://") or href.lower().startswith("https://"):
                enlaces.append(href)
    except Exception:
        pass
    vistos = set()
    res = []
    for e in enlaces:
        if e in vistos:
            continue
        vistos.add(e)
        res.append(e)
    return res


def descargar_texto_con_enlaces(html: str) -> str:
    """Intenta extraer texto incluyendo enlaces si es posible."""
    try:
        texto = trafilatura.extract(html, include_comments=False, include_tables=False)
        if texto:
            return texto
    except Exception:
        pass
    try:
        parser = HTMLParser(html)
        for tag in parser.css("script, style, noscript"):
            tag.decompose()
        texto = parser.text(separator=" ")
        texto = re.sub(r"\s+", " ", texto).strip()
        return texto
    except Exception:
        return ""


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


def _dividir_frases(texto: str) -> list[str]:
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


def _seleccionar_fragmentos_relevantes(texto_fuente: str, afirmacion: str) -> str:
    if not texto_fuente:
        return ""
    if not afirmacion:
        return recortar_fuente(texto_fuente)
    # Palabras clave simples
    palabras = re.findall(r"\b\w{4,}\b", afirmacion.lower())
    if not palabras:
        return recortar_fuente(texto_fuente)
    oraciones = _dividir_frases(texto_fuente)
    puntuadas = []
    for s in oraciones:
        sl = s.lower()
        score = sum(1 for p in palabras if p in sl)
        if score > 0:
            puntuadas.append((score, s))
    puntuadas.sort(key=lambda x: x[0], reverse=True)
    seleccionadas = [s for _, s in puntuadas[:10]]
    if not seleccionadas:
        return recortar_fuente(texto_fuente)
    combinado = " ".join(seleccionadas)
    if len(combinado) <= MAX_CARACTERES_FUENTE:
        return combinado.strip()
    return recortar_fuente(combinado)


def recortar_fuente(texto: str) -> str:
    texto = (texto or "").strip()
    if len(texto) <= MAX_CARACTERES_FUENTE:
        return texto
    fragmento = texto[:MAX_CARACTERES_FUENTE]
    ultimo = _ultimo_fin_frase(fragmento)
    if ultimo:
        return fragmento[:ultimo].strip()
    return fragmento.strip()


def _es_arxiv(item) -> bool:
    try:
        cat = getattr(item, "categoria", "") or ""
        if cat == "arxiv":
            return True
        enlace = getattr(item, "enlace_real", "") or getattr(item, "enlace", "") or ""
        if "arxiv.org" in enlace:
            return True
    except Exception:
        pass
    return False


def _resolver_fuente_item(
    item, session: requests.Session, resultado: dict[str, Any]
) -> tuple[str, str, str]:
    """Resuelve y descarga la fuente primaria una sola vez por ítem.

    Devuelve (url, texto, cobertura), con cobertura:
    - "texto_completo": se obtuvo el texto del documento.
    - "solo_abstract": solo se pudo usar el resumen/abstract.
    - "sin_fuente": no se encontró fuente utilizable.
    """
    url_item = getattr(item, "enlace_real", "") or getattr(item, "enlace", "") or ""
    abstract = (getattr(item, "resumen", "") or getattr(item, "texto", "") or "").strip()
    es_arxiv_item = _es_arxiv(item)
    dominio_confianza = _dominio_es_fuente_primaria_confiada(url_item)

    cache: dict[str, tuple[bytes | None, str, str]] = {}

    def descargar(url: str) -> tuple[bytes | None, str, str]:
        if url not in cache:
            cache[url] = descargar_recurso(url, session)
        return cache[url]

    def leer(url: str, registrar_error: bool = True) -> tuple[str, str]:
        if not url:
            return "", ""
        datos, tipo, content_type = descargar(url)
        if datos is None:
            if registrar_error and tipo.startswith("error:"):
                resultado["errores"].append(f"Error descargando {url}: {tipo}")
            return "", tipo
        return _texto_de_recurso(datos, tipo, content_type), tipo

    def usar_abstract() -> tuple[str, str, str]:
        if abstract and _texto_util(abstract):
            cobertura = (
                "solo_abstract"
                if getattr(item, "solo_titular", False)
                else "texto_completo"
            )
            return url_item, abstract, cobertura
        return "", "", "sin_fuente"

    if es_arxiv_item:
        arxiv_id = _extraer_id_arxiv(url_item)
        if arxiv_id:
            url_html = f"https://arxiv.org/html/{arxiv_id}"
            texto, tipo_html = leer(url_html, registrar_error=False)
            if _texto_util(texto):
                return url_html, texto, "texto_completo"

            url_pdf = f"https://arxiv.org/pdf/{arxiv_id}"
            texto_pdf, tipo_pdf = leer(url_pdf, registrar_error=False)
            if _texto_util(texto_pdf):
                return url_pdf, texto_pdf, "texto_completo"
            if tipo_pdf.startswith("error:"):
                resultado["errores"].append(f"Error descargando PDF de arXiv: {tipo_pdf}")
            elif PdfReader is None:
                resultado["errores"].append("pypdf no disponible para extraer el PDF de arXiv")
            else:
                resultado["errores"].append(f"No se pudo extraer texto útil del PDF de arXiv: {url_pdf}")

            if tipo_html.startswith("error:"):
                resultado["errores"].append(f"Error descargando {url_html}: {tipo_html}")

            url_abs = url_item or f"https://arxiv.org/abs/{arxiv_id}"
            if not abstract:
                texto_abs, _ = leer(url_abs)
                if texto_abs:
                    abstract = texto_abs
            if abstract:
                return url_abs, abstract, "solo_abstract"
            return url_abs, "", "sin_fuente"

        texto, _ = leer(url_item)
        if _texto_util(texto):
            return url_item, texto, "texto_completo"
        if abstract:
            return url_item, abstract, "solo_abstract"
        return url_item, "", "sin_fuente"

    if dominio_confianza:
        texto, _ = leer(url_item)
        if _texto_util(texto):
            return url_item, texto, "texto_completo"
        if abstract:
            return url_item, abstract, "solo_abstract"
        return url_item, "", "sin_fuente"

    if not url_item:
        return "", "", "sin_fuente"

    texto_item = (getattr(item, "texto", "") or "").strip()
    if _texto_util(texto_item):
        return url_item, texto_item, "texto_completo"
    if abstract and _texto_util(abstract):
        return url_item, abstract, "solo_abstract"

    datos_item, tipo_item, content_type_item = descargar(url_item)
    if datos_item is None:
        if tipo_item.startswith("error:"):
            resultado["errores"].append(f"Error descargando {url_item}: {tipo_item}")
        return usar_abstract()
    if tipo_item == "pdf" or datos_item[:4] == b"%PDF":
        return usar_abstract()

    html_item = _bytes_a_texto(datos_item, content_type_item)
    if not html_item:
        return usar_abstract()

    enlaces = extraer_enlaces_html(html_item, base_url=url_item)
    candidatos: list[str] = []
    for e in enlaces:
        el = e.lower()
        if any(
            k in el
            for k in [
                "arxiv.org",
                "ugr.es",
                "intef.es",
                "unesco.org",
                "oecd.org",
                "/revistas/",
                "revistas.",
                "doi.org",
            ]
        ):
            if e not in candidatos:
                candidatos.append(e)
        elif ".pdf" in el or "/handle/" in el or "/bitstream/" in el:
            if e not in candidatos:
                candidatos.append(e)

    for cand in candidatos:
        if cand == url_item:
            continue
        texto, _ = leer(cand)
        if _texto_util(texto, 100):
            return cand, texto, "texto_completo"

    return usar_abstract()


def _es_afirmacion_autosuficiente(texto: str) -> bool:
    t = (texto or "").strip()
    if len(t) < 15:
        return False
    bajo = t.lower()
    prohibidos = (
        "publicado en ",
        "publicado por ",
        "sobre tutor",
        "sobre el tutor",
        "sobre la ia",
        "sobre inteligencia artificial",
    )
    return not any(bajo.startswith(p) for p in prohibidos)


def _extraer_afirmaciones_groq(item, cliente, modelo: str, resultado: dict[str, Any]) -> list[str]:
    titulo = getattr(item, "titulo", "") or ""
    texto_item = getattr(item, "texto", "") or ""
    resumen = getattr(item, "resumen", "") or ""
    contenido = (texto_item or resumen or titulo).strip()
    prompt = (
        "Extrae de 1 a 3 afirmaciones COMPROBABLES del siguiente texto. "
        "Escribe TODAS las afirmaciones en ESPAÑOL, aunque el texto esté en otro "
        "idioma; traduce el contenido y conserva nombres propios, cifras y fechas. "
        "Cada afirmación debe ser ATÓMICA y AUTOSUFICIENTE: incluye el sujeto "
        "(quién, qué estudio o institución) y un dato concreto (cifra, fecha, "
        "nombre, resultado). Si el texto contiene cifras, inclúyelas. "
        "PROHIBIDO devolver fragmentos incompletos como \"publicado en X\", "
        "\"sobre tutor de IA\" o similares, así como opiniones, valoraciones "
        "o conclusiones subjetivas. "
        "Devuelve ÚNICAMENTE un JSON con la clave \"afirmaciones\" como lista de strings.\n\n"
        f"Texto:\n{contenido[:4000]}"
    )
    datos, limite = _llamar_groq_json(
        cliente,
        modelo,
        "Eres un extractor estricto que escribe siempre en español. "
        "Respondes solo con JSON válido.",
        prompt,
        resultado,
        "extrayendo afirmaciones",
    )
    if limite:
        resultado["limite_api"] = True
    if not isinstance(datos, dict):
        return []
    afirmaciones = datos.get("afirmaciones") or []
    res = []
    for a in afirmaciones:
        if isinstance(a, str) and a.strip():
            a = a.strip()
            if _es_afirmacion_autosuficiente(a):
                res.append(a)
    return res[:3]


def _extraer_fragmentos_entre_comillas(evidencia: str) -> list[str] | None:
    if not evidencia:
        return None
    if not re.search(r'["“”«»]', evidencia):
        return None
    fragmentos = re.findall(r'["“”«»]([^"“”«»]+)["“”«»]', evidencia)
    return [f.strip() for f in fragmentos if f.strip()]


def _limpiar_evidencia(evidencia: str) -> str:
    fragmentos = _extraer_fragmentos_entre_comillas(evidencia)
    if fragmentos is None:
        return evidencia.strip()
    return " [...] ".join(fragmentos)


def _validar_evidencia_en_fuente(evidencia: str, texto_fuente: str) -> bool:
    if not evidencia or not texto_fuente:
        return False
    fragmentos = _extraer_fragmentos_entre_comillas(evidencia)
    if fragmentos is None:
        fragmentos = [evidencia.strip()]
    if not fragmentos:
        return False
    tf_norm = _normalizar_texto_para_comparacion(texto_fuente)
    tf_sin_espacios = re.sub(r"\s+", "", tf_norm)
    for fragmento in fragmentos:
        frag_norm = _normalizar_texto_para_comparacion(fragmento)
        if not frag_norm:
            return False
        candidatos = [frag_norm]
        sin_puntuacion = frag_norm.strip(" \t\r\n.,;:!?()[]{}\"'")
        if sin_puntuacion and sin_puntuacion != frag_norm:
            candidatos.append(sin_puntuacion)
        if any(c in tf_norm for c in candidatos):
            continue
        if any(
            re.sub(r"\s+", "", c) in tf_sin_espacios for c in candidatos
        ):
            continue
        return False
    return True


_UNIDADES_NUMERO = {
    "cero": 0,
    "un": 1,
    "uno": 1,
    "una": 1,
    "dos": 2,
    "tres": 3,
    "cuatro": 4,
    "cinco": 5,
    "seis": 6,
    "siete": 7,
    "ocho": 8,
    "nueve": 9,
    "diez": 10,
    "once": 11,
    "doce": 12,
    "trece": 13,
    "catorce": 14,
    "quince": 15,
    "dieciseis": 16,
    "diecisiete": 17,
    "dieciocho": 18,
    "diecinueve": 19,
    "veinte": 20,
    "veintiuno": 21,
    "veintiun": 21,
    "veintiuna": 21,
    "veintidos": 22,
    "veintitres": 23,
    "veinticuatro": 24,
    "veinticinco": 25,
    "veintiseis": 26,
    "veintisiete": 27,
    "veintiocho": 28,
    "veintinueve": 29,
    "treinta": 30,
    "cuarenta": 40,
    "cincuenta": 50,
    "sesenta": 60,
    "setenta": 70,
    "ochenta": 80,
    "noventa": 90,
}

_CENTENAS_NUMERO = {
    "cien": 100,
    "ciento": 100,
    "doscientos": 200,
    "doscientas": 200,
    "trescientos": 300,
    "trescientas": 300,
    "cuatrocientos": 400,
    "cuatrocientas": 400,
    "quinientos": 500,
    "quinientas": 500,
    "seiscientos": 600,
    "seiscientas": 600,
    "setecientos": 700,
    "setecientas": 700,
    "ochocientos": 800,
    "ochocientas": 800,
    "novecientos": 900,
    "novecientas": 900,
}

_ESCALAS_MILLON = {
    "millon": 1_000_000,
    "millones": 1_000_000,
    "millardo": 1_000_000_000,
    "millardos": 1_000_000_000,
    "billon": 1_000_000_000_000,
    "billones": 1_000_000_000_000,
}

_PALABRAS_NUMERO = (
    set(_UNIDADES_NUMERO) | set(_CENTENAS_NUMERO) | set(_ESCALAS_MILLON) | {"mil"}
)


def _unir_miles_con_espacios(texto: str) -> str:
    """Une grupos de miles separados por espacios: '3 000' -> '3000'."""
    patron = re.compile(
        r"(?<!\d)(\d{1,3}(?:[ \u00a0\u202f]\d{3})+)(?!\d)"
    )
    return patron.sub(
        lambda m: re.sub(r"[ \u00a0\u202f]", "", m.group(1)), texto
    )


def _normalizar_numero(numero: str) -> str:
    n = (numero or "").strip().replace(" ", "")
    if not n:
        return ""
    if re.fullmatch(r"\d{1,3}(?:[.,]\d{3})+", n):
        return re.sub(r"[.,]", "", n)
    if "," in n and "." in n:
        if n.rfind(",") > n.rfind("."):
            n = n.replace(".", "").replace(",", ".")
        else:
            n = n.replace(",", "")
    elif "," in n:
        n = n.replace(",", ".")
    if "." in n:
        n = n.rstrip("0").rstrip(".")
    return n


def _quitar_acentos(texto: str) -> str:
    normalizado = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in normalizado if not unicodedata.combining(c))


def _convertir_palabras_numero(tokens: list[str]) -> int:
    total = 0
    actual = 0
    for token in tokens:
        if token == "y":
            continue
        if token in _UNIDADES_NUMERO:
            actual += _UNIDADES_NUMERO[token]
        elif token in _CENTENAS_NUMERO:
            actual += _CENTENAS_NUMERO[token]
        elif token == "mil":
            actual = actual or 1
            total += actual * 1000
            actual = 0
        elif token in _ESCALAS_MILLON:
            escala = _ESCALAS_MILLON[token]
            if actual:
                total = (total + actual) * escala
            elif total:
                total *= escala
            else:
                total = escala
            actual = 0
    return total + actual


def _numeros_en_palabras(texto: str) -> set[str]:
    """Extrae cifras escritas en letras ('doscientos mil' -> '200000')."""
    tokens = re.findall(r"[a-z]+", _quitar_acentos((texto or "").lower()))
    resultado: set[str] = set()
    i = 0
    while i < len(tokens):
        if tokens[i] not in _PALABRAS_NUMERO:
            i += 1
            continue
        secuencia: list[str] = []
        while i < len(tokens):
            token = tokens[i]
            if token in _PALABRAS_NUMERO:
                secuencia.append(token)
                i += 1
            elif (
                token == "y"
                and secuencia
                and i + 1 < len(tokens)
                and tokens[i + 1] in _PALABRAS_NUMERO
            ):
                secuencia.append(token)
                i += 1
            else:
                break
        while secuencia and secuencia[0] == "y":
            secuencia.pop(0)
        while secuencia and secuencia[-1] == "y":
            secuencia.pop()
        if not secuencia:
            continue
        if (
            len(secuencia) == 1
            and secuencia[0] in _UNIDADES_NUMERO
            and _UNIDADES_NUMERO[secuencia[0]] < 10
        ):
            continue
        valor = _convertir_palabras_numero(secuencia)
        if valor > 0:
            resultado.add(str(valor))
    return resultado


def _numeros_de_texto(texto: str) -> set[str]:
    if not texto:
        return set()
    texto_unido = _unir_miles_con_espacios(texto)
    encontrados = re.findall(
        r"(?<!\d)\d(?:[\d.,]*\d)?(?!\d)", texto_unido
    )
    numeros = {_normalizar_numero(n) for n in encontrados if n}
    numeros |= _numeros_en_palabras(texto)
    return {n for n in numeros if n}


def _validar_matices(
    matices: list[dict[str, Any]],
    texto_fuente: str,
    afirmacion: str = "",
    evidencia: str = "",
) -> list[dict[str, str]]:
    validados: list[dict[str, str]] = []
    if not isinstance(matices, list):
        return validados
    af_norm = _normalizar_texto_para_comparacion(afirmacion)
    ev_norm = _normalizar_texto_para_comparacion(evidencia)
    vistos: set[str] = set()
    for m in matices:
        if not isinstance(m, dict):
            continue
        texto_matiz = m.get("texto") or m.get("matiz") or m.get("descripcion") or ""
        frag = m.get("fragmento_literal") or m.get("evidencia") or m.get("fragmento") or ""
        por_que = m.get("por_que_matiza") or m.get("motivo") or ""
        texto_matiz = texto_matiz.strip() if isinstance(texto_matiz, str) else ""
        frag = frag.strip() if isinstance(frag, str) else ""
        por_que = por_que.strip() if isinstance(por_que, str) else ""
        if not frag:
            continue
        if not _validar_evidencia_en_fuente(frag, texto_fuente):
            continue
        frag_limpio = _limpiar_evidencia(frag)
        if not _matiz_relacionado(
            texto_matiz, por_que, afirmacion, frag_limpio
        ):
            continue
        frag_norm = _normalizar_texto_para_comparacion(frag_limpio)
        if af_norm and (frag_norm == af_norm or frag_norm in af_norm):
            continue
        if ev_norm and frag_norm == ev_norm:
            continue
        if frag_norm in vistos:
            continue
        vistos.add(frag_norm)
        validados.append(
            {
                "texto": texto_matiz or frag_limpio,
                "fragmento_literal": frag_limpio,
                "por_que_matiza": por_que or "limitación declarada por la fuente",
            }
        )
    return validados


def _comparar_afirmacion_groq(
    afirmacion: str,
    texto_fuente: str,
    fuente_url: str,
    cliente,
    modelo: str,
    resultado: dict[str, Any],
) -> tuple[dict[str, Any] | None, bool]:
    prompt = (
        "Compara la AFIRMACIÓN con la FUENTE PRIMARIA. "
        "Responde ÚNICAMENTE con JSON válido con estas claves EXACTAS:\n"
        "- texto: string (la afirmación)\n"
        "- tipo: string (cifra/fecha/publicacion/hecho/otro)\n"
        "- fidelidad: 'fiel' | 'distorsiona' | 'no aparece en la fuente citada'\n"
        "- respaldo: 'respaldado' | 'parcial' | 'contradicho' | 'no verificado'\n"
        "- fuente_primaria_url: string (la URL de la fuente primaria usada)\n"
        "- evidencia: string (CITA LITERAL exacta de la fuente que justifica el veredicto). "
        "Si necesitas varios pasajes, encierra cada uno en comillas separadas; no unas "
        "frases separadas con conectores ni omitas palabras intermedias.\n"
        "- explicacion_parcial: string (si respaldo es 'parcial', indica qué parte "
        "concreta de la afirmación no queda respaldada; si no, cadena vacía)\n"
        "- matices_omitidos: lista de objetos con 'fragmento_literal' (cita literal), "
        "'por_que_matiza' (motivo corto) y 'texto' (descripción breve). "
        "Incluye SOLO limitaciones o condiciones que la propia fuente declara y que "
        "restrinjan DIRECTAMENTE la afirmación concreta (tamaño/muestra, un solo centro, "
        "alcance, modelos usados...). Descarta cualquier dato que no limite esa afirmación. "
        "El campo 'texto' debe mencionar el sujeto o el dato de la afirmación. "
        "Escribe 'texto' y 'por_que_matiza' en español. "
        "NO repitas la afirmación ni los resultados principales. Si no hay, lista vacía.\n"
        "- confianza: number (0-1)\n"
        "- revisado_humano: false\n\n"
        "Regla de cifras: si la afirmación contiene cifras y respaldo es 'respaldado', "
        "TODAS las cifras de la afirmación deben aparecer literalmente en la evidencia; "
        "si no, no uses 'respaldado'.\n"
        "Si la evidencia respalda solo una parte de la afirmación, usa respaldo 'parcial' "
        "(NO 'contradicho') y explica en 'explicacion_parcial' qué parte concreta falla. "
        "Usa 'contradicho' solo cuando la evidencia contradiga directamente el núcleo "
        "de la afirmación. Cita siempre la evidencia literal que lo demuestra.\n\n"
        f"AFIRMACIÓN:\n{afirmacion}\n\n"
        f"FUENTE_PRIMARIA_URL:\n{fuente_url}\n\n"
        f"CONTENIDO_FUENTE:\n{texto_fuente}\n"
    )
    datos, limite = _llamar_groq_json(
        cliente,
        modelo,
        "Eres un verificador estricto. Solo usas la fuente proporcionada. "
        "Escribes en español. Respondes solo JSON válido.",
        prompt,
        resultado,
        "comparando afirmación",
    )
    if limite:
        resultado["limite_api"] = True
    return (datos if isinstance(datos, dict) else None), limite


def _pedir_evidencia_mas_larga(
    afirmacion: str,
    texto_fuente: str,
    fuente_url: str,
    evidencia_actual: str,
    faltantes: set[str],
    cliente,
    modelo: str,
    resultado: dict[str, Any],
) -> tuple[str | None, bool]:
    """Pide una única cita más larga cuando la anterior no cubre las cifras."""
    prompt = (
        "La cita de evidencia propuesta es demasiado corta y no contiene todas "
        "las cifras de la afirmación. Devuelve ÚNICAMENTE un JSON con la clave "
        "EXACTA 'evidencia', cuyo valor sea una CITA LITERAL y CONTINUA copiada "
        "exactamente del CONTENIDO_FUENTE. La cita debe ser lo más larga posible "
        "y debe incluir TODAS estas cifras de la afirmación: "
        f"{', '.join(sorted(faltantes))}. No parafrasees, no omitas palabras "
        "intermedias y no unas pasajes separados con conectores. Si no existe "
        "una cita literal que cubra esas cifras, devuelve una cadena vacía.\n\n"
        f"AFIRMACIÓN:\n{afirmacion}\n\n"
        f"FUENTE_PRIMARIA_URL:\n{fuente_url}\n\n"
        f"EVIDENCIA_ACTUAL:\n{evidencia_actual}\n\n"
        f"CONTENIDO_FUENTE:\n{texto_fuente}\n"
    )
    datos, limite = _llamar_groq_json(
        cliente,
        modelo,
        "Eres un verificador estricto. Copias citas literales de la fuente. "
        "Escribes en español. Respondes solo JSON válido.",
        prompt,
        resultado,
        "ampliando evidencia",
    )
    if limite:
        resultado["limite_api"] = True
        return None, True
    if not isinstance(datos, dict):
        return None, False
    evidencia = datos.get("evidencia") or ""
    if not isinstance(evidencia, str):
        return None, False
    return evidencia.strip(), False


def _resultado_no_verificado(
    afirmacion: str,
    fuente_url: str = "",
    cobertura: str = "",
    datos_cmp: dict[str, Any] | None = None,
    motivo: str = "",
) -> dict[str, Any]:
    res = {
        "texto": afirmacion,
        "tipo": (datos_cmp.get("tipo") if isinstance(datos_cmp, dict) else None) or "otro",
        "fidelidad": "no aparece en la fuente citada",
        "respaldo": "no verificado",
        "fuente_primaria_url": fuente_url,
        "evidencia": "",
        "matices_omitidos": [],
        "confianza": 0.0,
        "revisado_humano": False,
        "fuente_cobertura": cobertura,
    }
    if datos_cmp is not None:
        res["veredicto_modelo_sin_validar"] = datos_cmp
    res["motivo_rechazo"] = (
        motivo or "No se pudo validar el veredicto con la evidencia disponible."
    )
    return res


def _resultado_limite_api(
    afirmacion: str,
    fuente_url: str = "",
    cobertura: str = "",
    datos_cmp: dict[str, Any] | None = None,
) -> dict[str, Any]:
    res = {
        "texto": afirmacion,
        "tipo": (datos_cmp.get("tipo") if isinstance(datos_cmp, dict) else None) or "otro",
        "fidelidad": "no verificable (límite de la API)",
        "respaldo": "limite_api",
        "fuente_primaria_url": fuente_url,
        "evidencia": "",
        "matices_omitidos": [],
        "confianza": 0.0,
        "revisado_humano": False,
        "fuente_cobertura": cobertura,
        "motivo_rechazo": (
            "No se pudo verificar: la API alcanzó su límite de peticiones (429)."
        ),
    }
    if datos_cmp is not None:
        res["veredicto_modelo_sin_validar"] = datos_cmp
    return res


def _resultado_verificado(
    afirmacion: str,
    datos_cmp: dict[str, Any],
    evidencia: str,
    matices: list[dict[str, str]],
    fuente_url: str,
    cobertura: str,
) -> dict[str, Any]:
    try:
        confianza_val = float(datos_cmp.get("confianza") or 0.0)
    except Exception:
        confianza_val = 0.0
    res = {
        "texto": afirmacion,
        "tipo": datos_cmp.get("tipo") or "otro",
        "fidelidad": datos_cmp.get("fidelidad") or "no aparece en la fuente citada",
        "respaldo": datos_cmp.get("respaldo") or "no verificado",
        "fuente_primaria_url": fuente_url,
        "evidencia": evidencia,
        "matices_omitidos": matices,
        "confianza": confianza_val,
        "revisado_humano": False,
        "fuente_cobertura": cobertura,
    }
    explicacion = datos_cmp.get("explicacion_parcial") or ""
    if isinstance(explicacion, str) and explicacion.strip():
        res["explicacion_parcial"] = explicacion.strip()
    if str(res["respaldo"]).strip().lower() == "no verificado":
        res["veredicto_modelo_sin_validar"] = datos_cmp
        res["motivo_rechazo"] = "El modelo no encontró respaldo suficiente en la fuente."
    return res


def verificar_item(item) -> dict[str, Any]:
    """Verifica afirmaciones comprobables de un ítem."""
    cargar_env()
    clave, modelo_verif, usado_defecto = config_groq_verificador()
    resultado = {
        "item_titulo": getattr(item, "titulo", ""),
        "modelo_verificador": modelo_verif,
        "modelo_caido_a_defecto": usado_defecto,
        "fuente_cobertura": "",
        "afirmaciones": [],
        "errores": [],
    }

    if Groq is None or not clave:
        resultado["errores"].append("Groq no disponible o GROQ_API_KEY no configurada")
        return resultado

    try:
        cliente = Groq(api_key=clave)
    except Exception as e:
        resultado["errores"].append(f"No se pudo crear cliente Groq: {e}")
        return resultado

    llamadas_total = 0
    afirmaciones = []
    try:
        llamadas_total += 1
        afirmaciones = _extraer_afirmaciones_groq(item, cliente, modelo_verif, resultado)
    except RateLimitError as e:
        resultado["errores"].append(f"RateLimitError extrayendo afirmaciones: {e}")
        return resultado
    except Exception as e:
        resultado["errores"].append(f"Error extrayendo afirmaciones: {e}")
        return resultado

    if not afirmaciones:
        if not resultado.get("limite_api") and not resultado["errores"]:
            resultado["errores"].append("No se extrajeron afirmaciones comprobables")
        return resultado

    session = sesion_http()
    try:
        fuente_url_elegida, texto_fuente, cobertura = _resolver_fuente_item(
            item, session, resultado
        )
    except Exception as e:
        resultado["errores"].append(f"Error obteniendo fuente primaria: {e}")
        fuente_url_elegida, texto_fuente, cobertura = "", "", "sin_fuente"

    resultado["fuente_cobertura"] = cobertura

    if not texto_fuente:
        for afirmacion in afirmaciones:
            resultado["afirmaciones"].append(
                _resultado_no_verificado(
                    afirmacion,
                    fuente_url_elegida,
                    cobertura,
                    motivo="No se pudo obtener texto de la fuente primaria.",
                )
            )
        return resultado

    for afirmacion in afirmaciones:
        llamadas_afirmacion = 0
        limite_api_afirmacion = False

        # Comparar con modelo - contar llamadas por afirmación
        try:
            llamadas_afirmacion += 1
            if llamadas_afirmacion > MAX_LLAMADAS_HERRAMIENTA_POR_AFIRMACION:
                resultado["afirmaciones"].append(
                    _resultado_no_verificado(
                        afirmacion,
                        fuente_url_elegida,
                        cobertura,
                        motivo="Se superó el límite de llamadas para verificar la afirmación.",
                    )
                )
                continue
            texto_fuente_seleccionado = _seleccionar_fragmentos_relevantes(
                texto_fuente, afirmacion
            )
            datos_cmp, limite_api_afirmacion = _comparar_afirmacion_groq(
                afirmacion,
                texto_fuente_seleccionado,
                fuente_url_elegida,
                cliente,
                modelo_verif,
                resultado,
            )
        except RateLimitError as e:
            resultado["errores"].append(f"RateLimitError comparando afirmación: {e}")
            limite_api_afirmacion = True
            datos_cmp = None
        except Exception as e:
            resultado["errores"].append(f"Error comparando afirmación: {e}")
            datos_cmp = None

        if limite_api_afirmacion:
            resultado["afirmaciones"].append(
                _resultado_limite_api(
                    afirmacion,
                    fuente_url_elegida,
                    cobertura,
                    datos_cmp,
                )
            )
            continue

        if not isinstance(datos_cmp, dict) or not datos_cmp:
            resultado["afirmaciones"].append(
                _resultado_no_verificado(
                    afirmacion,
                    fuente_url_elegida,
                    cobertura,
                    motivo="El modelo no devolvió un veredicto válido.",
                )
            )
            continue

        # Validaciones por código
        evidencia = datos_cmp.get("evidencia") or ""
        evidencia = evidencia.strip() if isinstance(evidencia, str) else ""
        evidencia_limpia = _limpiar_evidencia(evidencia)

        fuente_final = fuente_url_elegida
        motivo_rechazo = ""
        respaldo_norm = (datos_cmp.get("respaldo") or "").strip().lower()

        if not evidencia:
            motivo_rechazo = "El modelo no proporcionó evidencia literal de la fuente."
        elif not _validar_evidencia_en_fuente(evidencia, texto_fuente):
            motivo_rechazo = "La evidencia no es una cita literal de la fuente."
        elif respaldo_norm == "respaldado":
            faltan = _numeros_de_texto(afirmacion) - _numeros_de_texto(evidencia_limpia)
            if faltan:
                # Una sola petición adicional de una cita más larga.
                if llamadas_afirmacion < MAX_LLAMADAS_HERRAMIENTA_POR_AFIRMACION:
                    llamadas_afirmacion += 1
                    nueva_evidencia, limite_api_afirmacion = (
                        _pedir_evidencia_mas_larga(
                            afirmacion,
                            texto_fuente_seleccionado,
                            fuente_url_elegida,
                            evidencia,
                            faltan,
                            cliente,
                            modelo_verif,
                            resultado,
                        )
                    )
                    if limite_api_afirmacion:
                        resultado["afirmaciones"].append(
                            _resultado_limite_api(
                                afirmacion,
                                fuente_url_elegida,
                                cobertura,
                                datos_cmp,
                            )
                        )
                        continue
                    if nueva_evidencia:
                        evidencia = nueva_evidencia
                        evidencia_limpia = _limpiar_evidencia(evidencia)
                        if not _validar_evidencia_en_fuente(
                            evidencia, texto_fuente
                        ):
                            motivo_rechazo = (
                                "La evidencia ampliada no es una cita literal "
                                "de la fuente."
                            )
                        else:
                            faltan = _numeros_de_texto(
                                afirmacion
                            ) - _numeros_de_texto(evidencia_limpia)
                            if faltan:
                                motivo_rechazo = (
                                    "La evidencia no contiene todas las cifras de "
                                    "la afirmación: "
                                    + ", ".join(sorted(faltan))
                                    + "."
                                )
                    else:
                        motivo_rechazo = (
                            "El modelo no proporcionó una cita literal más larga "
                            "que cubriera todas las cifras."
                        )
                else:
                    motivo_rechazo = (
                        "La evidencia no contiene todas las cifras de la "
                        "afirmación y se superó el límite de llamadas: "
                        + ", ".join(sorted(faltan))
                        + "."
                    )

        if motivo_rechazo:
            resultado["afirmaciones"].append(
                _resultado_no_verificado(
                    afirmacion,
                    fuente_final,
                    cobertura,
                    datos_cmp=datos_cmp,
                    motivo=motivo_rechazo,
                )
            )
            continue

        # Validar matices
        matices_validados = _validar_matices(
            datos_cmp.get("matices_omitidos") or [],
            texto_fuente,
            afirmacion,
            evidencia_limpia,
        )

        resultado["afirmaciones"].append(
            _resultado_verificado(
                afirmacion,
                datos_cmp,
                evidencia_limpia,
                matices_validados,
                fuente_final,
                cobertura,
            )
        )

    return resultado
