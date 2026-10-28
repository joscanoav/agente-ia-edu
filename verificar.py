"""Módulo de verificación de afirmaciones (fase 2).

Verifica afirmaciones comprobables de ítems ya seleccionados siguiendo
los criterios de VERIFICACION.md.
"""
from __future__ import annotations

import io
import json
import os
import re
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


def _seleccionar_fragmentos_relevantes(texto_fuente: str, afirmacion: str) -> str:
    if not texto_fuente:
        return ""
    if not afirmacion:
        return recortar_fuente(texto_fuente)
    # Palabras clave simples
    palabras = re.findall(r"\b\w{4,}\b", afirmacion.lower())
    if not palabras:
        return recortar_fuente(texto_fuente)
    oraciones = re.split(r"(?<=[.!?])\s+", texto_fuente)
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
    ultimo = None
    for m in re.finditer(r"[.!?](?=\s|$)", fragmento):
        ultimo = m.end()
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

    datos_item, tipo_item, content_type_item = descargar(url_item)
    if datos_item is None:
        if tipo_item.startswith("error:"):
            resultado["errores"].append(f"Error descargando {url_item}: {tipo_item}")
        return "", "", "sin_fuente"
    if tipo_item == "pdf" or datos_item[:4] == b"%PDF":
        return "", "", "sin_fuente"

    html_item = _bytes_a_texto(datos_item, content_type_item)
    if not html_item:
        return "", "", "sin_fuente"

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
                "revistas",
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

    return "", "", "sin_fuente"


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
    try:
        resp = cliente.chat.completions.create(
            model=modelo,
            messages=[
                {"role": "system", "content": "Eres un extractor estricto que escribe siempre en español. Respondes solo con JSON válido."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        contenido_resp = resp.choices[0].message.content or "{}"
        datos = json.loads(contenido_resp)
        afirmaciones = datos.get("afirmaciones") or []
        res = []
        for a in afirmaciones:
            if isinstance(a, str) and a.strip():
                a = a.strip()
                if _es_afirmacion_autosuficiente(a):
                    res.append(a)
        return res[:3]
    except RateLimitError as e:
        resultado["errores"].append(f"RateLimitError extrayendo afirmaciones: {e}")
        return []
    except Exception as e:
        resultado["errores"].append(f"Error extrayendo afirmaciones: {e}")
        return []


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


def _normalizar_numero(numero: str) -> str:
    n = (numero or "").strip()
    if re.fullmatch(r"\d{1,3}(?:[.,]\d{3})+", n):
        n = re.sub(r"[.,]", "", n)
    else:
        n = n.replace(",", ".")
        if "." in n:
            n = n.rstrip("0").rstrip(".")
    return n


def _numeros_de_texto(texto: str) -> set[str]:
    if not texto:
        return set()
    encontrados = re.findall(r"(?<!\d)\d(?:[\d.,]*\d)?(?!\d)", texto)
    return {_normalizar_numero(n) for n in encontrados if n}


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
) -> dict[str, Any] | None:
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
    for intento in range(2):  # reintento si falla JSON
        try:
            resp = cliente.chat.completions.create(
                model=modelo,
                messages=[
                    {"role": "system", "content": "Eres un verificador estricto. Solo usas la fuente proporcionada. Escribes en español. Respondes solo JSON válido."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )
            contenido_resp = resp.choices[0].message.content or "{}"
            datos = json.loads(contenido_resp)
            return datos
        except RateLimitError as e:
            resultado["errores"].append(f"RateLimitError comparando afirmación: {e}")
            if intento == 1:
                return None
            continue
        except Exception as e:
            resultado["errores"].append(f"Error comparando afirmación: {e}")
            if intento == 1:
                return None
            continue
    return None


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
        if not resultado["errores"]:
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
            datos_cmp = _comparar_afirmacion_groq(
                afirmacion,
                texto_fuente_seleccionado,
                fuente_url_elegida,
                cliente,
                modelo_verif,
                resultado,
            )
        except RateLimitError as e:
            resultado["errores"].append(f"RateLimitError comparando afirmación: {e}")
            datos_cmp = None
        except Exception as e:
            resultado["errores"].append(f"Error comparando afirmación: {e}")
            datos_cmp = None

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
                motivo_rechazo = (
                    "La evidencia no contiene todas las cifras de la afirmación: "
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
