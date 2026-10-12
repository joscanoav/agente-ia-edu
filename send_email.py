"""Envía por correo el informe principal más reciente de `informes/`.

Usa solo la librería estándar (`smtplib`, `ssl`, `email.message`) y
`python-dotenv`. Variables de entorno necesarias:

- ``GMAIL_USER``          cuenta de Gmail remitente.
- ``GMAIL_APP_PASSWORD``  contraseña de aplicación de Gmail.
- ``MAIL_TO``             destinatarios, separados por comas.

El asunto es ``Informe IA y Educación - AAAA-MM-DD`` y el informe se adjunta
como archivo ``.md``. Si falta una variable o el envío falla, termina con
código de error.
"""

from __future__ import annotations

import os
import re
import smtplib
import ssl
import sys
from email.message import EmailMessage
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
INFORMES_DIR = BASE_DIR / "informes"

# Solo informes principales: AAAA-MM-DD.md (excluye AAAA-MM-DD-titulares.md).
PATRON_INFORME = re.compile(r"^\d{4}-\d{2}-\d{2}\.md$")
PATRON_INDICE = re.compile(r"^-\s+(.+?):\s*(\d+)\s*$")

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 465


def cargar_env() -> None:
    load_dotenv(BASE_DIR / ".env")


def informe_mas_reciente() -> Path | None:
    """Devuelve el informe principal más reciente o None si no hay ninguno."""
    if not INFORMES_DIR.exists():
        return None
    candidatos = [
        ruta
        for ruta in INFORMES_DIR.glob("*.md")
        if PATRON_INFORME.match(ruta.name)
    ]
    if not candidatos:
        return None
    return max(candidatos, key=lambda ruta: ruta.name)


def leer_indice(texto: str) -> list[tuple[str, str]]:
    """Extrae las líneas "- Sección: N" de la sección "## Índice"."""
    indice: list[tuple[str, str]] = []
    dentro = False
    for linea in texto.splitlines():
        desnuda = linea.strip()
        if desnuda.startswith("## "):
            if dentro:
                break
            dentro = desnuda.lower().startswith("## índice")
            continue
        if dentro:
            coincide = PATRON_INDICE.match(desnuda)
            if coincide:
                indice.append((coincide.group(1).strip(), coincide.group(2)))
    return indice


def construir_cuerpo(fecha: str, indice: list[tuple[str, str]]) -> str:
    lineas = [f"Informe de IA y Educación - {fecha}", "", "Ítems por sección:"]
    if indice:
        lineas += [f"- {etiqueta}: {numero}" for etiqueta, numero in indice]
    else:
        lineas.append("- (índice no disponible)")
    lineas += [
        "",
        "Se adjunta el informe completo en Markdown.",
        "",
        "Generado automáticamente por agente-ia-edu.",
    ]
    return "\n".join(lineas)


def construir_mensaje(
    remitente: str,
    destinatarios: list[str],
    ruta: Path,
    indice: list[tuple[str, str]],
) -> EmailMessage:
    fecha = ruta.stem
    mensaje = EmailMessage()
    mensaje["Subject"] = f"Informe IA y Educación - {fecha}"
    mensaje["From"] = remitente
    mensaje["To"] = ", ".join(destinatarios)
    mensaje.set_content(construir_cuerpo(fecha, indice))
    mensaje.add_attachment(
        ruta.read_text(encoding="utf-8").encode("utf-8"),
        maintype="text",
        subtype="markdown",
        filename=ruta.name,
    )
    return mensaje


def enviar(mensaje: EmailMessage, usuario: str, password: str) -> None:
    contexto = ssl.create_default_context()
    with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=contexto) as servidor:
        servidor.login(usuario, password)
        servidor.send_message(mensaje)


def main() -> int:
    cargar_env()

    usuario = os.environ.get("GMAIL_USER", "").strip()
    password = os.environ.get("GMAIL_APP_PASSWORD", "").strip()
    destinatarios = [
        correo.strip()
        for correo in os.environ.get("MAIL_TO", "").split(",")
        if correo.strip()
    ]

    faltan = []
    if not usuario:
        faltan.append("GMAIL_USER")
    if not password:
        faltan.append("GMAIL_APP_PASSWORD")
    if not destinatarios:
        faltan.append("MAIL_TO")
    if faltan:
        print(
            f"Error: faltan variables de entorno: {', '.join(faltan)}",
            file=sys.stderr,
        )
        return 1

    ruta = informe_mas_reciente()
    if ruta is None:
        print(
            "Error: no se encontró ningún informe AAAA-MM-DD.md en informes/.",
            file=sys.stderr,
        )
        return 1

    indice = leer_indice(ruta.read_text(encoding="utf-8"))
    mensaje = construir_mensaje(usuario, destinatarios, ruta, indice)

    try:
        enviar(mensaje, usuario, password)
    except Exception as exc:  # noqa: BLE001
        print(
            f"Error al enviar el correo: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1

    print(f"Informe enviado a {', '.join(destinatarios)}: {ruta.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
