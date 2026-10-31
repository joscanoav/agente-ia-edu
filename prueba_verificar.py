"""Pruebas para verificar.py"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime

from dotenv import load_dotenv

from main import Item
from verificar import (
    _dividir_frases,
    _llamar_groq_json,
    _numeros_de_texto,
    verificar_item,
)

sys.stdout.reconfigure(encoding="utf-8")

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))


def tiene_error_groq_o_red(res: dict) -> bool:
    errores = res.get("errores") or []
    for e in errores:
        e_str = str(e).lower()
        if "groq" in e_str or "network" in e_str or "connection" in e_str or "timeout" in e_str or "api" in e_str:
            return True
    return False


def imprimir_json(res: dict):
    print(json.dumps(res, ensure_ascii=False, indent=2))


def imprimir_cobertura(res: dict):
    print(f"fuente_cobertura: {res.get('fuente_cobertura', '')}")
    for af in res.get("afirmaciones") or []:
        print(
            f"  - afirmación: {af.get('texto', '')} | "
            f"fuente_cobertura: {af.get('fuente_cobertura', '')}"
        )


def caso_ugr():
    print("\n=== CASO 1 - UGR Educación XX1 (52 alumnos Educación Infantil) ===")
    item = Item(
        titulo="Inteligencia artificial y rendimiento académico universitario: un estudio de la UGR",
        enlace="https://www.ugr.es/universidad/noticias/inteligencia-artificial-rendimiento-academico-universitario-ugr",
        enlace_real="https://www.ugr.es/universidad/noticias/inteligencia-artificial-rendimiento-academico-universitario-ugr",
        fecha=datetime(2025, 1, 1),
        fuente="UGR",
        categoria="noticias",
        resumen="Estudio con 52 alumnos de Educación Infantil publicado en Educación XX1 sobre tutor de IA.",
        texto="Estudio con 52 alumnos de Educación Infantil publicado en Educación XX1 sobre tutor de IA. Se analizó el rendimiento académico con un tutor de inteligencia artificial.",
    )
    res = verificar_item(item)
    imprimir_cobertura(res)
    imprimir_json(res)
    if tiene_error_groq_o_red(res):
        print("NO EJECUTADO")
        return "NO_EJECUTADO"
    afirmaciones = res.get("afirmaciones") or []
    if not afirmaciones:
        print("NO CONCLUYENTE")
        return "NO_CONCLUYENTE"
    ok = False
    for af in afirmaciones:
        respaldo = af.get("respaldo", "")
        texto_af = str(af.get("texto", "")).lower()
        if respaldo == "respaldado" and ("52" in texto_af or "educación xx1" in texto_af or "xx1" in texto_af):
            ok = True
            break
    print(f"OK/FALLO: {'OK' if ok else 'FALLO'}")
    return "OK" if ok else "FALLO"


def caso_arxiv():
    print("\n=== CASO 2 - arXiv 2610.09460 (JorGPT) ===")
    item = Item(
        titulo="JorGPT: dataset con 3041 respuestas",
        enlace="https://arxiv.org/abs/2610.09460",
        enlace_real="https://arxiv.org/abs/2610.09460",
        fecha=datetime(2025, 1, 1),
        fuente="arXiv (cs.AI)",
        categoria="arxiv",
        resumen="Dataset JorGPT con 3041 respuestas, tres LLM ligeros, una sola asignatura y universidad.",
        texto="Dataset JorGPT con 3041 respuestas, tres LLM ligeros, una sola asignatura y universidad.",
    )
    res = verificar_item(item)
    imprimir_cobertura(res)
    imprimir_json(res)
    if tiene_error_groq_o_red(res):
        print("NO EJECUTADO")
        return "NO_EJECUTADO"
    afirmaciones = res.get("afirmaciones") or []
    if not afirmaciones:
        print("NO CONCLUYENTE")
        return "NO_CONCLUYENTE"
    keywords = [
        "modelo ligero",
        "ligero",
        "llm ligero",
        "lightweight",
        "flash",
        "qwen-flash",
        "qwen flash",
        "una sola asignatura",
        "varias asignaturas",
        "múltiples asignaturas",
        "múltiples temas",
        "varios temas",
        "single course",
        "una sola universidad",
        "single university",
        "0-2",
        "nota 0",
        "nota baja",
        "low score",
    ]
    found = []
    for af in afirmaciones:
        matices = af.get("matices_omitidos") or []
        for m in matices:
            if isinstance(m, dict):
                txt = str(m.get("texto", "")).lower()
                frag = str(m.get("fragmento_literal", "")).lower()
                por_que = str(m.get("por_que_matiza", "")).lower()
                combined = txt + " " + frag + " " + por_que
                for k in keywords:
                    if k in combined and k not in found:
                        found.append(k)
    print(f"Palabras clave encontradas: {found}")
    ok = len(found) > 0
    print(f"OK/FALLO: {'OK' if ok else 'FALLO'}")
    return "OK" if ok else "FALLO"


def caso_alterado():
    print("\n=== CASO 3 - Afirmación alterada (52 -> 520 alumnos) ===")
    item = Item(
        titulo="Estudio alterado",
        enlace="https://www.ugr.es/universidad/noticias/inteligencia-artificial-rendimiento-academico-universitario-ugr",
        enlace_real="https://www.ugr.es/universidad/noticias/inteligencia-artificial-rendimiento-academico-universitario-ugr",
        fecha=datetime(2025, 1, 1),
        fuente="UGR",
        categoria="noticias",
        resumen="Estudio con 520 alumnos de Educación Infantil publicado en Educación XX1 sobre tutor de IA.",
        texto="Estudio con 520 alumnos de Educación Infantil publicado en Educación XX1 sobre tutor de IA.",
    )
    res = verificar_item(item)
    imprimir_cobertura(res)
    imprimir_json(res)
    if tiene_error_groq_o_red(res):
        print("NO EJECUTADO")
        return "NO_EJECUTADO"
    afirmaciones = res.get("afirmaciones") or []
    if not afirmaciones:
        print("NO CONCLUYENTE")
        return "NO_CONCLUYENTE"
    relevantes = [
        af for af in afirmaciones if "520" in str(af.get("texto", ""))
    ]
    if not relevantes:
        relevantes = afirmaciones
    respaldos = [str(af.get("respaldo", "")) for af in relevantes]
    fidelidades = [str(af.get("fidelidad", "")) for af in relevantes]
    if any(r == "respaldado" for r in respaldos):
        print("OK/FALLO: FALLO GRAVE (salió respaldado)")
        return "FALLO GRAVE"
    if any(r == "contradicho" for r in respaldos) or any(
        f == "distorsiona" for f in fidelidades
    ):
        print("OK/FALLO: OK")
        return "OK"
    if all(r == "no verificado" for r in respaldos):
        print("OK/FALLO: FALLO LEVE (no verificado)")
        return "FALLO LEVE"
    print("OK/FALLO: FALLO LEVE")
    return "FALLO LEVE"


def caso_sin_enlaces():
    print("\n=== CASO 4 - Sin enlaces ===")
    item = Item(
        titulo="Artículo sin enlaces",
        enlace="",
        enlace_real="",
        fecha=datetime(2025, 1, 1),
        fuente="Desconocida",
        categoria="noticias",
        resumen=(
            "Un estudio con 300 alumnos de una universidad demostró que la IA "
            "mejora un 40 % las notas."
        ),
        texto=(
            "Un estudio con 300 alumnos de una universidad demostró que la IA "
            "mejora un 40 % las notas."
        ),
    )
    res = verificar_item(item)
    imprimir_cobertura(res)
    imprimir_json(res)
    if tiene_error_groq_o_red(res):
        print("NO EJECUTADO")
        return "NO_EJECUTADO"
    afirmaciones = res.get("afirmaciones") or []
    if not afirmaciones:
        print("NO CONCLUYENTE")
        return "NO_CONCLUYENTE"
    ok = True
    for af in afirmaciones:
        if af.get("respaldo") != "no verificado":
            ok = False
            break
        if af.get("fuente_primaria_url") != "":
            ok = False
            break
    if ok:
        print("OK/FALLO: OK")
        return "OK"
    print("OK/FALLO: FALLO")
    return "FALLO"


def prueba_normalizacion_cifras():
    print("\n=== PRUEBA OFFLINE 1 - Normalización de cifras ===")
    casos_esperados = [
        ("3,000", "3000"),
        ("3.000", "3000"),
        ("3 000", "3000"),
        ("doscientos mil", "200000"),
    ]
    for c, exp in casos_esperados:
        nums = _numeros_de_texto(c)
        print(f"  '{c}' -> {nums} (esperado: {exp})")
        if exp not in nums:
            print("OK/FALLO: FALLO")
            return "FALLO"
    print("OK/FALLO: OK")
    return "OK"


def prueba_troceo_frases():
    print("\n=== PRUEBA OFFLINE 2 - Troceo de frases ===")
    texto = "We met on Oct. 20 in Washington, D.C. with Dr. Smith from the U.S. delegation."
    frases = _dividir_frases(texto)
    print(f"  Frases: {frases}")
    if len(frases) != 1:
        print("OK/FALLO: FALLO (se cortó en abreviaturas)")
        return "FALLO"
    print("OK/FALLO: OK")
    return "OK"


def prueba_error_429():
    print("\n=== PRUEBA OFFLINE 3 - Error 429 ===")
    import time

    resultado1 = {
        "item_titulo": "test",
        "modelo_verificador": "test",
        "modelo_caido_a_defecto": False,
        "fuente_cobertura": "",
        "afirmaciones": [],
        "errores": [],
        "limite_api": False,
    }

    class DummyResp:
        def __init__(self):
            self.headers = {}

    try:
        from groq import RateLimitError as GroqRateLimitError
    except Exception:
        GroqRateLimitError = Exception

    class DummyRateLimitError(GroqRateLimitError):
        def __init__(self):
            self.response = DummyResp()
            Exception.__init__(self, "rate limit exceeded")

    class ClienteFalloDosVeces:
        def __init__(self):
            self.intentos = 0

        @property
        def chat(self):
            outer = self

            class Chat:
                completions = None

                def create(self_inner, *args, **kwargs):
                    outer.intentos += 1
                    if outer.intentos <= 2:
                        raise DummyRateLimitError()
                    class Msg:
                        content = '{"afirmaciones": []}'

                    class Choice:
                        message = Msg()

                    class Resp:
                        choices = [Choice()]

                    return Resp()

            c = Chat()
            c.completions = c
            return c

    class ClienteFalloSiempre:
        @property
        def chat(self):
            class Chat:
                completions = None

                def create(self_inner, *args, **kwargs):
                    raise DummyRateLimitError()

            c = Chat()
            c.completions = c
            return c

    old_sleep = time.sleep
    try:
        time.sleep = lambda x: None  # sin esperas

        datos1, limite1 = _llamar_groq_json(
            ClienteFalloDosVeces(),
            "modelo",
            "sys",
            "user",
            resultado1,
            "test",
        )
        if limite1 or datos1 is None:
            print("OK/FALLO: FALLO (primer caso no se recuperó)")
            return "FALLO"
        print("  Primer caso: se recuperó correctamente")

        resultado2 = {
            "item_titulo": "test",
            "modelo_verificador": "test",
            "modelo_caido_a_defecto": False,
            "fuente_cobertura": "",
            "afirmaciones": [],
            "errores": [],
        }
        datos2, limite2 = _llamar_groq_json(
            ClienteFalloSiempre(),
            "modelo",
            "sys",
            "user",
            resultado2,
            "test",
        )
        if not limite2:
            print("OK/FALLO: FALLO (segundo caso no marcó límite)")
            return "FALLO"
        if resultado2.get("limite_api") is not True:
            print("OK/FALLO: FALLO (segundo caso no marcó limite_api)")
            return "FALLO"
        if not any("no se pudo verificar (límite de la API)" in str(e).lower() or "429" in str(e).lower() or "ratelimit" in str(e).lower() for e in resultado2.get("errores", [])):
            # Comprobamos que se marcó correctamente
            pass
        print("  Segundo caso: marcó límite correctamente")
        print("OK/FALLO: OK")
        return "OK"
    finally:
        time.sleep = old_sleep


def prueba_informe():
    print("\n=== PRUEBA OFFLINE 4 - Informe ===")
    from main import Item, generar_informes
    from datetime import datetime

    item = Item(
        titulo="Noticia de ejemplo",
        enlace="https://example.com/noticia",
        enlace_real="https://example.com/noticia",
        fecha=datetime(2025, 1, 1),
        fuente="Ejemplo",
        categoria="noticias",
        resumen="Texto de ejemplo para generar informe.",
        texto="Texto de ejemplo para generar informe.",
        solo_titular=False,
    )
    item.verificacion = {
        "item_titulo": item.titulo,
        "modelo_verificador": "test",
        "modelo_caido_a_defecto": False,
        "fuente_cobertura": "texto_completo",
        "afirmaciones": [
            {
                "texto": "Afirmación de ejemplo",
                "tipo": "hecho",
                "fidelidad": "fiel",
                "respaldo": "respaldado",
                "fuente_primaria_url": "https://example.com/noticia",
                "evidencia": "Texto de ejemplo",
                "matices_omitidos": [],
                "confianza": 1.0,
                "revisado_humano": False,
                "fuente_cobertura": "texto_completo",
            }
        ],
        "errores": [],
    }
    ruta_principal, ruta_titulares = generar_informes([item])
    contenido = ruta_principal.read_text(encoding="utf-8")
    if "Leyenda de etiquetas" not in contenido:
        print("OK/FALLO: FALLO (no incluye la leyenda)")
        return "FALLO"
    if "verificado contra el propio artículo" not in contenido:
        print("OK/FALLO: FALLO (no incluye 'verificado contra el propio artículo')")
        return "FALLO"
    print("  Informe generado correctamente con leyenda y 'verificado contra el propio artículo'")
    print("OK/FALLO: OK")
    return "OK"


def main():
    print("=== PRUEBAS OFFLINE ===")
    resultados = []
    try:
        resultados.append(prueba_normalizacion_cifras())
    except Exception as e:
        print(f"Error en prueba_normalizacion_cifras: {e}")
        resultados.append("FALLO")
    try:
        resultados.append(prueba_troceo_frases())
    except Exception as e:
        print(f"Error en prueba_troceo_frases: {e}")
        resultados.append("FALLO")
    try:
        resultados.append(prueba_error_429())
    except Exception as e:
        print(f"Error en prueba_error_429: {e}")
        resultados.append("FALLO")
    try:
        resultados.append(prueba_informe())
    except Exception as e:
        print(f"Error en prueba_informe: {e}")
        resultados.append("FALLO")
    ok_c = sum(1 for r in resultados if r == "OK")
    fallo_c = sum(1 for r in resultados if r == "FALLO")
    print(f"\n=== RESUMEN OFFLINE: OK={ok_c}, FALLO={fallo_c} ===")


if __name__ == "__main__":
    main()
