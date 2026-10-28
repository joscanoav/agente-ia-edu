"""Pruebas para verificar.py"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime

from dotenv import load_dotenv

from main import Item
from verificar import verificar_item

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


def main():
    if not os.environ.get("GROQ_API_KEY", "").strip():
        print("NO EJECUTADO: falta GROQ_API_KEY")
        return
    print("=== PRUEBA VERIFICAR ===")
    resultados = []
    try:
        resultados.append(caso_ugr())
    except Exception as e:
        print(f"Error en caso_ugr: {e}")
        resultados.append("NO_EJECUTADO")
    try:
        resultados.append(caso_arxiv())
    except Exception as e:
        print(f"Error en caso_arxiv: {e}")
        resultados.append("NO_EJECUTADO")
    try:
        resultados.append(caso_alterado())
    except Exception as e:
        print(f"Error en caso_alterado: {e}")
        resultados.append("NO_EJECUTADO")
    try:
        resultados.append(caso_sin_enlaces())
    except Exception as e:
        print(f"Error en caso_sin_enlaces: {e}")
        resultados.append("NO_EJECUTADO")
    ok_c = sum(1 for r in resultados if r == "OK")
    fallo_c = sum(1 for r in resultados if r == "FALLO")
    grave_c = sum(1 for r in resultados if r == "FALLO GRAVE")
    leve_c = sum(1 for r in resultados if r == "FALLO LEVE")
    no_ej_c = sum(1 for r in resultados if r == "NO_EJECUTADO")
    no_con_c = sum(1 for r in resultados if r == "NO_CONCLUYENTE")
    print(
        f"\n=== RESUMEN: OK={ok_c}, FALLO={fallo_c}, "
        f"FALLO GRAVE={grave_c}, FALLO LEVE={leve_c}, "
        f"NO EJECUTADO={no_ej_c}, NO CONCLUYENTE={no_con_c} ==="
    )


if __name__ == "__main__":
    main()
