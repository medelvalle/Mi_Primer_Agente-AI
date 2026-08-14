"""
CHECKPOINT 1 — Un agente en ~40 líneas.

Un agente no es una librería. Es un while.

Este archivo hace dos cosas, con la MISMA pregunta y el MISMO modelo:
  PARTE 1: le pregunta al modelo sin darle herramientas  -> inventa el número
  PARTE 2: le da una herramienta y corre el loop a mano  -> calcula el número

Lo único que cambia entre las dos partes es el parámetro `tools`.

    python cp1_loop.py
"""

import json
import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

MODELO = "openai/gpt-oss-120b"
cliente = Groq(api_key=os.environ["GROQ_API_KEY"])

PREGUNTA = "¿Cuánto me sale el NBL-204 en línea con la beca de egresados?"

# Este system prompt es exigente a propósito, y es EL MISMO en las dos partes.
# Con instrucciones más suaves el modelo simplemente contesta "no tengo esa
# información", que no es mentira pero tampoco es útil. Así, empujado a dar una
# cifra —como lo empujan casi todos los prompts de producto reales—, el modelo
# obedece y se inventa el número. Ese es el punto de la Parte 1.
INSTRUCCIONES = (
    "Eres el asistente de pagos del Instituto Nébula. Das siempre la cifra exacta "
    "en pesos mexicanos, con el desglose. Nunca respondas que no tienes la "
    "información ni sugieras consultar a otra persona: el alumno espera un número."
)


# =====================================================================================
# LA HERRAMIENTA
# =====================================================================================

# Precios sin IVA, en pesos mexicanos. Esta tabla es idéntica a la que está publicada
# en docs/becas_y_pagos.md. Si una cambia y la otra no, el agente se contradice.
PRECIOS = {
    "NBL-101": {"nombre": "Fundamentos de IA Aplicada", "presencial": 6800, "linea": 4900},
    "NBL-204": {"nombre": "Recuperación Aumentada (RAG) en Producción", "presencial": 9500, "linea": 7200},
    "NBL-210": {"nombre": "Agentes Autónomos con Herramientas", "presencial": 11000, "linea": 8400},
    "NBL-305": {"nombre": "Evaluación y Observabilidad de Sistemas LLM", "presencial": 8900, "linea": 6500},
    "NBL-402": {"nombre": "Despliegue y Costos de Modelos en la Nube", "presencial": 12500, "linea": 9800},
}

BECAS = {
    "egresados": {"nombre": "Beca Egresado Nébula", "porcentaje": 0.25, "solo_en_linea": False},
    "excelencia": {"nombre": "Beca de Excelencia Académica", "porcentaje": 0.40, "solo_en_linea": False},
    "comunidad": {"nombre": "Beca Comunidad Abierta", "porcentaje": 0.15, "solo_en_linea": True},
}

IVA = 0.16


def calcular_costo(codigo_curso, modalidad, tipo_beca=None):
    """Devuelve el desglose del costo de un curso, línea por línea."""
    codigo = codigo_curso.strip().upper()
    if codigo not in PRECIOS:
        return f"No existe el curso {codigo}. Códigos válidos: {', '.join(PRECIOS)}."

    # El modelo puede escribir "en línea", "online", "En Línea"... normalizamos.
    texto_modalidad = modalidad.strip().lower().replace("í", "i")
    if "presencial" in texto_modalidad:
        clave_modalidad, etiqueta_modalidad = "presencial", "presencial"
    elif "linea" in texto_modalidad or "online" in texto_modalidad or "virtual" in texto_modalidad:
        clave_modalidad, etiqueta_modalidad = "linea", "en línea"
    else:
        return f"Modalidad '{modalidad}' no reconocida. Usa 'presencial' o 'en línea'."

    curso = PRECIOS[codigo]
    base = curso[clave_modalidad]

    lineas = [
        f"{codigo} — {curso['nombre']} ({etiqueta_modalidad})",
        f"  {'Precio base':<34} ${base:>9,.2f}",
    ]

    descuento = 0.0
    if tipo_beca:
        texto_beca = tipo_beca.strip().lower()
        clave_beca = next((k for k in BECAS if k in texto_beca or texto_beca in k), None)
        if clave_beca is None:
            return f"No existe la beca '{tipo_beca}'. Becas válidas: {', '.join(BECAS)}."
        beca = BECAS[clave_beca]
        if beca["solo_en_linea"] and clave_modalidad != "linea":
            lineas.append(
                f"  {beca['nombre']}: NO APLICA — solo es válida en modalidad en línea."
            )
        else:
            descuento = base * beca["porcentaje"]
            etiqueta_beca = f"{beca['nombre']} ({beca['porcentaje']:.0%})"
            lineas.append(f"  {etiqueta_beca:<34}-${descuento:>9,.2f}")

    subtotal = base - descuento
    impuesto = subtotal * IVA
    total = subtotal + impuesto

    lineas.append(f"  {'Subtotal':<34} ${subtotal:>9,.2f}")
    lineas.append(f"  {f'IVA ({IVA:.0%})':<34} ${impuesto:>9,.2f}")
    lineas.append("  " + "─" * 45)
    lineas.append(f"  {'TOTAL A PAGAR':<34} ${total:>9,.2f}")
    return "\n".join(lineas)


# El esquema JSON no es burocracia: es el prompt que el modelo lee para decidir si
# llama a esta herramienta y con qué argumentos. Cada descripción cuenta.
ESQUEMA_HERRAMIENTAS = [
    {
        "type": "function",
        "function": {
            "name": "calcular_costo",
            "description": (
                "Calcula el costo total de un curso del Instituto Nébula, con el desglose "
                "línea por línea: precio base, descuento de beca, subtotal, IVA y total. "
                "Úsala siempre que te pregunten un precio, un costo o cuánto sale algo. "
                "Nunca calcules precios de memoria."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "codigo_curso": {
                        "type": "string",
                        "description": "Código del curso, por ejemplo 'NBL-204'.",
                    },
                    "modalidad": {
                        "type": "string",
                        "enum": ["presencial", "en línea"],
                        "description": (
                            "Modalidad en la que se cursa. Si la persona no la menciona, "
                            "usa 'presencial'."
                        ),
                    },
                    "tipo_beca": {
                        "type": "string",
                        "enum": ["egresados", "excelencia", "comunidad"],
                        "description": (
                            "Beca que se quiere aplicar. Omite este parámetro si la "
                            "persona no menciona ninguna beca."
                        ),
                    },
                },
                "required": ["codigo_curso", "modalidad"],
            },
        },
    }
]

# El despachador: del nombre que dijo el modelo, a la función de Python.
HERRAMIENTAS = {"calcular_costo": calcular_costo}


# =====================================================================================
# PARTE 1 — El mismo modelo, SIN herramientas
# =====================================================================================

print("=" * 79)
print("PARTE 1 — Sin herramientas. El modelo responde de memoria.")
print("=" * 79)
print(f"\nPregunta: {PREGUNTA}\n")

respuesta_sin_herramientas = cliente.chat.completions.create(
    model=MODELO,
    messages=[
        {"role": "system", "content": INSTRUCCIONES},
        {"role": "user", "content": PREGUNTA},
    ],
    temperature=0,
    seed=42,
    include_reasoning=False,
)

print(respuesta_sin_herramientas.choices[0].message.content)
print("\n>>> El Instituto Nébula no existe. Cualquier número de arriba es inventado.\n")


# =====================================================================================
# PARTE 2 — El loop del agente
# =====================================================================================

print("=" * 79)
print("PARTE 2 — Con una herramienta. El loop del agente.")
print("=" * 79)
print(f"\nPregunta: {PREGUNTA}\n")


def imprimir_historial(mensajes):
    """Imprime la lista de mensajes completa, que es TODO el estado del agente."""
    print("-" * 79)
    print(f"HISTORIAL ({len(mensajes)} mensajes)")
    print("-" * 79)
    for i, mensaje in enumerate(mensajes):
        rol = mensaje["role"].upper()
        if mensaje.get("tool_calls"):
            for llamada in mensaje["tool_calls"]:
                nombre = llamada["function"]["name"]
                argumentos = llamada["function"]["arguments"]
                print(f"[{i}] {rol:<9} -> pide {nombre}({argumentos})")
        elif mensaje["role"] == "tool":
            primera_linea = mensaje["content"].splitlines()[0]
            print(f"[{i}] {rol:<9} <- {primera_linea}")
            for linea in mensaje["content"].splitlines()[1:]:
                print(f"    {'':<9}    {linea}")
        else:
            print(f"[{i}] {rol:<9}    {mensaje['content']}")
    print("-" * 79 + "\n")


# --- El agente empieza aquí ----------------------------------------------------------

mensajes = [
    {"role": "system", "content": INSTRUCCIONES},
    {"role": "user", "content": PREGUNTA},
]

vuelta = 0
while True:
    vuelta += 1
    print(f"### VUELTA {vuelta} — le mandamos {len(mensajes)} mensajes al modelo")

    respuesta = cliente.chat.completions.create(
        model=MODELO,
        messages=mensajes,
        tools=ESQUEMA_HERRAMIENTAS,  # <- lo ÚNICO que cambia respecto de la Parte 1
        temperature=0,
        seed=42,
        include_reasoning=False,
    )
    mensaje = respuesta.choices[0].message

    # Guardamos lo que dijo el modelo en el historial, tal cual.
    mensajes.append(
        {
            "role": "assistant",
            "content": mensaje.content,
            "tool_calls": [
                {
                    "id": llamada.id,
                    "type": "function",
                    "function": {
                        "name": llamada.function.name,
                        "arguments": llamada.function.arguments,
                    },
                }
                for llamada in (mensaje.tool_calls or [])
            ],
        }
    )

    # ¿Pidió herramientas? Si no, terminó: esa es su respuesta final.
    if not mensaje.tool_calls:
        imprimir_historial(mensajes)
        print("RESPUESTA FINAL:\n")
        print(mensaje.content)
        break

    # Sí pidió. Ejecutamos cada herramienta y le devolvemos el resultado.
    for llamada in mensaje.tool_calls:
        nombre = llamada.function.name
        argumentos = json.loads(llamada.function.arguments)  # llega como texto JSON
        print(f"    ejecutando {nombre}({argumentos})")

        resultado = HERRAMIENTAS[nombre](**argumentos)

        mensajes.append(
            {
                "role": "tool",
                "tool_call_id": llamada.id,  # amarra el resultado con la petición
                "name": nombre,
                "content": resultado,
            }
        )

    imprimir_historial(mensajes)
