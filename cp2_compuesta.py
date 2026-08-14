"""
CHECKPOINT 2 — El agente decide el orden.

Ahora hay TRES herramientas. Nosotros no le decimos cuál usar ni en qué orden:
solo se las ofrecemos y él arma el plan.

Corremos dos preguntas seguidas con exactamente el mismo agente:
  SIMPLE:    "¿cuánto cuesta el NBL-204?"                    -> 1 herramienta
  COMPUESTA: "...y qué necesito para la beca, y hay lugar?"  -> 3 herramientas

El contraste es el punto de la demo.

    python cp2_compuesta.py
"""

import json
import os
import pathlib

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from groq import Groq

load_dotenv()

MODELO = "openai/gpt-oss-120b"
MODELO_EMBEDDINGS = "gemini-embedding-001"
DIMENSIONES = 768

# Si el mejor fragmento no llega a este parecido, decimos que no sabemos.
# El número salió de medir preguntas dentro y fuera del corpus, no de adivinar.
UMBRAL_SIMILITUD = 0.62

cliente = Groq(api_key=os.environ["GROQ_API_KEY"])
cliente_gemini = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

INSTRUCCIONES = (
    "Eres el asistente del Instituto Nébula. Respondes preguntas sobre cursos, precios, "
    "becas y disponibilidad. Usa siempre las herramientas: nunca inventes precios, "
    "requisitos ni cupos. Si necesitas varios datos, pide varias herramientas. "
    "Responde en español, de forma breve y concreta."
)


# =====================================================================================
# LAS TRES HERRAMIENTAS
# =====================================================================================

# --- 1. buscar_en_corpus: el RAG de la sesión pasada, ahora como herramienta ----------

ARCHIVO_INDICE = pathlib.Path(__file__).parent / "indice.json"
INDICE = json.loads(ARCHIVO_INDICE.read_text(encoding="utf-8"))

# Todos los vectores del índice, apilados. Ya vienen normalizados desde generar_indice.py,
# así que el producto punto ES la similitud del coseno.
MATRIZ = np.array([f["vector"] for f in INDICE["fragmentos"]], dtype=np.float32)


def buscar_en_corpus(consulta):
    """Busca en los documentos del Instituto y devuelve los 3 fragmentos más parecidos."""
    respuesta = cliente_gemini.models.embed_content(
        model=MODELO_EMBEDDINGS,
        contents=consulta,
        config=types.EmbedContentConfig(
            # RETRIEVAL_QUERY, no RETRIEVAL_DOCUMENT: esto es una pregunta, no un documento.
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=DIMENSIONES,
        ),
    )
    vector = np.array(respuesta.embeddings[0].values, dtype=np.float32)
    vector = vector / np.linalg.norm(vector)  # a 768 dimensiones hay que normalizar a mano

    similitudes = MATRIZ @ vector
    mejores = np.argsort(-similitudes)[:3]

    # Si ni el mejor fragmento se parece lo suficiente, decirlo. No es una excepción:
    # es una respuesta legítima que el agente puede leer y usar.
    if similitudes[mejores[0]] < UMBRAL_SIMILITUD:
        return "No encontré información sobre eso en los documentos del Instituto Nébula."

    partes = []
    for posicion in mejores:
        fragmento = INDICE["fragmentos"][posicion]
        partes.append(
            f"[fuente: {fragmento['fuente']} — {fragmento['titulo']} "
            f"| parecido: {similitudes[posicion]:.2f}]\n{fragmento['texto']}"
        )
    return "\n\n".join(partes)


# --- 2. calcular_costo: la tabla en código -------------------------------------------

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
        f"  Precio base            ${base:>9,.2f}",
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
            lineas.append(
                f"  {beca['nombre']} ({beca['porcentaje']:.0%})".ljust(25)
                + f"-${descuento:>9,.2f}"
            )

    subtotal = base - descuento
    impuesto = subtotal * IVA
    total = subtotal + impuesto

    lineas.append(f"  Subtotal               ${subtotal:>9,.2f}")
    lineas.append(f"  IVA ({IVA:.0%})               ${impuesto:>9,.2f}")
    lineas.append("  " + "─" * 31)
    lineas.append(f"  TOTAL A PAGAR          ${total:>9,.2f}")
    return "\n".join(lineas)


# --- 3. consultar_cupo: el servicio externo simulado ---------------------------------

# Cámbiala en vivo. En este checkpoint la dejamos en "ok"; el cp3 es el de las fallas.
ESCENARIO = "ok"  # "ok" | "timeout" | "error500"

CUPOS = {
    "NBL-101": {"disponibles": 12, "total": 30},
    "NBL-204": {"disponibles": 3, "total": 25},
    "NBL-210": {"disponibles": 0, "total": 28},
    "NBL-305": {"disponibles": 8, "total": 20},
    "NBL-402": {"disponibles": 5, "total": 15},
}


def consultar_cupo(codigo_curso):
    """Consulta los lugares disponibles. Simula un servicio externo del Instituto."""
    if ESCENARIO == "timeout":
        raise TimeoutError("El servicio de inscripciones no respondió en 5 segundos.")
    if ESCENARIO == "error500":
        raise RuntimeError("HTTP 500 del servicio de inscripciones. Intenta de nuevo.")

    codigo = codigo_curso.strip().upper()
    if codigo not in CUPOS:
        return f"No existe el curso {codigo}. Códigos válidos: {', '.join(CUPOS)}."

    cupo = CUPOS[codigo]
    if cupo["disponibles"] == 0:
        return (
            f"{codigo}: SIN CUPO. Los {cupo['total']} lugares están ocupados. "
            f"Hay lista de espera abierta."
        )
    return (
        f"{codigo}: {cupo['disponibles']} lugares disponibles de {cupo['total']}. "
        f"Inscripciones abiertas."
    )


# --- Los esquemas: lo que el modelo lee para decidir ----------------------------------

ESQUEMA_HERRAMIENTAS = [
    {
        "type": "function",
        "function": {
            "name": "buscar_en_corpus",
            "description": (
                "Busca en los documentos oficiales del Instituto Nébula: catálogo de "
                "cursos, prerrequisitos, horarios, reglas de becas, requisitos para "
                "solicitarlas, formas de pago, reembolsos y preguntas frecuentes. "
                "Úsala para cualquier pregunta de reglas o requisitos. No sirve para "
                "calcular precios ni para consultar cupos."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "consulta": {
                        "type": "string",
                        "description": (
                            "Qué quieres buscar, redactado como una frase completa. "
                            "Por ejemplo: 'requisitos para la beca de egresados'."
                        ),
                    }
                },
                "required": ["consulta"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calcular_costo",
            "description": (
                "Calcula el costo total de un curso, con el desglose línea por línea: "
                "precio base, descuento de beca, subtotal, IVA y total. Úsala siempre "
                "que te pregunten un precio o cuánto sale algo. Nunca calcules de memoria."
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
    },
    {
        "type": "function",
        "function": {
            "name": "consultar_cupo",
            "description": (
                "Consulta cuántos lugares quedan disponibles en un curso. Es el único "
                "dato que cambia día a día, así que consúltalo siempre que pregunten si "
                "hay lugar, si está lleno o si todavía se pueden inscribir."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "codigo_curso": {
                        "type": "string",
                        "description": "Código del curso, por ejemplo 'NBL-204'.",
                    }
                },
                "required": ["codigo_curso"],
            },
        },
    },
]

HERRAMIENTAS = {
    "buscar_en_corpus": buscar_en_corpus,
    "calcular_costo": calcular_costo,
    "consultar_cupo": consultar_cupo,
}


# =====================================================================================
# EL AGENTE — el mismo loop del cp1, ahora con tres herramientas
# =====================================================================================


def correr_agente(pregunta):
    """Corre el loop hasta que el modelo deje de pedir herramientas."""
    mensajes = [
        {"role": "system", "content": INSTRUCCIONES},
        {"role": "user", "content": pregunta},
    ]
    trayectoria = []  # qué herramientas pidió, en qué orden, con qué argumentos
    vuelta = 0

    while True:
        vuelta += 1
        respuesta = cliente.chat.completions.create(
            model=MODELO,
            messages=mensajes,
            tools=ESQUEMA_HERRAMIENTAS,
            temperature=0,
            seed=42,
            include_reasoning=False,
        )
        mensaje = respuesta.choices[0].message

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

        if not mensaje.tool_calls:
            return mensaje.content, trayectoria, vuelta

        for llamada in mensaje.tool_calls:
            nombre = llamada.function.name
            argumentos = json.loads(llamada.function.arguments)

            resultado = HERRAMIENTAS[nombre](**argumentos)

            trayectoria.append({"vuelta": vuelta, "herramienta": nombre, "argumentos": argumentos})
            print(f"    vuelta {vuelta}: {nombre}({json.dumps(argumentos, ensure_ascii=False)})")

            mensajes.append(
                {
                    "role": "tool",
                    "tool_call_id": llamada.id,
                    "name": nombre,
                    "content": resultado,
                }
            )


# =====================================================================================
# LAS DOS PREGUNTAS
# =====================================================================================

PREGUNTA_SIMPLE = "¿Cuánto cuesta el NBL-204?"

PREGUNTA_COMPUESTA = (
    "¿Cuánto me sale el NBL-204 en línea con la beca de egresados, "
    "qué necesito para que me la aprueben, y todavía hay lugar?"
)

for etiqueta, pregunta in [
    ("SIMPLE", PREGUNTA_SIMPLE),
    ("COMPUESTA", PREGUNTA_COMPUESTA),
]:
    print("=" * 79)
    print(f"PREGUNTA {etiqueta}")
    print("=" * 79)
    print(f"\n{pregunta}\n")
    print("  Trayectoria:")

    contenido, trayectoria, vueltas = correr_agente(pregunta)

    print(f"\n  -> {len(trayectoria)} llamada(s) a herramientas en {vueltas} vueltas al modelo")
    print(f"  -> orden: {' → '.join(paso['herramienta'] for paso in trayectoria)}")
    print(f"\nRESPUESTA:\n{contenido}\n")
