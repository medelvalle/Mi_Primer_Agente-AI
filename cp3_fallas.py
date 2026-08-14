"""
CHECKPOINT 3 — Qué pasa cuando una herramienta falla.

Hasta aquí todo salió bien. En producción no sale bien.

Cuatro sub-escenarios. Cambia la constante SUB_ESCENARIO de abajo y vuelve a correr:

  "a_sin_atrapar"  La herramienta lanza una excepción y nadie la atrapa.
                   El traceback mata el proceso. El agente ni se entera.

  "b_atrapada"     La misma falla, pero el despachador la convierte en TEXTO.
                   El agente la lee, la entiende y responde con lo que sí pudo.

  "c_sin_limite"   La herramienta falla SIEMPRE y el error invita a reintentar.
                   El agente insiste, y insiste, y no para. Sin límite, esto
                   es una factura abierta. Lo cortamos a mano para verlo.

  "d_con_limite"   La misma falla, con MAX_VUELTAS. Cuando se acaban los
                   intentos, el agente se abstiene y lo dice honestamente.

    python cp3_fallas.py
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

# =====================================================================================
# LA PERILLA — cámbiala y vuelve a correr
# =====================================================================================

SUB_ESCENARIO = "a_sin_atrapar"  # "a_sin_atrapar" | "b_atrapada" | "c_sin_limite" | "d_con_limite"

# =====================================================================================

MODELO = "openai/gpt-oss-120b"
MODELO_EMBEDDINGS = "gemini-embedding-001"
DIMENSIONES = 768
# Si el mejor fragmento no llega a este parecido, decimos que no sabemos.
# El número está medido, no adivinado: con 10 preguntas dentro del corpus y 8 fuera,
# las de dentro puntuaron entre 0.718 y 0.813, y las claramente ajenas por debajo de
# 0.66. En medio quedan las preguntas que SUENAN a Instituto pero no están escritas
# en ningún documento ("¿tienen convenio con la UNAM?" da 0.714). Ningún umbral las
# separa bien; preferimos dejarlas pasar y que el modelo, que ve la fuente y el score
# de cada fragmento, diga que eso no está documentado.
UMBRAL_SIMILITUD = 0.68

# Precio público de openai/gpt-oss-120b en Groq, por millón de tokens.
PRECIO_ENTRADA = 0.15
PRECIO_SALIDA = 0.60

# Corte de emergencia para el escenario "c": si no lo ponemos, no para nunca.
CORTE_DE_EMERGENCIA = 10

# El límite de verdad, el del escenario "d".
MAX_VUELTAS = 4

cliente = Groq(api_key=os.environ["GROQ_API_KEY"])
cliente_gemini = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

INSTRUCCIONES_BASE = (
    "Eres el asistente del Instituto Nébula. Respondes preguntas sobre cursos, precios, "
    "becas y disponibilidad. Usa siempre las herramientas: nunca inventes precios, "
    "requisitos ni cupos. Si algún dato no lo logras obtener, entrega los que sí "
    "conseguiste y di claramente cuál te faltó. "
    "Responde en español, de forma breve y concreta."
)

# Esta frase es la que convierte el escenario "b" en el "c". No es un detalle:
# es UNA LÍNEA de prompt, escrita con toda la buena intención, la que hace que el
# agente insista para siempre cuando el servicio nunca se recupera.
INSISTIR = " Si una herramienta falla, vuelve a intentarlo hasta obtener el dato."

if SUB_ESCENARIO in ("c_sin_limite", "d_con_limite"):
    INSTRUCCIONES = INSTRUCCIONES_BASE + INSISTIR
else:
    INSTRUCCIONES = INSTRUCCIONES_BASE

PREGUNTA = "¿Todavía hay lugar en el NBL-204 y cuánto me sale en línea?"


# =====================================================================================
# LAS TRES HERRAMIENTAS (idénticas al cp2)
# =====================================================================================

ARCHIVO_INDICE = pathlib.Path(__file__).parent / "indice.json"
INDICE = json.loads(ARCHIVO_INDICE.read_text(encoding="utf-8"))
MATRIZ = np.array([f["vector"] for f in INDICE["fragmentos"]], dtype=np.float32)


def buscar_en_corpus(consulta):
    """Busca en los documentos del Instituto y devuelve los 3 fragmentos más parecidos."""
    respuesta = cliente_gemini.models.embed_content(
        model=MODELO_EMBEDDINGS,
        contents=consulta,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=DIMENSIONES,
        ),
    )
    vector = np.array(respuesta.embeddings[0].values, dtype=np.float32)
    vector = vector / np.linalg.norm(vector)

    similitudes = MATRIZ @ vector
    mejores = np.argsort(-similitudes)[:3]

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


CUPOS = {
    "NBL-101": {"disponibles": 12, "total": 30},
    "NBL-204": {"disponibles": 3, "total": 25},
    "NBL-210": {"disponibles": 0, "total": 28},
    "NBL-305": {"disponibles": 8, "total": 20},
    "NBL-402": {"disponibles": 5, "total": 15},
}

# Aquí es donde el sub-escenario se convierte en una falla concreta.
# No hay azar: la misma constante produce siempre la misma excepción.
if SUB_ESCENARIO in ("a_sin_atrapar", "b_atrapada"):
    ESCENARIO = "timeout"
else:
    ESCENARIO = "error500"


def consultar_cupo(codigo_curso):
    """Consulta los lugares disponibles. Simula un servicio externo del Instituto."""
    if ESCENARIO == "timeout":
        # Lanzamos de inmediato: simulamos que el tiempo límite YA se cumplió.
        # Esperar 5 segundos de verdad no enseñaría nada más y la clase esperaría.
        raise TimeoutError("El servicio de inscripciones no respondió (timeout de 5 s).")
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
# EL DESPACHADOR — aquí está toda la diferencia entre "a" y el resto
# =====================================================================================


def despachar(nombre, argumentos):
    """Ejecuta una herramienta y devuelve su resultado como texto."""
    if SUB_ESCENARIO == "a_sin_atrapar":
        # Sin red de seguridad. Si la herramienta truena, truena el programa entero.
        return HERRAMIENTAS[nombre](**argumentos)

    # La versión buena: la excepción se convierte en texto y sigue siendo un
    # resultado de herramienta como cualquier otro. El agente la LEE.
    try:
        return HERRAMIENTAS[nombre](**argumentos)
    except Exception as error:
        return f"ERROR al ejecutar {nombre}: {type(error).__name__}: {error}"


# =====================================================================================
# EL AGENTE
# =====================================================================================

# Cada sub-escenario decide cuántas vueltas se permiten y por qué.
if SUB_ESCENARIO == "c_sin_limite":
    limite = CORTE_DE_EMERGENCIA
elif SUB_ESCENARIO == "d_con_limite":
    limite = MAX_VUELTAS
else:
    limite = CORTE_DE_EMERGENCIA  # red de seguridad, no debería alcanzarse

print("=" * 79)
print(f"SUB-ESCENARIO: {SUB_ESCENARIO}   (consultar_cupo -> {ESCENARIO})")
print("=" * 79)
print(f"\nPregunta: {PREGUNTA}\n")

mensajes = [
    {"role": "system", "content": INSTRUCCIONES},
    {"role": "user", "content": PREGUNTA},
]

tokens_entrada = 0
tokens_salida = 0
vuelta = 0
se_acabaron_las_vueltas = False
contenido_final = None

while True:
    vuelta += 1

    if vuelta > limite:
        se_acabaron_las_vueltas = True
        break

    respuesta = cliente.chat.completions.create(
        model=MODELO,
        messages=mensajes,
        tools=ESQUEMA_HERRAMIENTAS,
        temperature=0,
        seed=42,
        include_reasoning=False,
    )
    mensaje = respuesta.choices[0].message

    tokens_entrada += respuesta.usage.prompt_tokens
    tokens_salida += respuesta.usage.completion_tokens
    costo = tokens_entrada / 1e6 * PRECIO_ENTRADA + tokens_salida / 1e6 * PRECIO_SALIDA

    print(
        f"### VUELTA {vuelta:>2}  |  {len(mensajes):>2} mensajes enviados  |  "
        f"{tokens_entrada:>6,} tokens de entrada acumulados  |  ${costo:.5f} gastados"
    )

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
        contenido_final = mensaje.content
        break

    for llamada in mensaje.tool_calls:
        nombre = llamada.function.name
        argumentos = json.loads(llamada.function.arguments)

        resultado = despachar(nombre, argumentos)

        primera_linea = resultado.splitlines()[0] if resultado else ""
        print(f"       {nombre}({json.dumps(argumentos, ensure_ascii=False)})")
        print(f"       -> {primera_linea}")

        mensajes.append(
            {
                "role": "tool",
                "tool_call_id": llamada.id,
                "name": nombre,
                "content": resultado,
            }
        )


# =====================================================================================
# EL CIERRE — qué hacemos cuando se acabaron las vueltas
# =====================================================================================

costo = tokens_entrada / 1e6 * PRECIO_ENTRADA + tokens_salida / 1e6 * PRECIO_SALIDA

if se_acabaron_las_vueltas and SUB_ESCENARIO == "c_sin_limite":
    print()
    print("!" * 79)
    print(f"CORTE DE EMERGENCIA a las {limite} vueltas. El agente NO se iba a detener solo.")
    print(f"Gastado hasta aquí: {tokens_entrada:,} tokens de entrada + "
          f"{tokens_salida:,} de salida = ${costo:.5f}")
    print(f"Proyección a 1,000 preguntas iguales: ${costo * 1000:.2f}")
    print("Y esto es una sola pregunta, de un solo usuario, con un servicio caído.")
    print("!" * 79)

elif se_acabaron_las_vueltas and SUB_ESCENARIO == "d_con_limite":
    print()
    print(f"--- Se agotaron las {MAX_VUELTAS} vueltas. Pedimos una respuesta honesta. ---\n")

    # Última llamada SIN herramientas: ya no puede pedir nada más, solo redactar.
    mensajes.append(
        {
            "role": "user",
            "content": (
                "Ya no puedes usar más herramientas. Responde con la información que "
                "SÍ lograste obtener y di claramente y sin rodeos qué dato no pudiste "
                "conseguir y por qué. No inventes el dato que falta."
            ),
        }
    )
    respuesta = cliente.chat.completions.create(
        model=MODELO,
        messages=mensajes,
        temperature=0,
        seed=42,
        include_reasoning=False,
    )
    tokens_entrada += respuesta.usage.prompt_tokens
    tokens_salida += respuesta.usage.completion_tokens
    costo = tokens_entrada / 1e6 * PRECIO_ENTRADA + tokens_salida / 1e6 * PRECIO_SALIDA
    contenido_final = respuesta.choices[0].message.content

elif se_acabaron_las_vueltas:
    # Red de seguridad: en los escenarios "a" y "b" no deberíamos llegar aquí.
    # Si llegamos, es que el agente se cicló y conviene verlo, no quedarnos callados.
    print()
    print(f"--- Tocó el corte de emergencia de {limite} vueltas sin dar una respuesta. ---")

if contenido_final:
    print("RESPUESTA FINAL:\n")
    print(contenido_final)

print(f"\n[{vuelta - 1 if se_acabaron_las_vueltas else vuelta} vueltas | "
      f"{tokens_entrada:,} tokens entrada | {tokens_salida:,} tokens salida | ${costo:.5f}]")
