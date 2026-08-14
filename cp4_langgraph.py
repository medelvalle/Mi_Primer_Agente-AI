"""
CHECKPOINT 4 — El mismo agente, ahora como grafo.

No cambia el modelo, ni las herramientas, ni el resultado. Cambia quién lleva el loop:
en vez de un `while` que escribimos nosotros, es un grafo que LangGraph recorre.

    while + if                    ->   nodos + arista condicional
    la variable `mensajes`        ->   el estado tipado
    `if not mensaje.tool_calls`   ->   la función que decide a qué nodo ir
    `if vuelta > MAX_VUELTAS`     ->   un contador que vive en el estado

Seguimos usando el cliente de Groq directo y despachamos las herramientas a mano.
Hay un ToolNode que hace el despacho por ti, pero justo eso es lo que queremos ver.

    python cp4_langgraph.py
"""

import json
import operator
import os
import pathlib
from typing import Annotated, TypedDict

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from groq import Groq
from langgraph.graph import END, START, StateGraph

load_dotenv()

MODELO = "openai/gpt-oss-120b"
MODELO_EMBEDDINGS = "gemini-embedding-001"
DIMENSIONES = 768
UMBRAL_SIMILITUD = 0.62
MAX_VUELTAS = 4

cliente = Groq(api_key=os.environ["GROQ_API_KEY"])
cliente_gemini = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

INSTRUCCIONES = (
    "Eres el asistente del Instituto Nébula. Respondes preguntas sobre cursos, precios, "
    "becas y disponibilidad. Usa siempre las herramientas: nunca inventes precios, "
    "requisitos ni cupos. Si necesitas varios datos, pide varias herramientas. "
    "Responde en español, de forma breve y concreta."
)

PREGUNTA = (
    "¿Cuánto me sale el NBL-204 en línea con la beca de egresados, "
    "qué necesito para que me la aprueben, y todavía hay lugar?"
)


# =====================================================================================
# LAS TRES HERRAMIENTAS (idénticas al cp2 y al cp3)
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
# EL ESTADO — antes era la variable `mensajes`; ahora tiene nombre y tipo
# =====================================================================================


class Estado(TypedDict):
    # Annotated[..., operator.add] es el "reductor": cuando un nodo devuelve mensajes,
    # LangGraph los CONCATENA a los que ya había en vez de reemplazarlos.
    mensajes: Annotated[list, operator.add]
    trayectoria: Annotated[list, operator.add]
    # Sin reductor, gana el último valor escrito. Es lo que queremos para un contador.
    vueltas: int


# =====================================================================================
# LOS NODOS — dos funciones que reciben el estado y devuelven qué cambió
# =====================================================================================


def nodo_modelo(estado: Estado):
    """Le pregunta al modelo qué sigue. Es el 'cerebro' del grafo."""
    vuelta = estado["vueltas"] + 1
    print(f"\n[nodo_modelo] vuelta {vuelta} — {len(estado['mensajes'])} mensajes en el estado")

    respuesta = cliente.chat.completions.create(
        model=MODELO,
        messages=estado["mensajes"],
        tools=ESQUEMA_HERRAMIENTAS,
        temperature=0,
        seed=42,
        include_reasoning=False,
    )
    mensaje = respuesta.choices[0].message

    nuevo = {
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
    return {"mensajes": [nuevo], "vueltas": vuelta}


def nodo_herramientas(estado: Estado):
    """Ejecuta lo que el modelo pidió. El despacho está aquí, a la vista."""
    ultimo = estado["mensajes"][-1]
    mensajes_nuevos = []
    pasos = []

    for llamada in ultimo["tool_calls"]:
        nombre = llamada["function"]["name"]
        argumentos = json.loads(llamada["function"]["arguments"])
        print(f"[nodo_herramientas] {nombre}({json.dumps(argumentos, ensure_ascii=False)})")

        # La excepción se vuelve texto, como aprendimos en el cp3.
        try:
            resultado = HERRAMIENTAS[nombre](**argumentos)
        except Exception as error:
            resultado = f"ERROR al ejecutar {nombre}: {type(error).__name__}: {error}"

        mensajes_nuevos.append(
            {
                "role": "tool",
                "tool_call_id": llamada["id"],
                "name": nombre,
                "content": resultado,
            }
        )
        pasos.append(
            {"vuelta": estado["vueltas"], "herramienta": nombre, "argumentos": argumentos}
        )

    return {"mensajes": mensajes_nuevos, "trayectoria": pasos}


def nodo_sin_vueltas(estado: Estado):
    """Se acabaron los intentos. Una última llamada sin herramientas, para abstenerse bien."""
    print(f"\n[nodo_sin_vueltas] se agotaron las {MAX_VUELTAS} vueltas")

    aviso = {
        "role": "user",
        "content": (
            "Ya no puedes usar más herramientas. Responde con la información que SÍ "
            "lograste obtener y di claramente qué dato no pudiste conseguir y por qué. "
            "No inventes el dato que falta."
        ),
    }
    respuesta = cliente.chat.completions.create(
        model=MODELO,
        messages=estado["mensajes"] + [aviso],
        temperature=0,
        seed=42,
        include_reasoning=False,
    )
    final = {"role": "assistant", "content": respuesta.choices[0].message.content}
    return {"mensajes": [aviso, final]}


# =====================================================================================
# LA ARISTA CONDICIONAL — el criterio de paro, que antes era un `if`
# =====================================================================================


def decidir_siguiente(estado: Estado):
    """Mira el último mensaje y devuelve el NOMBRE del nodo al que hay que ir."""
    ultimo = estado["mensajes"][-1]

    if not ultimo.get("tool_calls"):
        return "fin"  # el modelo ya no pide nada: terminó
    if estado["vueltas"] >= MAX_VUELTAS:
        return "sin_vueltas"  # pide más, pero se le acabaron los intentos
    return "herramientas"


# =====================================================================================
# EL GRAFO
# =====================================================================================

constructor = StateGraph(Estado)
constructor.add_node("modelo", nodo_modelo)
constructor.add_node("herramientas", nodo_herramientas)
constructor.add_node("sin_vueltas", nodo_sin_vueltas)

constructor.add_edge(START, "modelo")
constructor.add_conditional_edges(
    "modelo",
    decidir_siguiente,
    {
        "herramientas": "herramientas",
        "sin_vueltas": "sin_vueltas",
        "fin": END,
    },
)
constructor.add_edge("herramientas", "modelo")  # <- el ciclo. Esto era el `while`.
constructor.add_edge("sin_vueltas", END)

grafo = constructor.compile()


# =====================================================================================
# CORRER EL GRAFO
# =====================================================================================

print("=" * 79)
print("EL GRAFO")
print("=" * 79)
print("""
          START
            |
            v
       +---------+   pide herramientas y le quedan vueltas
       | modelo  | ------------------------------------+
       +---------+                                     |
         |     |  ya no pide nada                      v
         |     +--------------------> END       +--------------+
         |                                      | herramientas |
         |  pide herramientas pero              +--------------+
         |  ya no le quedan vueltas                    |
         v                                             |
   +-------------+                                     |
   | sin_vueltas | --> END        el ciclo: <----------+
   +-------------+                (esto era el `while`)
""")
# Si quieres el diagrama que LangGraph genera solo:
#     print(grafo.get_graph().draw_mermaid())
# Devuelve texto en formato Mermaid, que se pega en https://mermaid.live

print("=" * 79)
print("CORRIENDO")
print("=" * 79)
print(f"\nPregunta: {PREGUNTA}")

estado_final = grafo.invoke(
    {
        "mensajes": [
            {"role": "system", "content": INSTRUCCIONES},
            {"role": "user", "content": PREGUNTA},
        ],
        "trayectoria": [],
        "vueltas": 0,
    }
)

print("\n" + "=" * 79)
print("TRAYECTORIA")
print("=" * 79)
for paso in estado_final["trayectoria"]:
    print(
        f"  vuelta {paso['vuelta']}: {paso['herramienta']}"
        f"({json.dumps(paso['argumentos'], ensure_ascii=False)})"
    )

print(f"\n  {len(estado_final['trayectoria'])} herramientas en "
      f"{estado_final['vueltas']} vueltas al modelo")

print("\nRESPUESTA FINAL:\n")
print(estado_final["mensajes"][-1]["content"])
