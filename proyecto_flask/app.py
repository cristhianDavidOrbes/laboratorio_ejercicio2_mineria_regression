import os
import re
import unicodedata

import joblib
import pandas as pd
from flask import Flask, render_template, request


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "modelo")


EJERCICIOS = {
    "dolar": {
        "titulo": "Prediccion del precio del dolar",
        "archivo_modelo": "paquete_modelo_dolar.joblib",
        "campos": [
            {"nombre": "dia", "label": "Dia"},
            {"nombre": "inflacion", "label": "Inflacion"},
            {"nombre": "tasa_interes", "label": "Tasa de interes"},
        ],
        "objetivo_esperado": "precio_dolar",
    },
    "glucosa": {
        "titulo": "Prediccion del nivel de glucosa",
        "archivo_modelo": "paquete_modelo_glucosa.joblib",
        "campos": [
            {"nombre": "edad", "label": "Edad"},
            {"nombre": "imc", "label": "IMC"},
            {"nombre": "actividad_fisica", "label": "Actividad fisica"},
        ],
        "objetivo_esperado": "nivel_glucosa",
    },
    "energia": {
        "titulo": "Prediccion del consumo de energia electrica",
        "archivo_modelo": "paquete_modelo_energia.joblib",
        "campos": [
            {"nombre": "temperatura", "label": "Temperatura"},
            {"nombre": "hora", "label": "Hora"},
            {"nombre": "dia_semana", "label": "Dia de semana"},
        ],
        "objetivo_esperado": "consumo_energia",
    },
}


def normalizar_nombre(texto):
    normalizado = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("ascii")
    normalizado = normalizado.strip().lower().replace(" ", "_")
    normalizado = re.sub(r"[^a-z0-9_]", "_", normalizado)
    normalizado = re.sub(r"_+", "_", normalizado).strip("_")
    return normalizado


def cargar_paquete_modelo(nombre_archivo):
    ruta_modelo = os.path.join(MODEL_DIR, nombre_archivo)
    if not os.path.exists(ruta_modelo):
        return None, f"No se encontro el archivo del modelo: {ruta_modelo}"

    try:
        paquete = joblib.load(ruta_modelo)
    except Exception as exc:
        return None, f"No se pudo cargar el modelo: {exc}"

    if not isinstance(paquete, dict):
        return None, "El archivo de modelo no contiene un paquete valido."

    claves_requeridas = {"modelo", "variables", "variable_objetivo", "nombre_modelo"}
    if not claves_requeridas.issubset(paquete.keys()):
        return None, "El paquete del modelo esta incompleto o corrupto."

    return paquete, None


def construir_fila_entrada(variables_modelo, valores_formulario):
    valores_normalizados = {normalizar_nombre(k): v for k, v in valores_formulario.items()}
    fila = {}
    faltantes = []

    for var_modelo in variables_modelo:
        clave = normalizar_nombre(str(var_modelo))
        if clave not in valores_normalizados:
            faltantes.append(str(var_modelo))
            continue
        fila[var_modelo] = valores_normalizados[clave]

    if faltantes:
        return None, f"No se pudieron mapear variables del modelo: {', '.join(faltantes)}"
    return fila, None


def crear_contexto_base(ejercicio_seleccionado="dolar", datos_formulario=None):
    if datos_formulario is None:
        datos_formulario = {}
    return {
        "ejercicios": EJERCICIOS,
        "ejercicio_seleccionado": ejercicio_seleccionado,
        "datos_formulario": datos_formulario,
        "error": None,
        "resultado": None,
    }


app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    contexto = crear_contexto_base()
    return render_template("index.html", **contexto)


@app.route("/predecir", methods=["POST"])
def predecir():
    ejercicio = request.form.get("ejercicio", "").strip()
    if ejercicio not in EJERCICIOS:
        contexto = crear_contexto_base()
        contexto["error"] = "Debes seleccionar un ejercicio valido."
        return render_template("index.html", **contexto)

    config = EJERCICIOS[ejercicio]
    datos_formulario = {}

    for campo in config["campos"]:
        nombre = campo["nombre"]
        valor_texto = request.form.get(nombre, "").strip()
        datos_formulario[nombre] = valor_texto

    contexto = crear_contexto_base(ejercicio, datos_formulario)

    valores_numericos = {}
    for campo in config["campos"]:
        nombre = campo["nombre"]
        valor_texto = datos_formulario.get(nombre, "")
        if valor_texto == "":
            contexto["error"] = f"El campo '{campo['label']}' es obligatorio."
            return render_template("index.html", **contexto)
        try:
            valores_numericos[nombre] = float(valor_texto)
        except ValueError:
            contexto["error"] = f"El campo '{campo['label']}' debe ser numerico."
            return render_template("index.html", **contexto)

    paquete, error_carga = cargar_paquete_modelo(config["archivo_modelo"])
    if error_carga:
        contexto["error"] = error_carga
        return render_template("index.html", **contexto)

    fila, error_mapeo = construir_fila_entrada(paquete["variables"], valores_numericos)
    if error_mapeo:
        contexto["error"] = error_mapeo
        return render_template("index.html", **contexto)

    try:
        entrada_df = pd.DataFrame([fila], columns=paquete["variables"])
        prediccion = paquete["modelo"].predict(entrada_df)
        valor_predicho = float(prediccion[0])
    except Exception as exc:
        contexto["error"] = f"No se pudo realizar la prediccion: {exc}"
        return render_template("index.html", **contexto)

    objetivo = str(paquete.get("variable_objetivo") or config["objetivo_esperado"])
    nombre_modelo = str(paquete.get("nombre_modelo") or "Modelo")

    contexto["resultado"] = {
        "ejercicio": config["titulo"],
        "nombre_modelo": nombre_modelo,
        "objetivo": objetivo,
        "valor": f"{valor_predicho:.2f}",
    }
    return render_template("index.html", **contexto)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
