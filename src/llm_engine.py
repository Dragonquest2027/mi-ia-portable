import os
from llama_cpp import Llama

class MotorLogicoIA:
    def __init__(self, ruta_modelo="./modelos/modelo.gguf"):
        if not os.path.exists(ruta_modelo):
            raise FileNotFoundError(f"Coloca tu archivo .gguf en: {os.path.abspath(ruta_modelo)}")
        
        self.llm = Llama(
            model_path=ruta_modelo,
            n_ctx=4096,          # Ventana de contexto (memoria de trabajo)
            n_threads=os.cpu_count() or 4,
            verbose=False
        )

    def generar_respuesta(self, pregunta, contexto="", modo="consulta"):
        if modo == "informe":
            instruccion = (
                "Eres un analista experto. Redacta un INFORME TÉCNICO FORMAL basado estrictamente "
                "en el contexto proporcionado. Estructura el documento con: "
                "1. Resumen Ejecutivo, 2. Análisis Lógico de Hallazgos, 3. Conclusiones y Recomendaciones."
            )
        else:
            instruccion = (
                "Eres un asistente analítico y lógico. Analiza los datos paso a paso antes de responder. "
                "Si la respuesta está en el contexto, cítala con precisión. Si falta información, indícalo."
            )

        prompt = f"""<|im_start|>system
{instruccion}<|im_end|>
<|im_start|>user
CONTEXTO DE LA BASE DE DATOS:
{contexto if contexto else "Sin documentos cargados."}

SOLICITUD DEL USUARIO:
{pregunta}<|im_end|>
<|im_start|>assistant
"""
        salida = self.llm(
            prompt,
            max_tokens=1500,
            temperature=0.2,  # Temperatura baja = mayor precisión lógica y menos alucinaciones
            stop=["<|im_end|>"]
        )
        return salida["choices"][0]["text"].strip()
