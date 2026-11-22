import os
import lancedb
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

class BaseConocimiento:
    def __init__(self, db_path="./datos/memoria_ia"):
        os.makedirs(db_path, exist_ok=True)
        self.db = lancedb.connect(db_path)
        # Modelo ligero de embeddings (aprox. 90 MB)
        self.embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        self.table_name = "documentos"

    def indexar_pdf(self, ruta_pdf):
        reader = PdfReader(ruta_pdf)
        texto_completo = "\n".join([page.extract_text() or "" for page in reader.pages])
        
        # Dividir en bloques lógicos de 800 caracteres con solapamiento
        fragmentos = [texto_completo[i:i+800] for i in range(0, len(texto_completo), 650)]
        vectores = self.embedder.encode(fragmentos).tolist()
        
        registros = [
            {"vector": vec, "texto": frag, "fuente": os.path.basename(ruta_pdf)}
            for vec, frag in zip(vectores, fragmentos)
        ]
        
        if self.table_name in self.db.table_names():
            tabla = self.db.open_table(self.table_name)
            tabla.add(registros)
        else:
            self.db.create_table(self.table_name, data=registros)
        return len(registros)

    def buscar_contexto(self, pregunta, limite=5):
        if self.table_name not in self.db.table_names():
            return ""
        tabla = self.db.open_table(self.table_name)
        vec_pregunta = self.embedder.encode([pregunta])[0].tolist()
        resultados = tabla.search(vec_pregunta).limit(limite).to_list()
        
        contexto = "\n---\n".join([f"[Fuente: {r['fuente']}]\n{r['texto']}" for r in resultados])
        return contexto
