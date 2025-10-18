# 💜 Chatbot Educativo sobre SOP con Análisis de Imágenes

> Asistente virtual inteligente basado en guías médicas internacionales (ESHRE 2023) para educación sobre el Síndrome de Ovario Poliquístico, con capacidad de análisis educativo de imágenes médicas.

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Gemini](https://img.shields.io/badge/Google_Gemini-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://ai.google.dev/)
[![Langchain](https://img.shields.io/badge/Langchain-121212?style=for-the-badge)](https://www.langchain.com/)

---

## 🌟 Características Principales

### 💬 **Chat Inteligente con RAG**
- 🧠 Búsqueda semántica avanzada usando **Hugging Face embeddings**
- 📚 Basado 100% en la **Guía Internacional ESHRE 2023**
- 🎯 Respuestas precisas y contextualizadas
- 💜 Tono empático y cercano, no robótico

### 📸 **Análisis Educativo de Imágenes**
- 🧪 **Resultados de laboratorio** (hormonas, glucosa, lípidos)
- 📅 **Gráficas de ciclos menstruales**
- 🔬 **Ecografías** (explicación general, no interpretación)
- 💡 **Preguntas sugeridas** personalizadas para el médico

### 🔒 **Seguridad Médica**
- ❌ **Nunca diagnostica** ni prescribe tratamientos
- ✅ Solo información verificable de fuentes oficiales
- ⚠️ Múltiples capas de validación y disclaimers
- 🏥 Siempre redirige a profesionales de salud

### 🗺️ **Búsqueda de Profesionales**
- Integración con Google Maps
- Búsqueda de ginecólogos, endocrinólogos, psicólogos y nutriólogos
- Filtrado por ciudad/municipio

---

## 🛠️ Stack Tecnológico

| Componente | Tecnología |
|-----------|-----------|
| **Frontend** | Streamlit |
| **LLM** | Google Gemini 2.0 Flash |
| **Embeddings** | Hugging Face (sentence-transformers) |
| **Vector Store** | ChromaDB |
| **RAG Framework** | Langchain |
| **Vision API** | Gemini Vision (análisis de imágenes) |
| **PDF Processing** | PyPDF |

---

## 📦 Instalación Local

### Prerrequisitos

- Python 3.10 o superior
- API Key de Google Gemini ([obtener aquí](https://aistudio.google.com/app/apikey))
- PDF de la Guía ESHRE 2023

### Pasos de Instalación
```bash
# 1. Clonar el repositorio
git clone https://github.com/tu-usuario/chatbot-sop.git
cd chatbot-sop

# 2. Crear entorno virtual
python -m venv .venv

# 3. Activar entorno virtual
# Windows:
.\.venv\Scripts\Activate.ps1
# Linux/Mac:
source .venv/bin/activate

# 4. Instalar dependencias
pip install -r requirements.txt

# 5. Configurar variables de entorno
# Crear archivo .env con:
echo "GOOGLE_API_KEY=tu_api_key_aqui" > .env

# 6. Crear embeddings (SOLO PRIMERA VEZ - tarda 5-10 min)
python create_embeddings.py

# 7. Ejecutar la aplicación
streamlit run bot_sop.py
```

La aplicación se abrirá en `http://localhost:8501`

---

## 🚀 Deploy en Streamlit Cloud

### Configuración

1. **Fork** este repositorio
2. Ve a [Streamlit Cloud](https://share.streamlit.io/)
3. Click en **"New app"**
4. Conecta tu repositorio
5. Configura:
   - **Main file:** `bot_sop.py`
   - **Python version:** 3.10
6. En **Settings → Secrets**, agrega:
```toml
   GOOGLE_API_KEY = "tu_api_key_aqui"
```
7. Click **Deploy**

⏱️ El deploy tarda ~5-10 minutos la primera vez.

---

## 📂 Estructura del Proyecto
```
chatbot-sop/
├── bot_sop.py                 # Aplicación principal
├── create_embeddings.py       # Script para generar vectorstore
├── requirements.txt           # Dependencias Python
├── .env                       # Variables de entorno (NO subir a Git)
├── .gitignore                # Archivos ignorados por Git
├── guia_sop.pdf              # Guía médica ESHRE 2023
├── logo.png                  # Logo (opcional)
├── chroma_db_sop/            # Base de datos vectorial (8.57 MB)
│   └── ...
└── README.md                 # Este archivo
```

---

## 🎯 Casos de Uso

### Para Pacientes
- 📖 Entender qué es el SOP y sus síntomas
- 💊 Conocer opciones de tratamiento disponibles
- 🤰 Información sobre fertilidad y embarazo
- 🥗 Guía de alimentación y ejercicio
- 📸 Comprender sus estudios médicos antes de la consulta

### Para Educadores
- 📚 Material educativo basado en evidencia
- 🏥 Herramienta de pre-consulta para pacientes
- 💡 Reducir carga de preguntas frecuentes

### Para Desarrolladores
- 🔬 Ejemplo de RAG médico con Langchain
- 🖼️ Integración de Vision API para análisis de imágenes
- 🎨 UI/UX de chatbot médico responsable

---

## 🧪 Ejemplo de Uso
```python
# Pregunta del usuario
"¿Qué es el SOP y cuáles son sus síntomas?"

# El sistema:
1. 🔍 Busca semánticamente en el vectorstore
2. 📚 Extrae contexto relevante de la guía ESHRE 2023
3. 🤖 Genera respuesta con Gemini 2.0 Flash
4. 💜 Responde con tono empático y validador
5. ✅ Cita la fuente (guía internacional)
```

### Análisis de Imágenes
```python
# Usuario sube: Gráfica de ciclos menstruales
# Sistema analiza:
✅ Identifica tipo de app/registro
✅ Observa patrones (duración, síntomas)
✅ Explica criterios diagnósticos ESHRE 2023
✅ Sugiere qué agregar al registro
✅ Genera 5 preguntas específicas para el médico
```

---

## ⚙️ Configuración Avanzada

### Ajustar Temperatura del Modelo

En `bot_sop.py` línea ~40:
```python
generation_config={
    "temperature": 0.3,  # 0.0-1.0 (más bajo = más conservador)
    "top_p": 0.8,
    "top_k": 40,
    "max_output_tokens": 2048,
}
```

### Cambiar Modelo de Embeddings

En `create_embeddings.py` línea ~35:
```python
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    # Otros modelos:
    # "sentence-transformers/all-MiniLM-L6-v2"
    # "intfloat/multilingual-e5-large"
)
```

### Ajustar Chunks del PDF

En `create_embeddings.py` línea ~25:
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,      # Aumentar para más contexto
    chunk_overlap=200,    # Solapamiento entre chunks
)
```

---

## 🔐 Seguridad y Privacidad

- 🔒 **No almacena datos personales** del usuario
- 🚫 **No guarda historial** de conversaciones en servidores
- 🔑 **API Keys encriptadas** en Streamlit Secrets
- ⚠️ **Disclaimers claros** sobre uso educativo
- 🏥 **Siempre redirige** a profesionales de salud

---

## 📊 Limitaciones Conocidas

- ⏱️ **Límites de API**: Gemini free tier tiene cuotas diarias
- 🌍 **Idioma**: Optimizado solo para español
- 📱 **Imágenes**: Análisis educativo, no diagnóstico
- 🔬 **Alcance**: Solo información sobre SOP (no otras condiciones)
- 📄 **Fuente**: Basado en guía ESHRE 2023 (puede haber actualizaciones posteriores)

---

## 🗺️ Roadmap

- [ ] 🌐 Soporte multiidioma (inglés, portugués)
- [ ] 📊 Dashboard de analytics para educadores
- [ ] 🔔 Sistema de recordatorios para seguimiento
- [ ] 💬 Integración con WhatsApp/Telegram
- [ ] 🎤 Soporte de entrada por voz
- [ ] 📱 App móvil nativa

---

## 🤝 Contribuciones

¡Las contribuciones son bienvenidas! Por favor:

1. Fork el proyecto
2. Crea tu feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push al branch (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

### Áreas donde puedes contribuir
- 🐛 Reportar bugs
- 💡 Sugerir nuevas características
- 📝 Mejorar documentación
- 🌍 Traducir a otros idiomas
- 🎨 Mejorar UI/UX

---

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

---

## 👥 Autores

- **Tu Nombre** - *Desarrollo inicial* - [@Martin-carrizalez](https://github.com/Martin-carrizalez)

---

## 🙏 Agradecimientos

- 📚 [Guía Internacional ESHRE 2023](https://www.eshre.eu/) por la información médica
- 🤖 [Google Gemini](https://ai.google.dev/) por el modelo de lenguaje
- 🔗 [Langchain](https://www.langchain.com/) por el framework RAG
- 🎨 [Streamlit](https://streamlit.io/) por el framework de UI
- 🧠 [Hugging Face](https://huggingface.co/) por los embeddings

---

## 📞 Contacto

- **Email**: martin.carrizalez0823@alumnos.udg.mx
- **LinkedIn**: [MARTIN ANGEL CARRIZALEZ PINA](https://www.linkedin.com/in/martin-angel-carrizalez-pina-b55475371/)

---

## ⚠️ Disclaimer Médico

**IMPORTANTE:** Este chatbot es una herramienta **educativa** únicamente. No reemplaza la consulta médica profesional. Para diagnóstico, tratamiento o cualquier decisión relacionada con tu salud, siempre consulta con un profesional de salud calificado.

---

<div align="center">

**Hecho con 💜 para la comunidad SOP**

⭐ Si este proyecto te fue útil, considera darle una estrella

</div>
```

---

## 📝 ARCHIVO `LICENSE` (opcional)

Si quieres agregar licencia MIT, crea `LICENSE`:
```
MIT License

Copyright (c) 2025 QFB Martin Angel Carrizalez Piña

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
