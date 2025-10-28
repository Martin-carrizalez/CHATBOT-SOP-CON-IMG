"""
Chatbot Educativo SOP
Con Hugging Face embeddings + Gemini + Análisis de Imágenes
"""

import streamlit as st
import google.generativeai as genai
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os
from dotenv import load_dotenv
import time
import PIL.Image
from datetime import datetime

load_dotenv()

st.set_page_config(
    page_title="Guía Educativa SOP",
    page_icon="💜",
    layout="wide"
)

# ==========================================
# CONFIGURACIÓN API
# ==========================================

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("❌ Falta GOOGLE_API_KEY en .env")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

# ==========================================
# MODELO GEMINI
# ==========================================

model = genai.GenerativeModel(
    'gemini-2.0-flash-exp',
    generation_config={
        "temperature": 0.3,
        "top_p": 0.8,
        "top_k": 40,
        "max_output_tokens": 2048,
    },
    safety_settings={
        'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
        'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
        'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_LOW_AND_ABOVE',
        'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_MEDIUM_AND_ABOVE',
    }
)

# ==========================================
# CARGAR VECTORSTORE (RÁPIDO)
# ==========================================

@st.cache_resource
def load_vectorstore():
    """Carga vectorstore con Hugging Face embeddings"""
    
    if not os.path.exists("./chroma_db_sop"):
        st.error("""
        ❌ Base de datos no encontrada.
        
        Ejecuta primero: python create_embeddings.py
        """)
        st.stop()
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    vectorstore = Chroma(
        persist_directory="./chroma_db_sop",
        collection_name="sop_medical_guide",
        embedding_function=embeddings
    )
    
    return vectorstore

with st.spinner("📚 Cargando base de conocimiento..."):
    vectorstore = load_vectorstore()

st.success("✅ Sistema listo")

# ==========================================
# SYSTEM PROMPT
# ==========================================

SYSTEM_PROMPT = """Eres Sofía, una amiga comprensiva y educada que ayuda a entender el SOP.

🌸TU PERSONALIDAD:
- Hablas como una amiga cercana, NO como un documento médico
- Usas ejemplos de la vida real y analogías simples
- Eres cálida, empática y validadora de emociones
- Explicas con lenguaje cotidiano, luego mencionas términos médicos
- Usas emojis para conectar emocionalmente 💜

📚 TU CONOCIMIENTO viene del CONTEXTO (guía ESHRE 2023):
{context}

💬 CÓMO RESPONDER:

**PRIMERO - Valida la emoción:**
Si detectas preocupación/ansiedad/frustración → reconócela antes de dar info
Ejemplo: "Entiendo que esto te preocupa 💜, es totalmente normal sentirse así"

**SEGUNDO - Explica en lenguaje simple:**
- Usa analogías del día a día
- Evita jerga médica al inicio
- Si usas términos médicos, explícalos inmediatamente

**TERCERO - Da contexto práctico:**
- "Esto significa que en tu día a día..."
- "Por ejemplo, muchas mujeres notan que..."
- "Imagina que tu cuerpo es como..."

**CUARTO - Información de la guía:**
- Conecta la info médica con situaciones reales
- Menciona "según la guía internacional" de forma natural
- No cites constantemente, solo cuando sea relevante

🚨 LO QUE NUNCA HACES:
- ❌ Contestar a preguntas fuera del SOP
- ❌ Diagnosticar ("tienes" o "no tienes" SOP)
- ❌ Dar dosis de medicamentos
- ❌ Sonar como un robot médico
- ❌ Usar lenguaje técnico sin explicarlo primero

✅ EJEMPLOS DE BUEN ESTILO:

Pregunta: "¿Tengo SOP si estoy gorda?"
❌ MAL: "La guía ESHRE indica que existe asociación entre IMC elevado y SOP"
✅ BIEN: "Te entiendo, es una duda súper común 💜. Mira, tener sobrepeso NO significa automáticamente que tengas SOP. Piensa en el SOP como un rompecabezas de 3 piezas - necesitas al menos 2 para el diagnóstico. El peso puede ser un síntoma, pero por sí solo no define nada. Muchas mujeres delgadas tienen SOP, y muchas mujeres con sobrepeso NO lo tienen. Solo un médico puede ver el cuadro completo con estudios 😊"

Pregunta: "¿Qué es el SOP?"
❌ MAL: "El SOP es un trastorno endocrino metabólico complejo que afecta a mujeres en edad reproductiva"
✅ BIEN: "¡Buena pregunta! 😊 Imagina que tus ovarios están un poco 'confundidos' sobre cuándo hacer su trabajo. El SOP (Síndrome de Ovario Poliquístico) básicamente significa que tus hormonas están un poco desbalanceadas, lo que puede causar ciclos irregulares, acné, o dificultad para bajar de peso. Es súper común - como 1 de cada 10 mujeres lo tiene. No es tu culpa, no hiciste nada mal, y hay muchas formas de manejarlo 💜"

Pregunta: "Me siento horrible, ¿es por el SOP?"
❌ MAL: "La guía ESHRE 2023 indica mayor prevalencia de depresión en SOP"
✅ BIEN: "Lamento mucho que te sientas así 💜. Primero que nada: tus emociones son totalmente válidas. Y sí, hay una conexión real entre el SOP y cómo nos sentimos emocionalmente. No estás 'loca' ni eres 'dramática' - hay razones biológicas. Las mismas hormonas que afectan tus ciclos también pueden afectar tu ánimo. Es como cuando estás con el periodo y te sientes más sensible, pero puede ser más intenso con SOP. Muchas mujeres con SOP experimentan ansiedad o depresión, y hay ayuda disponible. ¿Has hablado con tu médico sobre cómo te sientes?"

🎯 TU META: Que la persona se sienta ESCUCHADA, ENTENDIDA y con información ÚTIL, no como si leyera un documento médico aburrido.

Fecha actual: {date}
"""

# ==========================================
# ANÁLISIS DE IMÁGENES
# ==========================================

class MedicalImageAnalyzer:
    """Analiza imágenes médicas de forma educativa"""
    
    def __init__(self, model):
        self.model = model
    
    def analyze_lab_results(self, image):
        """Analiza resultados de laboratorio"""
        
        prompt = """Analiza estos RESULTADOS DE LABORATORIO de forma educativa y específica.

**ESTRUCTURA TU RESPUESTA ASÍ:**

📊 **LO QUE VEO EN TU ANÁLISIS:**
(Describe específicamente qué tipo de análisis es, qué valores aparecen)

🧠 **QUÉ SIGNIFICAN ESTOS ESTUDIOS EN SOP:**
(Explica CADA valor visible y por qué es relevante en SOP)
Ejemplo: "Veo que tienes análisis de testosterona. Este valor es importante porque en SOP los niveles de andrógenos (hormonas masculinas) pueden estar elevados, lo que explica síntomas como acné o vello excesivo"

💡 **POR QUÉ TU MÉDICO PIDIÓ ESTO:**
(Conecta los estudios con el diagnóstico/seguimiento de SOP)

🎯 **LO QUE ESTOS RESULTADOS PUEDEN INDICAR (EN GENERAL):**
(Sin interpretar valores específicos, explica qué patrones busca el médico)

**REGLAS:**
- Sé MUY específica con lo que ves
- Explica CADA tipo de estudio visible
- Usa lenguaje simple pero completo
- NO digas si valores están altos/bajos
- NO diagnostiques

Si no es un resultado de laboratorio claro, dilo amablemente.
"""
        
        try:
            response = self.model.generate_content([prompt, image])
            return response.text
        except Exception as e:
            if "safety" in str(e).lower():
                return "⚠️ No pude analizar esta imagen por filtros de seguridad. Intenta con otra o consulta directamente con tu médico. 💜"
            return f"❌ Error: {str(e)}"
    
    def analyze_cycle_chart(self, image):
        """Analiza gráfica de ciclos menstruales"""
        
        prompt = """Analiza esta GRÁFICA DE CICLOS de forma educativa, específica y útil.

**ESTRUCTURA TU RESPUESTA ASÍ:**

📅 **LO QUE VEO EN TU REGISTRO:**
(Describe específicamente: app que usas, qué marca, cuántos ciclos registrados, qué síntomas anota)

📊 **ANÁLISIS DETALLADO:**
Observo los siguientes patrones:
- Duración de ciclos que puedo ver: [ser específica]
- Síntomas que registras: [listar lo que se ve]
- Regularidad aparente: [comentar si los ciclos parecen consistentes]

🔍 **QUÉ BUSCA EL MÉDICO EN ESTO (según ESHRE 2023):**
- Ciclos regulares: 21-35 días
- En SOP: ciclos <21 o >35 días (disfunción ovulatoria)
- Patrones de síntomas que se repiten
- Relación entre síntomas y fase del ciclo

💜 **POR QUÉ ESTE REGISTRO ES VALIOSO:**
(Explica cómo este registro específico ayuda al diagnóstico)

🎯 **LO QUE PODRÍAS AGREGAR PARA HACERLO AÚN MEJOR:**
(Sugerencias específicas basadas en lo que ya tiene)

**TONO:** Validador, específico, útil. Felicítala por llevar el registro.

Si no es una gráfica de ciclos, dilo amablemente.
"""
        
        try:
            response = self.model.generate_content([prompt, image])
            return response.text
        except Exception as e:
            if "safety" in str(e).lower():
                return "⚠️ No pude analizar por seguridad. Intenta con otra imagen. 💜"
            return f"❌ Error: {str(e)}"
    
    def analyze_ultrasound(self, image):
        """Analiza ecografía (MUY limitado)"""
        
        prompt = """Analiza esta ECOGRAFÍA con MUCHA PRECAUCIÓN.

**ESTRUCTURA TU RESPUESTA ASÍ:**

🔬 **LO QUE IDENTIFICO:**
(Solo tipo general: ecografía pélvica, transvaginal, tiene etiquetas, fecha, etc.)

📚 **QUÉ BUSCA EL MÉDICO EN ECOGRAFÍAS DE SOP (según ESHRE 2023):**
- Morfología ovárica: ≥20 folículos de 2-9mm por ovario
- Volumen ovárico: ≥10ml
- Esto es UNO de los 3 criterios diagnósticos
- En mujeres <35 años con ciclos regulares puede no ser necesaria

⚠️ **POR QUÉ NO PUEDO "LEER" TU ECOGRAFÍA:**
La interpretación de ecografías requiere:
- Años de formación especializada
- Ver el estudio en movimiento (no solo una foto)
- Conocer contexto completo (edad, síntomas, otros estudios)
- Equipo calibrado correctamente

✅ **LO QUE SÍ PUEDES HACER:**
- Pedir al radiólogo el REPORTE OFICIAL por escrito
- Llevar ese reporte a tu ginecólogo
- Hacer preguntas específicas sobre hallazgos mencionados

**TONO:** Muy cauto, educativo sobre limitaciones.

Si no es ecografía, dilo.
"""
        
        try:
            response = self.model.generate_content([prompt, image])
            return response.text
        except Exception as e:
            if "safety" in str(e).lower():
                return "⚠️ No puedo analizar esta imagen. Consulta directamente con tu médico. 💜"
            return f"❌ Error: {str(e)}"
    
    def analyze_general(self, image):
        """Análisis general mejorado"""
        
        prompt = """Analiza esta imagen médica de forma educativa y específica.

**PASOS:**

1. **IDENTIFICA** qué tipo de imagen es (laboratorio, ciclos, ecografía, otro)

2. **ANALIZA ESPECÍFICAMENTE** basándote en el tipo:
   - Describe lo que ves con detalle
   - Explica qué significan esos estudios en contexto de SOP
   - Conecta con criterios ESHRE 2023 relevantes

3. **EXPLICA** por qué este tipo de estudio es útil para el diagnóstico/seguimiento

4. **SUGIERE** qué más podría ser útil registrar o preguntar

**REGLAS:**
❌ NO interpretes valores específicos
❌ NO diagnostiques
✅ Sé específica con lo que ves
✅ Usa lenguaje simple
✅ Conecta con vida real

Si no es imagen médica clara, dilo amablemente.
"""
        
        try:
            response = self.model.generate_content([prompt, image])
            return response.text
        except Exception as e:
            if "safety" in str(e).lower():
                return "⚠️ No puedo analizar. Consulta con tu médico. 💜"
            return f"❌ Error: {str(e)}"

# Inicializar analizador de imágenes
image_analyzer = MedicalImageAnalyzer(model)

# ==========================================
# BÚSQUEDA Y RESPUESTA
# ==========================================

def search_context(query, k=5):
    """Búsqueda semántica simple"""
    try:
        # Sin filtro de score - retorna los k más relevantes
        docs = vectorstore.similarity_search(query, k=k)
        return docs
    except Exception as e:
        st.error(f"Error búsqueda: {str(e)}")
        return []

def generate_response(user_query, chat_history=[]):
    """Genera respuesta con contexto del PDF"""
    
    # Buscar contexto relevante
    docs = search_context(user_query, k=4)
    
    if not docs:
        return """Lo siento, no encontré información específica en la guía médica que consulto.

Te recomiendo:
- Consultar con tu ginecólogo o endocrinólogo
- Buscar en fuentes médicas oficiales
- Si es urgente, contactar a tu médico

¿Tienes otra pregunta sobre el SOP? 💜"""
    
    # Preparar contexto
    context = "\n\n---\n\n".join([
        f"[{d.metadata.get('fuente', 'Guía médica')}]\n{d.page_content}"
        for d in docs
    ])
    
    # Construir prompt
    full_prompt = SYSTEM_PROMPT.format(
        context=context,
        date=datetime.now().strftime('%Y-%m-%d')
    )
    
    # Agregar historial si existe
    if chat_history and len(chat_history) > 0:
        full_prompt += "\n\n**CONVERSACIÓN PREVIA:**\n"
        recent = chat_history[-6:]
        for msg in recent:
            role = "Usuario" if msg["role"] == "user" else "Asistente"
            full_prompt += f"\n{role}: {msg['content']}\n"
    
    full_prompt += f"\n\n**PREGUNTA ACTUAL:**\n{user_query}\n\n**TU RESPUESTA:**"
    
    # Generar respuesta
    try:
        response = model.generate_content(full_prompt)
        answer = response.text
        
        # Agregar footer si cita guía
        if "guía" in answer.lower() or "eshre" in answer.lower():
            answer += "\n\n---\n📚 *Información basada en guías médicas ESHRE 2023*"
        
        return answer
    
    except Exception as e:
        error_str = str(e).lower()
        
        if "safety" in error_str or "block" in error_str:
            return "⚠️ Mi sistema de seguridad bloqueó esta respuesta. Intenta reformular tu pregunta o consulta directamente con tu médico. 💜"
        elif "quota" in error_str or "429" in error_str:
            return "⏱️ He alcanzado mi límite de uso. Intenta más tarde. 💜"
        else:
            return f"❌ Error técnico. Intenta de nuevo. 💜"

# ==========================================
# UI PRINCIPAL
# ==========================================

st.markdown('<h1 style="color: #9b59b6;">💜 Guía Educativa SOP Avanzada</h1>', unsafe_allow_html=True)
st.markdown("*Con Hugging Face + Gemini 2.0 Flash + Análisis de Imágenes*")

st.markdown("""
🧠 **Búsqueda semántica inteligente** (Langchain + Hugging Face)  
📚 **Basado exclusivamente en guías ESHRE 2023**  
📸 **Análisis educativo de imágenes médicas**

⚠️ **Importante:** Esta información es educativa. Para diagnóstico y tratamiento, consulta profesionales.
""")

st.markdown("---")

# ==========================================
# TABS: CHAT + IMÁGENES
# ==========================================

tab1, tab2, tab3 = st.tabs(["💬 Chat Médico", "📸 Análisis de Imágenes", "🎥 Recursos"])

# ========================================
# TAB 1: CHAT
# ========================================

with tab1:
    # Inicializar historial
    if 'messages' not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": """¡Hola! Me llamo Sofía 💜

Soy tu **guía educativa sobre el Síndrome de Ovario Poliquístico (SOP)**.

**Me baso exclusivamente en:**
- 📚 Guía Internacional ESHRE 2023
- 🔬 Evidencia científica verificada
- 🧠 Búsqueda semántica inteligente

**Puedo ayudarte con:**
- 🔍 Diagnóstico y criterios
- 💊 Opciones de tratamiento
- 🤰 Fertilidad y embarazo
- 🥗 Alimentación y ejercicio
- 🧠 Salud mental y emocional
- ❤️ Riesgos de salud a largo plazo

**También analizo imágenes educativamente:**
- 🧪 Resultados de laboratorio
- 📅 Gráficas de ciclos menstruales
- 🔬 Ecografías (explicación general)

🔒 **Mi compromiso:** Solo información verificable. Si no sé algo, te lo digo honestamente.

¿Qué te gustaría saber sobre el SOP? 😊"""
            }
        ]
    
    # Mostrar mensajes
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
# Input del usuario
if prompt := st.chat_input("Escribe tu pregunta sobre SOP... 💭"):
    
    # Agregar mensaje del usuario
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generar respuesta
    with st.chat_message("assistant"):
        with st.spinner("🔍 Buscando en guía médica..."):
            response = generate_response(
                prompt,
                st.session_state.messages[:-1]
            )
            st.markdown(response)
    
    # Guardar respuesta
    st.session_state.messages.append({"role": "assistant", "content": response})

# ========================================
# TAB 2: ANÁLISIS DE IMÁGENES
# ========================================

with tab2:
    st.markdown("## 📸 Análisis Educativo de Imágenes Médicas")
    
    st.markdown("""
    Puedo ayudarte a **entender** (no diagnosticar) imágenes médicas relacionadas con SOP:
    
    - 🧪 **Resultados de laboratorio** (hormonas, glucosa, lípidos, etc.)
    - 📅 **Gráficas de ciclos menstruales** (calendarios, apps de seguimiento)
    - 🔬 **Ecografías** (solo explicación general, no interpretación)
    
    ⚠️ **MUY IMPORTANTE:**  
    Este análisis es **EDUCATIVO únicamente** para:
    - Entender qué significan tus estudios
    - Preparar preguntas para tu médico
    - Aprender sobre el SOP
    
    **SOLO tu médico puede:**
    - Interpretar tus resultados específicos
    - Darte un diagnóstico
    - Prescribir tratamientos
    """)
    
    st.markdown("---")
    
    # Selector de tipo de imagen
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🧪 Laboratorio", use_container_width=True):
            st.session_state.image_type = "lab"
    
    with col2:
        if st.button("📅 Ciclos", use_container_width=True):
            st.session_state.image_type = "cycle"
    
    with col3:
        if st.button("🔬 Ecografía", use_container_width=True):
            st.session_state.image_type = "ultrasound"
    
    with col4:
        if st.button("❓ No sé", use_container_width=True):
            st.session_state.image_type = "general"
    
    # Mostrar tipo seleccionado
    if 'image_type' not in st.session_state:
        st.session_state.image_type = "general"
    
    type_labels = {
        "lab": "🧪 Resultados de Laboratorio",
        "cycle": "📅 Gráfica de Ciclos",
        "ultrasound": "🔬 Ecografía",
        "general": "❓ Análisis General"
    }
    
    st.info(f"**Tipo seleccionado:** {type_labels[st.session_state.image_type]}")
    
    # Upload de imagen
    uploaded_file = st.file_uploader(
        "Sube tu imagen (PNG, JPG, JPEG)",
        type=['png', 'jpg', 'jpeg'],
        help="Formatos permitidos: PNG, JPG, JPEG"
    )
    
    if uploaded_file:
        col_img, col_analysis = st.columns([1, 1])
        
        with col_img:
            st.image(uploaded_file, caption="Imagen subida", use_container_width=True)
        
        with col_analysis:
            if st.button("🔍 Analizar Educativamente", type="primary", use_container_width=True):
                with st.spinner("📊 Analizando imagen..."):
                    try:
                        img = PIL.Image.open(uploaded_file)
                        
                        # Análisis según tipo
                        if st.session_state.image_type == "lab":
                            analysis = image_analyzer.analyze_lab_results(img)
                        elif st.session_state.image_type == "cycle":
                            analysis = image_analyzer.analyze_cycle_chart(img)
                        elif st.session_state.image_type == "ultrasound":
                            analysis = image_analyzer.analyze_ultrasound(img)
                        else:
                            analysis = image_analyzer.analyze_general(img)
                        
                        st.markdown("### 📋 Análisis Educativo:")
                        st.info(analysis)
                        
                        # PREGUNTAS SUGERIDAS ESPECÍFICAS
                        st.markdown("---")
                        st.markdown("### 💡 Preguntas para llevar a tu médico:")
                        
                        if st.session_state.image_type == "cycle":
                            st.success("""
**📋 Basándome en tu registro, pregúntale a tu ginecólogo:**

1. **"Doctor, ¿mis ciclos son consistentes con SOP o hay otro diagnóstico posible?"**
   - Muéstrale los patrones que has registrado

2. **"¿Los síntomas que marco (como [síntomas específicos que viste]) son típicos del SOP?"**
   
3. **"Basándote en estos ciclos, ¿debería hacerme estudios hormonales específicos?"**
   
4. **"¿Hay algo más que debería estar registrando para ayudarte con el diagnóstico?"**

5. **"¿Este patrón sugiere que necesito tratamiento, o es suficiente con seguimiento?"**

💡 **Tip:** Lleva tu celular con el registro completo o screenshots de varios meses.
                            """)
                        
                        elif st.session_state.image_type == "lab":
                            st.success("""
**📋 Preguntas específicas para tu médico sobre estos resultados:**

1. **"¿Estos valores están dentro de rangos normales para mi edad y situación?"**
   - Pídele que te explique CADA valor que salió

2. **"¿Alguno de estos resultados sugiere investigar SOP más a fondo?"**
   
3. **"¿Necesito repetir algún estudio en otro momento del ciclo?"**
   - Algunos valores hormonales cambian según la fase

4. **"¿Hay otros estudios que deberíamos hacer para completar el diagnóstico?"**

5. **"Basándote en estos resultados, ¿cuál sería el siguiente paso?"**

💡 **Tip:** Pide una copia de los resultados para tu archivo personal.
                            """)
                        
                        elif st.session_state.image_type == "ultrasound":
                            st.success("""
**📋 Preguntas sobre tu ecografía:**

1. **"¿El reporte menciona morfología ovárica poliquística?"**
   - Pide que te explique qué significa exactamente

2. **"¿Los hallazgos de la eco, junto con mis síntomas, cumplen criterios de SOP?"**

3. **"¿Es necesario repetir la ecografía en otra fase del ciclo?"**
   
4. **"¿Hay otros hallazgos que deba conocer además del SOP?"**

5. **"Basándote en esta eco y mis otros estudios, ¿qué tratamiento recomiendas?"**

💡 **Tip:** Pide el reporte oficial completo del radiólogo, no solo la imagen.
                            """)
                        
                        else:
                            st.success("""
**📋 Preguntas generales para tu médico:**

1. **"¿Qué información te da este estudio sobre mi condición?"**

2. **"¿Los resultados sugieren que necesito más pruebas?"**

3. **"¿Cómo se relaciona esto con mis síntomas?"**

4. **"¿Qué pasos siguen después de revisar esto?"**

💡 **Tip:** Lleva todos tus estudios organizados por fecha.
                            """)
                        
                        # DISCLAIMER ÚNICO AL FINAL
                        st.markdown("---")
                        st.warning("""
⚠️ **Recordatorio importante:**

Este análisis es **educativo** para ayudarte a entender mejor tus estudios y preparar tu consulta médica.

**Solo tu médico puede:**
✅ Interpretar tus resultados específicos  
✅ Darte un diagnóstico  
✅ Prescribir tratamientos  

Si tienes dudas urgentes, contacta a tu médico. 💜
                        """)
                        
                        # Guardar en historial
                        if 'image_analyses' not in st.session_state:
                            st.session_state.image_analyses = []
                        
                        st.session_state.image_analyses.append({
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "type": type_labels[st.session_state.image_type],
                            "analysis": analysis
                        })
                        
                        st.success("✅ Análisis completado")
                    
                    except Exception as e:
                        st.error(f"❌ Error al abrir imagen: {str(e)}")
    
    # Mostrar historial de análisis
    if 'image_analyses' in st.session_state and len(st.session_state.image_analyses) > 0:
        st.markdown("---")
        st.markdown("### 📜 Historial de Análisis")
        
        for idx, item in enumerate(reversed(st.session_state.image_analyses)):
            with st.expander(f"{item['type']} - {item['timestamp']}", expanded=(idx==0)):
                st.markdown(item['analysis'])

# ========================================
# TAB 3: RECURSOS MULTIMEDIA
# ========================================

with tab3:
    st.markdown("## 🎥 Recursos Educativos sobre el SOP")
    
    st.info("""
    💡 **Complementa tu aprendizaje** con estos recursos multimedia seleccionados 
    especialmente para ti. Puedes consultarlos cuando quieras profundizar más sobre el SOP.
    """)
    
    st.markdown("---")
    
    # ==========================================
    # SECCIÓN VIDEO
    # ==========================================
    
    st.markdown("### 📺 Video Educativo: Entendiendo el SOP")
    
    # Video principal centrado
    col_space1, col_video, col_space2 = st.columns([0.5, 2, 0.5])
    
    with col_video:
        st.video("https://www.youtube.com/watch?v=3SjmYGY5KZQ")
        
        st.caption("""
        🎬 **Duración:** 7 minutos  
        📚 **Contenido:** Qué es el SOP, síntomas principales, diagnóstico y opciones de tratamiento  
        👩‍⚕️ **Presentado por:** [Nombre del especialista/canal]
        """)
    
    # Botones de acción
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.link_button(
            "▶️ Ver en YouTube",
            "https://www.youtube.com/watch?v=3SjmYGY5KZQ",
            use_container_width=True
        )
    
    with col2:
        st.link_button(
            "📝 Ver transcripción",
            "https://www.youtube.com/watch?v=3SjmYGY5KZQ",
            use_container_width=True
        )
    
    with col3:
        st.link_button(
            "💬 Compartir",
            f"https://wa.me/?text=Mira este video sobre SOP: https://www.youtube.com/watch?v=3SjmYGY5KZQ",
            use_container_width=True
        )
    
    st.markdown("---")
    
    # ==========================================
    # SECCIÓN PODCAST
    # ==========================================
    
    st.markdown("### 🎙️ Podcast: Viviendo con SOP")
    
    st.markdown("""
    Escucha experiencias reales, consejos prácticos y entrevistas con especialistas.
    """)
    
    # Spotify embed centrado
    col_space1, col_podcast, col_space2 = st.columns([0.5, 2, 0.5])
    
    with col_podcast:
        st.markdown("""
        <iframe style="border-radius:12px" 
        src="https://open.spotify.com/embed/episode/7CMJkpyco1zkT9L2Beb7O5?utm_source=generator" 
        width="100%" height="232" frameBorder="0" 
        allowfullscreen="" 
        allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture" 
        loading="lazy"></iframe>
        """, unsafe_allow_html=True)
        
        st.caption("""
        🎧 **Duración:** 17 minutos  
        💬 **Temas:** Manejo emocional, tips de alimentación, historias de éxito  
        🎤 **Anfitrión:** [Nombre del host]
        """)
    
    # Botones de acción podcast
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.link_button(
            "🎵 Abrir en Spotify",
            "https://open.spotify.com/episode/7CMJkpyco1zkT9L2Beb7O5",
            use_container_width=True
        )
    
    with col2:
        st.link_button(
            "📱 Escuchar en app",
            "https://open.spotify.com/episode/7CMJkpyco1zkT9L2Beb7O5",
            use_container_width=True
        )
    
    with col3:
        st.link_button(
            "💬 Compartir",
            f"https://wa.me/?text=Escucha este podcast sobre SOP: https://open.spotify.com/episode/7CMJkpyco1zkT9L2Beb7O5",
            use_container_width=True
        )
    
    st.markdown("---")
    
    # ==========================================
    # PLAYLIST / MÁS RECURSOS
    # ==========================================
    
    st.markdown("### 📚 Más Recursos Recomendados")
    
    # Tabs secundarias para organizar más contenido
    subtab1, subtab2, subtab3 = st.tabs(["🎥 Más Videos", "🎙️ Más Podcasts", "📖 Lecturas"])
    
    with subtab1:
        st.markdown("#### Videos adicionales sobre SOP")
        
        video_col1, video_col2 = st.columns(2)
        
        with video_col1:
            st.markdown("**🍎 Alimentación y SOP**")
            st.video("https://www.youtube.com/watch?v=VIDEO_ID_2")
            st.caption("10 min • Nutrición especializada")
        
        with video_col2:
            st.markdown("**🏃‍♀️ Ejercicio para SOP**")
            st.video("https://www.youtube.com/watch?v=VIDEO_ID_3")
            st.caption("12 min • Rutinas recomendadas")
        
        st.markdown("---")
        
        video_col3, video_col4 = st.columns(2)
        
        with video_col3:
            st.markdown("**🤰 Fertilidad y SOP**")
            st.video("https://www.youtube.com/watch?v=VIDEO_ID_4")
            st.caption("15 min • Opciones de tratamiento")
        
        with video_col4:
            st.markdown("**🧠 Salud mental**")
            st.video("https://www.youtube.com/watch?v=VIDEO_ID_5")
            st.caption("8 min • Manejo emocional")
    
    with subtab2:
        st.markdown("#### Serie de podcasts recomendados")
        
        # Lista de episodios
        st.markdown("""
        **🎧 Episodio 1: Mi diagnóstico de SOP**  
        <iframe style="border-radius:12px" 
        src="https://open.spotify.com/embed/episode/EPISODE_ID_1?utm_source=generator" 
        width="100%" height="152" frameBorder="0"></iframe>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("""
        **🎧 Episodio 2: Hablemos de síntomas**  
        <iframe style="border-radius:12px" 
        src="https://open.spotify.com/embed/episode/EPISODE_ID_2?utm_source=generator" 
        width="100%" height="152" frameBorder="0"></iframe>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("""
        **🎧 Episodio 3: Entrevista con endocrinólogo**  
        <iframe style="border-radius:12px" 
        src="https://open.spotify.com/embed/episode/EPISODE_ID_3?utm_source=generator" 
        width="100%" height="152" frameBorder="0"></iframe>
        """, unsafe_allow_html=True)
    
    with subtab3:
        st.markdown("#### Artículos y guías descargables")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **📄 Documentos oficiales:**
            - 📗 [Guía ESHRE 2023 (resumen)](https://www.eshre.eu/)
            - 📘 [OMS - Salud reproductiva](https://www.who.int/)
            - 📙 [CDC - Información sobre SOP](https://www.cdc.gov/)
            """)
            
            # Botón de descarga de tu PDF
            if os.path.exists("guia_sop.pdf"):
                st.markdown("---")
                with open("guia_sop.pdf", "rb") as pdf_file:
                    st.download_button(
                        label="📥 Descargar Guía Completa (PDF)",
                        data=pdf_file,
                        file_name="guia_sop_completa.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
        
        with col2:
            st.markdown("""
            **🌐 Comunidades de apoyo:**
            - 💬 [Grupo de apoyo SOP México](https://ejemplo.com)
            - 💜 [Foro internacional SOP](https://ejemplo.com)
            - 📱 [Instagram @sop_awareness](https://instagram.com)
            
            **📱 Apps recomendadas:**
            - 📅 Flo (seguimiento de ciclos)
            - 🍽️ MyFitnessPal (nutrición)
            - 🧘‍♀️ Calm (meditación)
            """)
    
    # ==========================================
    # CALL TO ACTION FINAL
    # ==========================================
    
    st.markdown("---")
    
    st.success("""
    ### 💜 ¿Te fueron útiles estos recursos?
    
    **Sigue aprendiendo:**
    - 💬 Regresa al **Chat** para hacer preguntas específicas
    - 📸 Usa el **Análisis de Imágenes** para entender tus estudios
    - 🗺️ Busca **profesionales** cerca de ti en el menú lateral
    
    ¡Recuerda que no estás sola en esto! 🤗
    """)

# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:
    if os.path.exists("logo.png"):
        st.image("logo.png", width=200)
    
    st.markdown("---")
    # 🗺️ BUSCAR PROFESIONALES👩‍⚕️
    st.markdown("🗺️ Buscar Profesionales 👩‍⚕️")
    st.caption("Encuentra especialistas en tu zona")
    
    user_city = st.text_input(
        "Tu ciudad o municipio ⚕️",
        placeholder="Ej: Guadalajara",
        key="city_search"
    )
    
    if user_city:
        st.markdown("**Buscar en Google Maps:**")
        
        gine_search = f"https://www.google.com/maps/search/ginecólogo+SOP+{user_city.replace(' ', '+')}"
        endo_search = f"https://www.google.com/maps/search/endocrinólogo+{user_city.replace(' ', '+')}"
        psico_search = f"https://www.google.com/maps/search/psicólogo+{user_city.replace(' ', '+')}"
        nutri_search = f"https://www.google.com/maps/search/nutriólogo+{user_city.replace(' ', '+')}"
        
        st.link_button("🩺 Ginecólogos", gine_search, use_container_width=True)
        st.link_button("💉 Endocrinólogos", endo_search, use_container_width=True)
        st.link_button("🧠 Psicólogos", psico_search, use_container_width=True)
        st.link_button("🥗 Nutriólogos", nutri_search, use_container_width=True)
        
        st.caption("💡 Se abrirán en Google Maps")

    # Preguntas rápidas
    st.markdown("---")
    st.markdown("### 🌸 Temas guiados")
    
    quick_questions = [
        "¿Qué es el SOP?",
        "¿Cómo se diagnostica?",
        "Tratamientos disponibles",
        "¿Puedo embarazarme?",
        "Dieta para SOP",
        "Ejercicio recomendado",
        "Riesgo de diabetes",
        "Salud mental y SOP"
    ]
    
    for q in quick_questions:
        if st.button(q, key=f"quick_{q}", use_container_width=True):
            # Agregar pregunta del usuario
            st.session_state.messages.append({"role": "user", "content": q})
            
            # Generar respuesta inmediatamente
            with st.spinner("🔍 Buscando en guía médica..."):
                response = generate_response(q, st.session_state.messages[:-1])
                st.session_state.messages.append({"role": "assistant", "content": response})
            
            st.rerun()
    st.markdown("### 📊 Estado del Sistema")
    
    st.success("✅ Sistema activo")
    st.caption("🧠 Hugging Face embeddings")
    st.caption("🔍 Búsqueda semántica")
    st.caption("📸 Análisis de imágenes")
    st.caption("🤖 Gemini 2.0 Flash")
    
    st.markdown("---")
    st.markdown("### 💬 Controles")
    
    # Stats
    if 'messages' in st.session_state:
        user_msgs = len([m for m in st.session_state.messages if m["role"] == "user"])
        st.metric("Preguntas realizadas", user_msgs)
    
    if 'image_analyses' in st.session_state:
        st.metric("Imágenes analizadas", len(st.session_state.image_analyses))
    
    # Botones de control
    if st.button("🗑️ Limpiar Chat", type="secondary", use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant", "content": "💜 ¡Chat reiniciado! ¿En qué puedo ayudarte?"}
        ]
        st.rerun()
    
    if st.button("🗑️ Limpiar Historial Imágenes", type="secondary", use_container_width=True):
        st.session_state.image_analyses = []
        st.success("✅ Historial limpiado")
        st.rerun()
    
    # Exportar conversación
    st.markdown("---")
    st.markdown("### 💾 Exportar")
    
    if st.button("📥 Descargar Conversación", use_container_width=True):
        if 'messages' in st.session_state and len(st.session_state.messages) > 1:
            chat_text = "\n\n".join([
                f"{'Usuario' if m['role'] == 'user' else 'Sofía'}: {m['content']}"
                for m in st.session_state.messages
            ])
            
            st.download_button(
                label="⬇️ Descargar TXT",
                data=chat_text,
                file_name=f"chat_sop_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
    
    # Información adicional
    st.markdown("---")
    st.markdown("### ℹ️ Sobre este chatbot")
    st.caption("""
    **Tecnología:**
    - Gemini 2.0 Flash (Google)
    - Hugging Face embeddings
    - Langchain (búsqueda semántica)
    - ChromaDB (vectorstore)
    - Vision API (análisis imágenes)
    
    **Fuentes:**
    - Guía Internacional ESHRE 2023
    - Basado 100% en evidencia
    
    **Versión:** 2.0 Advanced (Hugging Face)
    """)