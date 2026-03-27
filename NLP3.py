# Librerias recomendadas para porder ejecutar 
#pip install streamlit
#pip install spacy
#python -m scapy download es_core_news_sm
#pip install -U scitkit-learn
#


import streamlit as st
import spacy
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import time

# --- Configuración de la página ---
st.set_page_config(page_title="LegalAI - NLP Assistant", page_icon="⚖️")

# --- Cargar Modelo NLP (spaCy) ---
@st.cache_resource
def load_nlp_model():
    try:
        return spacy.load("es_core_news_sm")
    except OSError:
        st.error("Modelo spaCy no encontrado. Ejecuta: `python -m spacy download es_core_news_sm`")
        return None

nlp = load_nlp_model()

# --- Preprocesamiento de texto ---
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\sáéíóúñ]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- Expansión de sinónimos legales ---
LEGAL_SYNONYMS = {
    "Familia": ["divorcio", "separación", "custodia", "hijos", "herencia", "testamento", "pensión", "alimentos", "familia", "cónyuge", "padres"],
    "Laboral": ["despido", "trabajo", "empleo", "jefe", "empresa", "salario", "sueldo", "horas extras", "acoso laboral", "finiquito", "paro"],
    "Penal": ["robo", "hurto", "denuncia", "policía", "cárcel", "delito", "acusado", "juicio", "abogado penalista", "detenido", "fiscal"],
    "Civil": ["contrato", "alquiler", "propiedad", "casa", "terreno", "demanda civil", "daños", "perjuicios", "vecino", "comunidad"]
}

def expand_text_with_synonyms(text):
    expanded = text.lower()
    for category, synonyms in LEGAL_SYNONYMS.items():
        for synonym in synonyms:
            if synonym in expanded:
                expanded += f" {category}"
    return expanded

# --- Entrenamiento del Clasificador ---
@st.cache_resource
def train_legal_classifier():
    training_data = [
        ("Mi esposo quiere el divorcio y la custodia de los hijos", "Familia"),
        ("Necesito tramitar la herencia de mi abuelo fallecido", "Familia"),
        ("Quiero pedir la pensión alimenticia para mis hijos", "Familia"),
        ("Mi pareja y yo nos separamos, ¿quién se queda con la casa?", "Familia"),
        ("Necesito hacer un testamento para dejar mis bienes a mis hijos", "Familia"),
        ("Mi ex no me deja ver a mis niños, ¿qué puedo hacer?", "Familia"),
        ("Quiero adoptar un niño, ¿cuáles son los pasos legales?", "Familia"),
        ("Mi cónyuge me fue infiel y quiero divorciarme", "Familia"),
        ("Necesito modificar la custodia de mis hijos por cambio de domicilio", "Familia"),
        ("¿Cómo puedo reclamar la patria potestad de mi hijo?", "Familia"),
        ("Me despidieron injustificadamente de mi trabajo", "Laboral"),
        ("La empresa no me paga las horas extras ni el seguro", "Laboral"),
        ("Mi jefe me acosa laboralmente y quiero denunciar", "Laboral"),
        ("No me quieren dar mi finiquito después de renunciar", "Laboral"),
        ("Trabajo sin contrato y me quieren correr sin indemnización", "Laboral"),
        ("Me bajaron el sueldo sin avisarme ni justificarlo", "Laboral"),
        ("La empresa no me da vacaciones ni días de descanso", "Laboral"),
        ("Me discriminaron en el trabajo por mi edad/género", "Laboral"),
        ("Tuve un accidente laboral y no me quieren pagar la incapacidad", "Laboral"),
        ("Me corrieron por estar enfermo, ¿es legal?", "Laboral"),
        ("Me robaron el coche y necesito denunciar", "Penal"),
        ("Me acusan de un delito que no cometí", "Penal"),
        ("Fui detenido injustamente y necesito un abogado penalista", "Penal"),
        ("Me amenazaron de muerte y quiero poner una denuncia", "Penal"),
        ("Me golpearon en la calle y el agresor sigue libre", "Penal"),
        ("Me estafaron por internet con una compra falsa", "Penal"),
        ("Alguien entró a mi casa sin permiso mientras no estaba", "Penal"),
        ("Me retuvieron contra mi voluntad, ¿es secuestro?", "Penal"),
        ("Me acusan de fraude fiscal pero yo declaré todo", "Penal"),
        ("Fui víctima de violencia doméstica y necesito protección", "Penal"),
        ("El inquilino no me paga el alquiler desde hace meses", "Civil"),
        ("Tengo un problema con los linderos de mi propiedad", "Civil"),
        ("Quiero demandar por negligencia médica", "Civil"),
        ("Necesito un contrato de arrendamiento", "Civil"),
        ("Mi vecino construyó invadiendo mi terreno", "Civil"),
        ("Me deben dinero por un préstamo personal y no me pagan", "Civil"),
        ("Tuve un accidente de tráfico y quiero reclamar daños", "Civil"),
        ("La comunidad de vecinos no me deja hacer obras en mi piso", "Civil"),
        ("Compré un producto defectuoso y la tienda no me lo cambia", "Civil"),
        ("Quiero reclamar por daños morales tras un incidente", "Civil"),
    ]
    texts, labels = zip(*training_data)
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    model.fit(texts, labels)
    return model

classifier = train_legal_classifier()

# --- Base de Datos de Abogados ---
LAWYERS_DB = {
    "Familia": {"nombre": "Dra. Elena Gómez", "exp": "15 años", "especialidad": "Divorcios y Custodias"},
    "Laboral": {"nombre": "Lic. Carlos Ruiz", "exp": "10 años", "especialidad": "Despidos y Derechos Laborales"},
    "Penal": {"nombre": "Dr. Mario Vargas", "exp": "20 años", "especialidad": "Defensa Criminal y Delitos"},
    "Civil": {"nombre": "Firma Inmobiliaria Legal", "exp": "12 años", "especialidad": "Contratos y Propiedades"}
}

# --- Funciones NLP ---

def extract_entities(text):
    if nlp is None:
        return ["Modelo NLP no disponible"]
    try:
        doc = nlp(text)
        entities = []
        legal_labels = {
            "PERSON": "Persona",
            "ORG": "Organización", 
            "DATE": "Fecha",
            "TIME": "Hora",
            "LOC": "Lugar",
            "MONEY": "Cantidad económica",
        }
        for ent in doc.ents:
            label = legal_labels.get(ent.label_, ent.label_)
            entities.append(f"{label}: {ent.text}")
        return entities if entities else ["Sin entidades específicas detectadas"]
    except:
        return ["Error al procesar entidades"]

def analyze_sentiment(text):
    text_lower = text.lower()
    urgent_score = 0
    urgent_words = {
        "urgente": 3, "ya": 2, "inmediato": 3, "ahora": 2,
        "policía": 4, "cárcel": 4, "robo": 3, "golpe": 3,
        "amenaza": 4, "peligro": 3, "emergencia": 4
    }
    for word, weight in urgent_words.items():
        if word in text_lower:
            urgent_score += weight
    if urgent_score >= 5:
        return "ALTA URGENCIA", urgent_score
    elif urgent_score >= 2:
        return "PRIORIDAD MEDIA", urgent_score
    else:
        return "CONSULTA ESTÁNDAR", urgent_score

def get_legal_category(text):
    expanded_text = expand_text_with_synonyms(text)
    prediction = classifier.predict([expanded_text])[0]
    probabilities = classifier.predict_proba([expanded_text])[0]
    confidence = max(probabilities) * 100
    return prediction, confidence

# === SISTEMA DE PREGUNTAS ACLARATORIAS MEJORADO ===

def get_clarifying_question(category, confidence, entities=None, sentiment=None):
    """
    Genera preguntas aclaratorias inteligentes para refinar el análisis.
    Retorna un diccionario con: question, suggestions, needs_clarification
    """
    
    # Confianza muy baja: pregunta general
    if confidence < 40:
        return {
            "question": "No estoy seguro del área legal de tu caso. ¿Podrías describirlo con más detalle?",
            "suggestions": ["Es un problema familiar", "Es sobre mi trabajo", "Es un asunto penal", "Es un tema civil o contractual"],
            "needs_clarification": True,
            "type": "general"
        }
    
    # Confianza baja: pregunta específica por categoría
    elif confidence < 60:
        questions = {
            "Familia": {
                "question": "Para ayudarte mejor en Derecho de Familia, ¿cuál es tu situación principal?",
                "suggestions": ["Divorcio o separación", "Custodia de hijos", "Herencia o testamento", "Pensión alimenticia"],
                "needs_clarification": True,
                "type": "category_specific"
            },
            "Laboral": {
                "question": "En materia laboral, ¿qué está ocurriendo en tu caso?",
                "suggestions": ["Despido injustificado", "Impago de salario u horas extras", "Acoso laboral", "Problema con contrato o finiquito"],
                "needs_clarification": True,
                "type": "category_specific"
            },
            "Penal": {
                "question": "En materia penal, ¿necesitas ayuda con?",
                "suggestions": ["Denuncia por robo o hurto", "Agresión o violencia", "Me acusan de un delito", "Necesito defensa legal urgente"],
                "needs_clarification": True,
                "type": "category_specific"
            },
            "Civil": {
                "question": "En ámbito civil, ¿tu caso está relacionado con?",
                "suggestions": ["Alquiler o propiedad", "Contrato incumplido", "Deuda o dinero prestado", "Daños y perjuicios"],
                "needs_clarification": True,
                "type": "category_specific"
            }
        }
        return questions.get(category, {
            "question": "¿Podrías darme más detalles sobre tu caso?",
            "suggestions": [],
            "needs_clarification": True,
            "type": "general"
        })
    
    # Confianza media: preguntas de refinamiento basadas en contexto
    elif confidence < 75:
        if entities:
            entity_types = [ent.get("tipo", "") if isinstance(ent, dict) else ent for ent in entities]
            
            if "DATE" in entity_types or "Fecha" in str(entities):
                return {
                    "question": "Mencionaste una fecha. ¿Este problema ocurrió recientemente o es algo antiguo?",
                    "suggestions": ["Esta semana o mes", "Hace varios meses", "Hace más de un año"],
                    "needs_clarification": True,
                    "type": "entity_followup"
                }
            if "MONEY" in entity_types or "Cantidad" in str(entities):
                return {
                    "question": "Mencionaste una cantidad económica. ¿Es un monto que te deben o que debes pagar?",
                    "suggestions": ["Me deben dinero", "Tengo que pagar", "Es una indemnización"],
                    "needs_clarification": True,
                    "type": "entity_followup"
                }
        
        if sentiment and sentiment[0] in ["ALTA URGENCIA", "PRIORIDAD MEDIA"]:
            return {
                "question": "Detecto urgencia en tu mensaje. ¿Necesitas atención inmediata?",
                "suggestions": ["Sí, es urgente", "Lo antes posible", "Puede esperar"],
                "needs_clarification": True,
                "type": "urgency_check"
            }
        
        return {
            "question": f"Detecté que es {category}. ¿Hay algún detalle específico que deba saber?",
            "suggestions": ["No, eso es todo", "Sí, hay más detalles", "Quiero hablar con un abogado"],
            "needs_clarification": True,
            "type": "refinement"
        }
    
    # Confianza alta: solo confirmación opcional
    else:
        return {
            "question": f"Estoy {confidence:.0f}% seguro de que es {category}. ¿Quieres que te conecte con un especialista?",
            "suggestions": ["Sí, contactar abogado", "Más información", "No, gracias"],
            "needs_clarification": False,
            "type": "confirmation"
        }

def refine_analysis_with_clarification(original_text, clarification_response, original_category):
    """
    Combina el texto original con la respuesta aclaratoria para re-analizar.
    """
    refined_text = f"{original_text}. Contexto adicional: {clarification_response}"
    return get_legal_category(refined_text)

# --- Estado de sesión para manejo de aclaraciones ---
if "awaiting_clarification" not in st.session_state:
    st.session_state.awaiting_clarification = False
if "clarification_context" not in st.session_state:
    st.session_state.clarification_context = {}
if "analysis_complete" not in st.session_state:
    st.session_state.analysis_complete = False

# --- Interfaz de Chat ---
st.title("LegalAI: Asistente con NLP")
st.markdown("Describe tu situación legal. Analizaré tu mensaje para recomendarte el especialista adecuado.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Input del usuario ---
if prompt := st.chat_input("Ej: Me despidieron sin justificación ayer..."):
    
    # === MODO ACLARACIÓN: procesar respuesta del usuario ===
    if st.session_state.awaiting_clarification:
        ctx = st.session_state.clarification_context
        
        # Refinar análisis con la nueva información
        categoria, confianza = refine_analysis_with_clarification(
            ctx["original_prompt"], prompt, ctx["original_category"]
        )
        
        # Resetear estado
        st.session_state.awaiting_clarification = False
        st.session_state.clarification_context = {}
        
        # Mostrar respuesta del usuario como mensaje adicional
        st.session_state.messages.append({
            "role": "user", 
            "content": f"*Respuesta aclaratoria: {prompt}*"
        })
        
        # Continuar con el flujo normal usando el análisis refinado
        clean_prompt = preprocess_text(prompt)
        entidades = extract_entities(prompt)
        urgencia_label, urgencia_score = analyze_sentiment(prompt)
        
    else:
        # === FLUJO NORMAL: primer mensaje del usuario ===
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        clean_prompt = preprocess_text(prompt)
        categoria, confianza = get_legal_category(clean_prompt)
        entidades = extract_entities(prompt)
        urgencia_label, urgencia_score = analyze_sentiment(prompt)
    
    # === Procesamiento y respuesta del asistente ===
    with st.chat_message("assistant"):
        with st.spinner("Analizando tu caso..."):
            time.sleep(1)
            
            # Obtener pregunta aclaratoria
            clarification = get_clarifying_question(categoria, confianza, entidades, (urgencia_label, urgencia_score))
            
            # === DECISIÓN: ¿Pedir aclaración o dar recomendación? ===
            if clarification["needs_clarification"] and not st.session_state.awaiting_clarification and confianza < 60:
                # Activar modo aclaración
                st.session_state.awaiting_clarification = True
                st.session_state.clarification_context = {
                    "original_prompt": prompt,
                    "original_category": categoria,
                    "original_confidence": confianza
                }
                
                # Construir mensaje de pregunta
                respuesta = f"""### Información Adicional Requerida

{clarification['question']}

**Opciones sugeridas:**
"""
                for i, suggestion in enumerate(clarification['suggestions'], 1):
                    respuesta += f"\n{i}. {suggestion}"
                
                respuesta += """

*Puedes seleccionar una opción o escribir tu propia respuesta.*
"""
                st.markdown(respuesta)
                st.session_state.messages.append({"role": "assistant", "content": respuesta})
                st.stop()  # Esperar siguiente input del usuario
            
            # === Tenemos suficiente información: dar recomendación ===
            abogado, found = get_lawyer_by_category(categoria) if 'get_lawyer_by_category' in globals() else (LAWYERS_DB.get(categoria, LAWYERS_DB["Civil"]), categoria in LAWYERS_DB)
            if not found:
                abogado = LAWYERS_DB["Civil"]
            
            # Nivel de confianza visual
            if confianza >= 70:
                confidence_text = f"Alta ({confianza:.1f}%)"
            elif confianza >= 50:
                confidence_text = f"Media ({confianza:.1f}%)"
            else:
                confidence_text = f"Baja ({confianza:.1f}%)"
            
            # Construir respuesta final
            respuesta = f"""### Analisis de tu Caso

| Concepto | Resultado |
|----------|-----------|
| Area Legal | {categoria} |
| Confianza | {confidence_text} |
| Urgencia | {urgencia_label} (puntuacion: {urgencia_score}) |

**Entidades detectadas:**
{chr(10).join(f"- {ent}" for ent in entidades)}

### Recomendacion

Basado en el analisis, te recomendamos:

> **{abogado['nombre']}**  
> Especialidad: {abogado['especialidad']}  
> Experiencia: {abogado['exp']} de experiencia

> Nota: Esta es una recomendacion preliminar. Consulta siempre directamente con un profesional.
"""
            st.markdown(respuesta)
            st.session_state.messages.append({"role": "assistant", "content": respuesta})

# --- Función auxiliar para obtener abogado (si no estaba definida) ---
def get_lawyer_by_category(category):
    category_clean = str(category).strip().title()
    if category_clean in LAWYERS_DB:
        return LAWYERS_DB[category_clean], True
    return LAWYERS_DB["Civil"], False

# --- Sidebar Técnica ---
with st.sidebar:
    st.header("Detalles del NLP")
    
    st.markdown("### Mejoras implementadas:")
    st.markdown("""
    - Preprocesamiento de texto
    - Expansion con sinonimos legales
    - Umbrales de confianza inteligentes
    - Preguntas aclaratorias contextuales
    - Re-analisis con informacion refinada
    - Entidades con etiquetas legales
    - Puntuacion de urgencia ponderada
    """)
    
    st.divider()
    
    if nlp:
        st.success("spaCy: Activo")
    else:
        st.error("spaCy: Inactivo")
    
    st.info("Clasificador: Naive Bayes + TF-IDF + Sinonimos")
    
    # Panel de prueba rápida
    if "messages" in st.session_state and len(st.session_state.messages) >= 2:
        last_user_msg = [m["content"] for m in st.session_state.messages if m["role"]=="user"][-1]
        with st.expander("Probar analisis con otro texto"):
            test_text = st.text_input("Escribe para probar:", value=last_user_msg[:50])
            if test_text:
                cat, conf = get_legal_category(test_text)
                urg, score = analyze_sentiment(test_text)
                st.write(f"**Categoria:** {cat} ({conf:.1f}%)")
                st.write(f"**Urgencia:** {urg} (score: {score})")