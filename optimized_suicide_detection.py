import os
import re
import gc
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import XLMRobertaTokenizer, XLMRobertaModel, AdamW
from sentence_transformers import SentenceTransformer, util
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score, accuracy_score, classification_report
from tqdm.auto import tqdm
import time

# Configuración
warnings.filterwarnings('ignore')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🧬 Iniciando Sistema Bioinformático en: {DEVICE}")

# ==============================================================================
# 1. BASE DE CONOCIMIENTO COMPLETA (FACTORES DE RIESGO Y VOCABULARIO)
# ==============================================================================
NEW_KNOWLEDGE_CATEGORIES = {
    "Psicológicos, emocionales o cognitivos": {
        "Sensaciones, sentimientos y emociones": {
            "Desesperanza crónica": [
                "ya no puedo más", "estoy bien durante poco tiempo y luego todo vuelve a estar igual",
                "nada va a cambiar", "por más personas que me ayuden, nadie me quita el mal",
                "nadie me quita lo que estoy sufriendo", "no me acabáis de entender",
                "nada cambia", "siempre todo igual", "siempre estoy mal", "nada sirve",
                "no tengo nada", "no hay futuro", "no voy a llegar", "no voy a conseguir nada en la vida",
                "no voy a hacer nada", "no vale la pena", "da igual lo que haga, nada sirve",
                "no va a cambiar", "ya lo he probado y no me ha servido", "sigo igual",
                "mi vida no va a cambiar", "no quiero vivir", "no quiero seguir así",
                "mejor un puente que la vida misma", "la vida me pesa", "nos bajamos de la vida",
                "quiero dejar los estudios", "por más que he ido a muchos sitios, nadie consigue darme la solución"
            ],
            "Tristeza profunda": [
                "nunca estoy bien", "nunca estoy feliz", "nunca siento felicidad", "no soy feliz",
                "quiero estar bien", "quiero ser feliz", "estoy muy triste", "me siento hundido",
                "me siento mal", "estoy disgustado", "no quiero vivir con este dolor",
                "no quiero vivir así", "todo es una mierda", "no veo futuro", "estoy en un túnel",
                "no hay salida", "mi vida no tiene sentido", "estoy en un túnel oscuro sin luz",
                "no salgo del pozo", "siento que me ahogo", "estoy harta y cansada de esta situación",
                "siento que nunca van a cambiar", "estoy en bucle", "estoy de bajón",
                "estoy chafada", "estoy roto", "estoy en la mierda", "estoy quemado",
                "estoy saturado", "estoy rayado", "estoy rallado", "estoy perdido",
                "no sé qué me pasa", "estoy fatal", "me siento como un pozo sin fondo",
                "no tengo ganas de nada, ni de levantarme"
            ],
            "Vacío y anhedonia": [
                "no hay nada que me haga ilusión", "no quiero hacer nada", "no tengo ganas de hacer nada",
                "no hay solución", "no hago nada bien", "me bajo de la vida", "no tengo a nadie",
                "me siento sola", "me siento solo", "no me encuentro bien",
                "siento que me falta el aire, pero no sé qué", "es como un agujero que no se llena",
                "no es que no sienta nada, es que siento demasiado y quiero dejar de sentir"
            ],
            "Culpa y autodesprecio": [
                "la cagué", "la cagué y la seguiré cagando", "no me merezco las cosas buenas",
                "soy un estorbo", "soy una carga", "no soy suficiente", "no valgo", "no sirvo",
                "siempre la cago", "siempre lo hago mal", "me siento culpable", "culpa tan bruta",
                "el problema he sido yo", "me da vergüenza compartir espacio con vosotros",
                "no debería estar haciendo esto", "mi familia se está preocupando",
                "me da asco mirarme al espejo", "siento que todo pasa porque yo lo he jodido",
                "me merezco sufrir"
            ],
            "Ira y frustración": [
                "vaya cabrón", "me voy a cagar en su puta madre", "le voy a quemar el coche",
                "puto capitalismo", "tendríamos que quemarlo todo", "fuera los moros", "feminazi",
                "cuñados", "no me entendéis", "no me estáis ayudando", "no me dejáis morirme",
                "no me dejáis tener mi vida", "no me dejáis en paz", "estoy controlado",
                "tenerme encerrado no me ayuda", "eso es una cárcel", "no podéis obligarme",
                "no podéis encerrarme", "no tenéis derecho", "no tenéis empatía",
                "me entra una rabia que me quema por dentro", "quiero pegar a alguien o romper algo",
                "me enfado hasta con mi sombra"
            ],
            "Ansiedad y agobio": [
                "me siento ahogado", "es agotador respirar", "le doy vueltas a todo lo que me pasa",
                "la única manera de relajar la ansiedad", "estoy hasta los cojones",
                "me va a explotar la cabeza", "no paro de darle vueltas", "llevo días con insomnio",
                "estoy a punto de petar", "agobio continuo", "agobiado", "sobrepasado",
                "sobrepienso", "overthinking"
            ],
            "Agotamiento emocional": [
                "aguanto hasta que peto", "me quedo sin batería social", "mi batería social se me acaba",
                "acabo petando", "lo doy todo y luego me quedo ahí sin energía", "estoy saturada",
                "estoy agotada", "estoy cansada de todo", "voy a ir súper ahogada hasta final de mes",
                "estoy cansado", "estoy cansada", "estoy harto", "estoy harta"
            ],
            "Soledad y aislamiento": [
                "me siento totalmente solo", "me siento aislado y desvalorizado", "nadie me quiere",
                "nadie me escucha", "a nadie le importa lo que digo", "nadie me entiende",
                "nadie me hace caso", "no tengo a nadie", "estoy muy desconectado"
            ],
            "Comparación y fracaso social": [
                "todos mis amigos pueden y yo no", "a todos les dejan y a mí no", "todos van y yo no"
            ]
        },
        "Rasgos cognitivos": {
            "Pensamientos intrusivos": [
                "tengo pensamientos intrusivos", "me vienen pensamientos intrusivos", "me viene la idea intrusiva"
            ],
            "Autoexigencia": [
                "tengo que ser feliz y no lo consigo", "si estoy mal, tengo que ser la que está peor",
                "yo quiero estar bien"
            ]
        },
        "Autovaloración o Autoevaluación": {
            "Autoestima": [
                "soy inútil", "me siento desvalorizado", "no soy capaz de dormir", "se me da mal todo",
                "no soy suficientemente válido", "somos débiles", "no merezco esto",
                "me odio tanto que no me aguanto", "me siento un fracasado total", "no valgo ni para ser escuchado"
            ],
            "Autodesprecio": [
                "yo a mi familia la hago feliz", "yo quiero que me quieran y si hago esto no me van a querer",
                "soy una carga para todos", "sería mejor que no estuviera aquí", "mi familia estaría mejor sin mí"
            ]
        }
    },
    "Conductuales": {
        "Pensamientos y conductas autolesivas": {
            "Ideación suicida": [
                "quiero morir", "dejadme morir", "ayudadme a morir", "no me dejáis morirme",
                "me voy a pegar un tiro", "me quiero ir de aquí", "deseo desaparecer",
                "quiero dejar de sentir", "quiero parar", "tirar la toalla", "no quiero seguir así",
                "no quiero vivir", "déjame", "ya me mato"
            ],
            "Autolesión no suicida": [
                "me corto", "me merezco hacerme daño", "me castigo", "necesito autolesionarme",
                "me calma", "me relaja", "lo he vuelto a hacer", "cuando estás mal, pues te autolesionas",
                "tengo ganas de cortarme", "cada vez que tenga ganas de cortarme, os escribiré",
                "me acabo de cortar", "me acabo de autolesionar", "me ha autolesionado",
                "me voy a rajar las venas", "me corté con una navaja en el brazo",
                "me hice unos rasguños hasta sangrar", "me pellizco hasta dejarme moratones"
            ]
        },
        "Estilos de vida": {
            "Sueño": {
                "Insomnio": [
                    "tengo insomnio", "no puedo dormir", "no he dormido en toda la noche",
                    "no he dormido", "llevo días sin dormir", "no duermo nada",
                    "me despierto muchas veces", "duermo y no descanso", "no soy capaz de dormir",
                    "a las cuatro de la mañana todavía no había pegado ni ojo", "no descanso aunque duerma",
                    "estoy durmiendo fatal", "no he dormido nada",
                    "llevo días sin dormir bien, duermo pero no descanso",
                    "me paso la noche dando vueltas en la cama", "me despierto a las 3 y ya no puedo volver a dormir"
                ]
            },
            "Dieta": {
                "Pérdida de apetito": [
                    "no se me ocurre comer", "no pienso ni en la comida",
                    "no tengo ganas de comer, todo me sabe a caspa", "llevo dos días solo con un café"
                ]
            }
        },
        "Habilidades de afrontamiento": {
            "Evitativas": [
                "cancelar o mover el día", "no me comunico mucho por whatsapp",
                "uso stickers y emojis para no sonar seria"
            ],
            "Buscar ayuda": [
                "¿tienes un momento esta tarde?", "necesito veros esta tarde",
                "les compartí la ubicación", "le escribí a mi terapeuta en plan urgente"
            ]
        },
        "Conductas de desregulación conductual": {
            "Impulsividad": [
                "me lanzo sin pensar, luego me arrepiento", "cuando me pongo así, hago lo primero que se me ocurre"
            ]
        }
    },
    "Clínicos": {
        "Trastorno mental": {
            "Trastorno depresivo mayor": [
                "depresión mayor", "estoy depre", "estoy deprimido", "tengo un brote", "estoy en crisis",
                "han visto que había depresión", "la depresión era la causa de su intento de suicidio",
                "me diagnosticaron depresión hace 10 años", "estaba tan baja que no me levantaba de la cama",
                "no quería ver a nadie, ni hablar con nadie"
            ],
            "Trastorno bipolar": [
                "es que tengo altibajos", "soy bipolar"
            ],
            "Trastorno de ansiedad generalizada": [
                "tengo ataques de ansiedad que me dejan paralizado",
                "me entra un miedo que no sé de dónde viene", "me sudan las manos y me tiembla el cuerpo"
            ],
            "TDAH": [
                "soy TDAH", "tengo TDAH de patrón"
            ],
            "Asperger/Autismo": [
                "soy asperger", "soy autista"
            ],
            "Neurodivergencia": [
                "soy neurodivergente", "tengo voces", "escucho voces", "voces que me dicen que me haga daño"
            ]
        },
        "Pensamientos y conductas autolesivas": {
            "Ideación suicida activa": [
                "tengo las pastillas en el armario y necesito tomármelas",
                "he pensado en saltar desde el balcón",
                "me he despedido de mi madre y le he regalado las plantas de mi abuela"
            ],
            "Ideación suicida pasiva": [
                "ojalá no despertarme mañana", "me da igual si me atropella un coche",
                "no tengo razones para seguir"
            ]
        }
    },
    "Victimización por violencia interpersonal": {
        "Acoso escolar (Bullying)": [
            "me tiene manía", "me hacen bullying", "he sufrido bullying desde que tengo 2 años",
            "me están haciendo bullying", "que le hacen bullying", "me hacen el vacío",
            "me llaman moro", "me llaman panchito", "no me aceptan",
            "siempre me decían gordo y me empujaban en el patio",
            "me metían en el armario y me dejaban encerrado", "me decían que no valía para nada"
        ],
        "Violencia sexual": [
            "abusó de mí", "me han violado", "abusaron de mí", "no me creyeron",
            "me pasó algo", "me hicieron algo", "ocurrió algo", "no hicieron nada", "no van a hacer nada"
        ],
        "Violencia de género": [
            "un señor horrible le tocó el culo a ella sin querer",
            "la abusaban sexualmente una vez por semana durante 4 años",
            "mi pareja me controlaba hasta lo que me ponía",
            "me decía que si me iba, me mataba"
        ]
    },
    "Familiares": {
        "Dinámica de las relaciones familiares": {
            "Falta de apoyo": [
                "mis padres estarán mejor sin mí", "podrán descansar cuando yo me muera",
                "mi madre no me conoce", "no me ven", "no me entienden", "no me hacen caso",
                "no me escuchan", "los padres no se enteran", "los padres se pasan del problema",
                "mi padre solo nos chilla", "si lo dije a mi madre me gritó",
                "no puedo hablar con mis amigos porque los agobio",
                "los padres de mis amigos se ríen de mí", "mis padres no me entienden",
                "no tengo forma alguna de huir de aquí", "estoy como atrapado"
            ],
            "Miedo a los padres": [
                "no es posible confiar en ellos", "no puedo hablar con nadie",
                "una vez se lo conté a mi madre, pero ya no se lo he contado más"
            ],
            "Conflictivas o tóxicas": [
                "mi madre es muy controladora, no me deja respirar",
                "mi hermana es tóxica, siempre me hace sentir mal",
                "en mi casa no se habla, se grita"
            ],
            "Control parental": [
                "me controlan mucho", "me están tratando como un niño", "no hay privacidad",
                "me tiene atada", "no me deja vivir"
            ],
            "Comparación con hermanos": [
                "es que a mi hermano tal", "es que a mi hermana tal"
            ],
            "Apoyo familiar": [
                "mi pareja es mi persona de confianza, con ella hablo de todo",
                "con mis abuelos me siento seguro"
            ]
        }
    },
    "Compañeros o amigos": {
        "Apoyo social": [
            "gracias por estar conmigo", "sois los únicos que me escucháis",
            "sois los únicos que me entendéis", "suerte que estéis aquí",
            "una amiga me dijo 'estoy rayadísima'", "me dijeron 'estás habituada a estar cansada'"
        ],
        "Conflictos": [
            "me dijo que me dejó porque no estaba bien",
            "se da cuenta de que le gusta a su mejor amigo"
        ],
        "Satisfacción con los compañeros o amigos": [
            "necesito hablar de esta situación",
            "me escribieron por privado para preguntarme si estaba bien"
        ]
    },
    "Académicos y contexto escolar": {
        "Fracaso escolar": [
            "suspendo todo", "no me interesa", "¿para qué?", "no sirve", "no se me da bien", "no me entero"
        ],
        "Absentismo emocional": [
            "mañana no voy al cole", "si no me encuentro bien, no voy",
            "no puedo levantarme", "no puedo ir"
        ],
        "Rendimiento académico": [
            "acabo de hacer un examen", "estar convencidos de no tener el reconocimiento de los padres"
        ],
        "Presión académica": [
            "esto se me viene grande", "no me voy a poder lidiar con la situación"
        ]
    },
    "Señales de alerta": {
        "Deseos de morir": [
            "quiero morir", "dejadme morir", "ayudadme a morir", "no me dejáis morirme",
            "no quiero vivir", "déjame", "ya me mato", "no quiero seguir así",
            "me quiero morir", "mejor un puente que la vida misma", "nos bajamos de la vida",
            "me voy a pegar un tiro", "me quiero ir de aquí", "deseo desaparecer",
            "quiero parar", "tirar la toalla"
        ],
        "Desesperanza absoluta": [
            "no hay solución", "no veo esperanza", "mi vida no va a cambiar", "no voy a llegar"
        ],
        "Autolesión como única vía": [
            "es lo único con lo que siento algo", "es lo único que me calma"
        ],
        "Despedirse o dar señales": [
            "gracias por todo lo que me dices", "para mí no hay solución",
            "yo ahora no la quiero, ser sufrir", "le he escrito una carta a mi madre diciéndole que la quiero",
            "he bajado cosas del desván para dejarlo todo preparado"
        ],
        "Regalar objetos importantes": [
            "le he dado las plantas de mi abuela a mi única amiga", "he regalado mi móvil y mi ordenador"
        ],
        "Aislamiento": [
            "no tengo a nadie", "no me comunico mucho por whatsapp", "me aislo cuando estoy mal",
            "no conozco a nadie", "no hago vida en mi pueblo", "estoy muy desconectado"
        ],
        "Cambios en la comunicación digital": [
            "uso menos emojis cuando estoy mal", "contesto más seco",
            "dejo de responder", "mi amiga puso solo 'sí' y supe que algo pasaba"
        ],
        "Cambios en hábitos": [
            "llevo nueve días vomitando, no aguanto nada", "no me baño desde hace una semana",
            "me he aislado de todos, ni contesto el teléfono"
        ]
    },
    "Sociales o comunitarios": {
        "Estigma": [
            "la sociedad no me quiere", "todo el mundo es happy, todo el mundo es normal",
            "se piensan que somos débiles", "se piensan que no somos suficientemente válidos"
        ],
        "Discriminación": [
            "los padres de mis amigos se ríen de mí", "no les importa",
            "estamos hartos de la inmigración", "estamos hartos de la delincuencia",
            "sensación de inseguridad", "reemplazo demográfico"
        ],
        "Aislamiento social": [
            "no hago vida en mi pueblo", "no conozco a nadie de donde vivo"
        ]
    },
    "Laborales o profesionales": {
        "Sobrecarga laboral": [
            "estoy explotada", "estoy quemado", "tengo mucha carga", "no llego a todo", "es demasiado"
        ],
        "Precariedad laboral": [
            "vamos a vivir en precariedad toda nuestra vida", "no se nos están dando oportunidades"
        ]
    },
    "Ambientales": {
        "Crisis climática": [
            "el mundo va a acabar siendo quemado", "estamos en la mierda", "es una farsa"
        ]
    },
    "Comunicación digital": {
        "Formas de expresar malestar": [
            "F", "LOL", "xd", "XD", "bua", "uff", "pf", "joder qué mierda",
            "red flag", "green flag"
        ]
    },
    "Individuales": {
        "Salud reproductiva": {
            "Síntomas menstruales": [
                "me baja la regla y me siento como una mierda", "estoy más baja que un tapón cuando me viene",
                "me duele el ovario izquierdo y me pongo muy irritable"
            ],
            "Embarazo no deseado": [
                "me pilló de sorpresa, no lo quería", "no sé si estar embarazada, pero me da mucho miedo",
                "me hice un test y salió positivo, pero no estoy preparada"
            ]
        }
    },
    "Uso de servicios sanitarios": {
        "Estar en psicoterapia": [
            "le escribí a mi terapeuta en plan urgente", "le dije que necesito hablar de esto en la próxima sesión"
        ]
    }
}

def recursive_flatten(d, parent_key='', sep='_'):
    """Aplana recursivamente el diccionario anidado para obtener clusters de frases."""
    items = {}
    for k, v in d.items():
        # Limpiar clave para que sea legible
        clean_k = "".join(x for x in k if x.isalnum() or x == " ")
        new_key = f"{parent_key}{sep}{clean_k}" if parent_key else clean_k
        if isinstance(v, dict):
            items.update(recursive_flatten(v, new_key, sep=sep))
        elif isinstance(v, list):
            items[new_key] = v
    return items

# ==============================================================================
# 2. MOTOR DE BIOINFORMÁTICA (Feature Extraction) - OPTIMIZADO
# ==============================================================================
class BioMarkerExtractor:
    def __init__(self, ontology_dict):
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device=DEVICE)
        self.clusters = recursive_flatten(ontology_dict)
        self.cluster_embeddings = {}
        
        print(f"🧪 Codificando {len(self.clusters)} clusters taxonómicos...")
        start_time = time.time()
        for name, phrases in tqdm(self.clusters.items(), desc="Encoding Knowledge"):
            if phrases:  # Verificar que no esté vacío
                self.cluster_embeddings[name] = self.model.encode(phrases, convert_to_tensor=True, show_progress_bar=False)
        elapsed = time.time() - start_time
        print(f"⏱️  Codificación completada en {elapsed:.2f} segundos")

    def compute_risk_matrix(self, texts, batch_size=64):
        """Optimized version with reduced memory footprint and faster computation"""
        all_features = []
        print("🔬 Extrayendo perfiles de riesgo del texto...")
        start_time = time.time()
        
        # Pre-compute cluster list to avoid repeated dictionary operations
        cluster_items = [(name, emb) for name, emb in self.cluster_embeddings.items() if emb.shape[0] > 0]
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Computing Similarity"):
            batch_texts = texts[i:i+batch_size]
            text_emb = self.model.encode(batch_texts, convert_to_tensor=True, show_progress_bar=False)
            
            batch_vectors = []
            for name, cluster_emb in cluster_items:
                # Compute similarity scores
                cos_scores = util.cos_sim(text_emb, cluster_emb)
                # Only compute max similarity (most relevant feature)
                max_sim, _ = torch.max(cos_scores, dim=1)
                batch_vectors.append(max_sim.unsqueeze(1))
            
            if batch_vectors:
                feature_tensor = torch.cat(batch_vectors, dim=1)
                all_features.append(feature_tensor.cpu())
                # Clear GPU memory
                del text_emb, feature_tensor
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                all_features.append(torch.zeros(len(batch_texts), 0))
        
        result = torch.cat(all_features, dim=0).numpy()
        elapsed = time.time() - start_time
        print(f"⏱️  Extracción completada en {elapsed:.2f} segundos")
        return result

# ==============================================================================
# 3. MODELO MULTIMODAL - OPTIMIZADO
# ==============================================================================
class MultimodalSuicideNet(nn.Module):
    def __init__(self, n_bio_features, n_classes=2):
        super(MultimodalSuicideNet, self).__init__()
        
        self.roberta = XLMRobertaModel.from_pretrained('xlm-roberta-base')
        # Freeze more layers for faster training
        for layer in self.roberta.encoder.layer[:10]:
            for param in layer.parameters():
                param.requires_grad = False
                
        self.bio_dense = nn.Sequential(
            nn.Linear(n_bio_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(768 + 128, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, n_classes)
        )

    def forward(self, input_ids, attention_mask, bio_features):
        roberta_out = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        text_vec = roberta_out.last_hidden_state[:, 0, :]
        bio_vec = self.bio_dense(bio_features)
        combined = torch.cat((text_vec, bio_vec), dim=1)
        logits = self.classifier(combined)
        return logits

# ==============================================================================
# 4. PIPELINE DE ENTRENAMIENTO Y EVALUACIÓN - OPTIMIZADO
# ==============================================================================
def evaluate_model(model, dataloader, device):
    """Calcula métricas F2 y Accuracy para un dataloader dado"""
    model.eval()
    preds_all, labels_all = [], []
    
    with torch.no_grad():
        for batch in dataloader:
            b_ids, b_mask, b_bio, b_labels = [t.to(device) for t in batch]
            logits = model(b_ids, b_mask, b_bio)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            preds_all.extend(preds)
            labels_all.extend(b_labels.cpu().numpy())
            
    f2 = fbeta_score(labels_all, preds_all, beta=2, zero_division=0)
    acc = accuracy_score(labels_all, preds_all)
    return f2, acc, labels_all, preds_all

def main_pipeline(df):
    print(f"📊 Procesando {len(df)} registros...")
    overall_start = time.time()
    
    # 1. Limpieza
    print("🧹 Limpiando datos...")
    df['clean_text'] = df['body_anonimizado'].fillna("").astype(str).apply(lambda x: x[:1200])  # Truncate
    texts = df['clean_text'].tolist()
    labels = df['suicidi_ideacion_label'].astype(int).values
    
    # 2. Bio Markers
    extractor = BioMarkerExtractor(NEW_KNOWLEDGE_CATEGORIES)
    bio_features = extractor.compute_risk_matrix(texts, batch_size=128)  # Larger batch size
    print(f"✅ Matriz de Biomarcadores: {bio_features.shape}")
    
    # Clear memory
    del extractor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 3. Tokenización
    print("🔤 Tokenizando textos...")
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=256, return_tensors='pt')
    
    # 4. Split Train/Test
    idxs = np.arange(len(labels))
    train_idx, test_idx = train_test_split(idxs, test_size=0.2, stratify=labels, random_state=42)
    
    print(f"📈 Train samples: {len(train_idx)}, Test samples: {len(test_idx)}")
    
    input_ids = encodings['input_ids']
    masks = encodings['attention_mask']
    bio_tensor = torch.tensor(bio_features, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    
    train_data = TensorDataset(input_ids[train_idx], masks[train_idx], bio_tensor[train_idx], labels_tensor[train_idx])
    test_data = TensorDataset(input_ids[test_idx], masks[test_idx], bio_tensor[test_idx], labels_tensor[test_idx])
    
    # Larger batch size for faster training
    train_loader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=32)
    train_eval_loader = DataLoader(train_data, sampler=SequentialSampler(train_data), batch_size=32) 
    test_loader = DataLoader(test_data, sampler=SequentialSampler(test_data), batch_size=32)
    
    # 5. Modelo
    print("🏗️  Construyendo modelo...")
    model = MultimodalSuicideNet(n_bio_features=bio_features.shape[1]).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=2e-5)
    loss_fn = nn.CrossEntropyLoss()
    
    # 6. Entrenamiento Loop
    epochs = 3  # Reduced from 4 for faster execution
    print(f"\n🚀 Entrenando por {epochs} épocas...")
    training_start = time.time()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        epoch_start = time.time()
        for batch in tqdm(train_loader, desc=f"Ep {epoch+1}/{epochs}", leave=False):
            b_ids, b_mask, b_bio, b_labels = [t.to(DEVICE) for t in batch]
            model.zero_grad()
            logits = model(b_ids, b_mask, b_bio)
            loss = loss_fn(logits, b_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / len(train_loader)
        print(f"✅ Época {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Tiempo: {epoch_time:.2f}s")
            
    training_time = time.time() - training_start
    print(f"\n✅ Entrenamiento finalizado en {training_time:.2f} segundos.")
    
    # 7. Evaluación de Overfitting
    print("\n" + "="*60)
    print("RESULTADOS DE EVALUACIÓN (OVERFITTING CHECK)")
    print("="*60)
    
    # Train Metrics
    f2_train, acc_train, _, _ = evaluate_model(model, train_eval_loader, DEVICE)
    # Test Metrics
    f2_test, acc_test, y_true, y_pred = evaluate_model(model, test_loader, DEVICE)
    
    print(f"TRAIN SET -> F2-Score: {f2_train:.4f} | Accuracy: {acc_train:.4f}")
    print(f"TEST SET  -> F2-Score: {f2_test:.4f} | Accuracy: {acc_test:.4f}")
    
    diff_f2 = f2_train - f2_test
    print("-" * 60)
    if diff_f2 > 0.15:
        print(f"⚠️ ALERTA DE OVERFITTING: La diferencia de F2 es alta ({diff_f2:.2f}).")
        print("Sugerencia: Aumenta Dropout, reduce capacidad del modelo o usa más regularización.")
    else:
        print(f"✅ MODELO ESTABLE: La diferencia de F2 es aceptable ({diff_f2:.2f}).")
    
    print("\nReporte Detallado (Test Set):")
    print(classification_report(y_true, y_pred, target_names=['No Riesgo', 'Riesgo Suicida']))
    
    total_time = time.time() - overall_start
    print(f"\n⏱️  TIEMPO TOTAL DE EJECUCIÓN: {total_time:.2f} segundos ({total_time/60:.2f} minutos)")

# Ejecución
if __name__ == "__main__":
    if 'merged_df1' in globals():
        main_pipeline(merged_df1)
    else:
        print("❌ Error: merged_df1 no está definido.")
        print("💡 Para probar el código, crea un DataFrame de ejemplo:")
        print("   import pandas as pd")
        print("   merged_df1 = pd.DataFrame({")
        print("       'body_anonimizado': ['texto de ejemplo'] * 100,")
        print("       'suicidi_ideacion_label': [0, 1] * 50")
        print("   })")
        print("   main_pipeline(merged_df1)")
