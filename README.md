# BERTherapy “Proyecto de Deep Learning | NLP | Conversational AI”

BERTherapy es un proyecto estudiantil de IA conversacional enfocado en crear asistentes (chatbots) tipo terapeuta y paciente, entrenados para interactuar y dar apoyo emocional en conversaciones simuladas. Utiliza modelos de lenguaje (BERT) adaptados y entrenados en un dataset propio, con variables como emoción y sentimiento, para generar respuestas empáticas y adecuadas al contexto.

Para entrenar los modelos se utiliza el dataset de Hugging Face ego02/mental-health-chatbot-training
---

## ¿Cómo montar el proyecto?

### 1. Clona el repositorio y crea un entorno virtual

```bash
git clone <url-del-repo>
cd BERTherapy
python3 -m venv venv
source venv/bin/activate    # En Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Obtención de los modelos entrenados

- **Modelos entrenados (archivos .safetensors)**  
  Por defecto los modelos entrenados NO están en el repositorio, ya que son muy pesados.  
  Si los necesitas, puedes solicitarlos aparte, o generarlos tú mismo entrenando desde cero (ver pasos siguientes).

---

## 3. Preparar los datos

Antes de entrenar, debes procesar los datos:

```bash
python scripts/prepare_dataset.py
```

Esto genera los archivos CSV procesados en `data/processed/` y los pools de respuestas.  
Verifica que los datos se hayan creado correctamente antes de continuar, es un proceso lento, se puede usar el 
archivo de prepare_dataset_sample.py para generar los datasets_sample y pools_sample de ejemplo y revisar las salidas, aunque 
estos archivos ya están incluidos en el repositorio

Tip: el archivo prepare_dataset está actualmente limitado a 5 bloques, 5k experiencias.
  - Total muestras terapeuta csv full: 71.392
  - Total muestras paciente csv full: 76.364
---

## 4. Entrenar los modelos

Lanza el entrenamiento ejecutando:

```bash
python models/training_models.py
```

Esto entrenará dos modelos: uno para el "terapeuta" y otro para el "paciente", y los guardará en la carpeta `models/`.  
> **Importante:**  
> - La versión que se está entrenando se configura en el diccionario `config` en `models/training_models.py` (`version`).  
> - Cuando más adelante quieras usar un modelo diferente, asegúrate de actualizar el número de versión tanto en el código de *Patient* como en *Therapist* para que cargue el modelo correcto.

---

## 5. Ejecutar la conversación

Después de entrenar (o de tener los modelos safetensors), puedes probar la conversación ejecutando el archivo principal:

```bash
python main.py
```

(También puedes personalizar el main para integrarlo en otras aplicaciones).

---

## Notas y recomendaciones

- Si sólo quieres los modelos entrenados, pídelo explícitamente; no se suben por defecto.
- El entrenamiento puede tardar bastante en CPU, se recomienda usar GPU si es posible.
- Los pools de respuestas y los csv se regeneran cada vez que corres el script de preparación de datos.
- El proyecto está en proceso de mejora y optimización, pero cualquier ayuda se agradece;
  ¡Aporta mejoras, testea y comparte si te interesa!

---

## Créditos y agradecimientos

- El dataset principal utilizado para el entrenamiento de los modelos es **"Mental Health Chatbot Training"** creado por [ego02](https://huggingface.co/datasets/ego02/mental-health-chatbot-training) y disponible públicamente en Hugging Face Datasets.
- Agradezco el trabajo y la generosidad de la comunidad open source, especialmente de ego02, por compartir este recurso.
