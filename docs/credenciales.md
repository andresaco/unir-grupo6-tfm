# Guía de Configuración de Credenciales

Este documento detalla los pasos necesarios para obtener las credenciales de **Bluesky**, **Google BigQuery** y **Hugging Face**, y cómo configurarlas en el archivo [.env](.env) en la raíz del proyecto.

---

## 🦋 1. Bluesky API (AT Protocol)

El pipeline de ingesta social descarga posts de Bluesky utilizando el protocolo AT (AT Protocol). Para autenticarse, se requiere el identificador de usuario y una contraseña de aplicación (App Password).

### Pasos para obtener las credenciales:
1. **Crear una cuenta**: Si aún no tienes cuenta, regístrate en [bsky.app](https://bsky.app).
2. **Obtener tu Handle**: Tu handle es tu nombre de usuario completo, por ejemplo, `tu_usuario.bsky.social`.
3. **Generar una Contraseña de Aplicación (App Password)**:
   - *Nota: Por motivos de seguridad, nunca utilices tu contraseña principal de Bluesky en scripts.*
   - Inicia sesión en [bsky.app](https://bsky.app).
   - Ve a **Settings** (Configuración) ⚙️ en el menú lateral.
   - Accede a la sección **App Passwords** (Contraseñas de aplicación) o visita directamente [bsky.app/settings/app-passwords](https://bsky.app/settings/app-passwords).
   - Haz clic en **Add App Password** (Añadir contraseña de aplicación).
   - Introduce un nombre descriptivo para identificarla (por ejemplo, `ml-pipeline-tfm`).
   - Haz clic en **Create App Password**.
   - Copia la contraseña generada (un código con formato similar a `xxxx-xxxx-xxxx-xxxx`).
4. **Configuración en `.env`**:
   Abre el archivo [.env](.env) y añade o edita las siguientes líneas:
   ```env
   BLUESKY_HANDLE=tu_usuario.bsky.social
   BLUESKY_PASSWORD=xxxx-xxxx-xxxx-xxxx
   ```

---

## 🔍 2. Google BigQuery (GDELT)

El pipeline utiliza BigQuery para realizar consultas sobre la base de datos pública **GDELT** (por ejemplo, en [history.py](src/etl/history.py)). Se requiere un proyecto en Google Cloud y credenciales de autenticación.

Hay dos formas recomendadas para configurar las credenciales:

### Opción A: Mediante un archivo de clave de cuenta de servicio (JSON)
Esta opción es la mejor si ejecutas el código en un entorno donde no tienes la CLI de Google Cloud instalada o en producción.

1. **Crear un Proyecto en Google Cloud**:
   - Entra en la [Consola de Google Cloud](https://console.cloud.google.com/).
   - Crea un nuevo proyecto (por ejemplo, `unir-grupo6-tfm`) o selecciona uno existente.
   - Copia el **ID del proyecto** (Project ID).
2. **Habilitar la API de BigQuery**:
   - En la consola, busca "BigQuery API" y asegúrate de que esté habilitada para tu proyecto.
3. **Crear una Cuenta de Servicio**:
   - Ve a **IAM & Admin** (Administración e IAM) > **Service Accounts** (Cuentas de servicio).
   - Haz clic en **Create Service Account** (Crear cuenta de servicio).
   - Asigna un nombre (por ejemplo, `bigquery-loader`) y haz clic en **Create and Continue**.
   - En la sección de roles, asigna el rol de **BigQuery User** (Usuario de BigQuery) o **BigQuery Admin** para permitir la ejecución de consultas y la creación de datasets.
   - Haz clic en **Done**.
4. **Generar y Descargar la Clave JSON**:
   - Selecciona la cuenta de servicio que acabas de crear.
   - Ve a la pestaña **Keys** (Claves).
   - Haz clic en **Add Key** (Agregar clave) > **Create new key** (Crear clave nueva).
   - Selecciona el formato **JSON** y haz clic en **Create**.
   - Guarda el archivo JSON descargado en un directorio seguro de tu máquina (por ejemplo, en tu carpeta home o en un subdirectorio del proyecto ignorado por Git, como `runtime/`).
5. **Configuración en `.env`**:
   Define el ID del proyecto y la ruta absoluta al archivo JSON:
   ```env
   GOOGLE_CLOUD_PROJECT=tu-project-id-gcp
   GOOGLE_APPLICATION_CREDENTIALS=/ruta/absoluta/a/tu/archivo-credenciales.json
   ```

### Opción B: Mediante Google Cloud CLI (Autenticación Local / ADC)
Si tienes instalado `gcloud` en tu máquina local, puedes generar las credenciales de aplicación por defecto (Application Default Credentials - ADC) automáticamente:

1. Ejecuta en tu terminal:
   ```bash
   gcloud auth application-default login
   ```
2. Esto abrirá tu navegador para iniciar sesión con tu cuenta de Google. Tras confirmar, generará un archivo JSON local.
3. Configura el archivo [.env](.env) apuntando a esa ruta por defecto (en Linux suele ser `~/.config/gcloud/application_default_credentials.json`):
   ```env
   GOOGLE_CLOUD_PROJECT=tu-project-id-gcp
   GOOGLE_APPLICATION_CREDENTIALS=/home/tu_usuario/.config/gcloud/application_default_credentials.json
   ```

---

## 🤗 3. Hugging Face API

El modelo de análisis de sentimientos de las publicaciones de redes sociales (por ejemplo, `distilbert-base-uncased-finetuned-sst-2-english`) requiere un token de acceso de Hugging Face para descargar los pesos de los modelos de forma autenticada.

### Pasos para obtener las credenciales:
1. **Crear una cuenta**: Regístrate en [huggingface.co](https://huggingface.co).
2. **Generar un Access Token**:
   - Ve a tu perfil en la esquina superior derecha y selecciona **Settings** (Configuración).
   - En el menú lateral izquierdo, haz clic en **Access Tokens** (Tokens de acceso) o accede directamente a [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
   - Haz clic en **New token** (Nuevo token).
   - Asígnale un nombre (por ejemplo, `tfm-pipeline`).
   - Elige el tipo de token:
     - **Read (Lectura)**: Es suficiente para descargar modelos y datasets públicos. (Recomendado).
     - **Write (Escritura)**: Requerido si planeas subir modelos entrenados o datasets a tu perfil de Hugging Face.
   - Haz clic en **Generate a token**.
   - Copia el token generado (que comienza con `hf_`).
3. **Configuración en `.env`**:
   Abre tu archivo [.env](.env) y añade el token:
   ```env
   HF_TOKEN=hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
   ```

---

## 📝 4. Plantilla Completa de `.env`

A continuación se muestra cómo debería quedar la estructura de tu archivo [.env](.env) con todas las credenciales configuradas:

```env
# --- HUGGING FACE CONFIGURATION ---
HF_TOKEN=hf_tu_token_de_huggingface_aqui
SENTIMENT_MODEL=distilbert-base-uncased-finetuned-sst-2-english

# --- BLUESKY CONFIGURATION ---
BLUESKY_HANDLE=tu_usuario.bsky.social
BLUESKY_PASSWORD=xxxx-xxxx-xxxx-xxxx

# --- GOOGLE CLOUD & BIGQUERY CONFIGURATION ---
GOOGLE_CLOUD_PROJECT=tu-project-id-de-gcp
GOOGLE_APPLICATION_CREDENTIALS=/ruta/absoluta/a/credenciales.json
```

> [!IMPORTANT]
> Asegúrate de **no subir nunca** tu archivo `.env` o tus claves JSON al repositorio de Git. El archivo `.env` ya se encuentra configurado en el archivo [.gitignore](.gitignore) para evitar fugas de información sensible.
