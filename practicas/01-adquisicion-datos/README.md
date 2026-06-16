# Adquisición de datos

- script main
- script partition

## main.py: Capturador Automatizado de Tweets (X)

Este script (main.py) utiliza la librería **Playwright** para automatizar la extracción de tweets y métricas (retweets y likes) desde X (Twitter). Está diseñado para ejecutarse de forma persistente durante varios días, recolectando datos periódicamente.

### **Características**

- **Automatización de Navegador**: Utiliza Chromium (Google Chrome) para navegar e interactuar con la plataforma.  
- **Métricas Incluidas**: Captura el ID del tweet, la fecha UTC, el texto completo, el número de Retweets y el número de Favoritos (Likes).  
- **Persistencia de Sesión**: Guarda el perfil de usuario en la carpeta perfil_X_DesktopApp para evitar tener que iniciar sesión en cada ejecución.  
- **Evasión de Detección**: Incluye argumentos para deshabilitar características de automatización y utiliza scrolls aleatorios para imitar el comportamiento humano.  
- **Ejecución Programada**: El script se ejecuta en un bucle cada hora aproximadamente durante el periodo de días definido.

### Requisitos

1. **Python 3.8+**  
2. **Librerías necesarias**:  
   pip install playwright
3. **Instalación de Navegadores**:  
   playwright install chrome

### **Configuración**

Dentro del archivo main.py, puedes ajustar las siguientes constantes:

- BUSQUEDA: El término o etiquetas a buscar (actualmente "AAPL;Apple").  
- OBJETIVO_TWEETS: Cuántos tweets intentar capturar en cada ráfaga (500).  
- DIAS_TOTALES: Cuántos días permanecerá activo el script (7 días).  
- ARCHIVO_CSV: La ruta de salida (por defecto, guarda "datos APPL.csv" en tu Escritorio).

### **Funcionamiento**

1. **Inicio**: Al ejecutar el script, se abrirá una ventana de Chrome.  
2. **Login**: Si no has iniciado sesión previamente, el script se detendrá y esperará a que realices el login manualmente. Una vez detecte la interfaz de tweets, continuará solo.  
3. **Scroll y Captura**: El script hará scroll hacia abajo automáticamente, detectando nuevos tweets y extrayendo sus datos.  
4. **Guardado**: Los datos se añaden (append) al archivo CSV en el escritorio. Si el archivo no existe, crea las cabeceras.  
5. **Espera**: Tras completar una captura, el script "duerme" durante una hora antes de iniciar la siguiente ráfaga.

### **Estructura del CSV**

El archivo generado contiene las siguientes columnas:

- ID_Tweet: Identificador único (formato texto).  
- Fecha_UTC: Fecha y hora original de publicación.  
- Contenido_Texto: Texto limpio del tweet.  
- Retweets: Cantidad de compartidos detectados.  
- Favoritos: Cantidad de likes detectados.

**Nota**: El uso de este script debe cumplir con los términos de servicio de la plataforma y se recomienda su uso para fines de investigación o análisis de datos personales.

## partition.py

Este script de Python está diseñado para procesar archivos CSV que contienen datos de Twitter y dividirlos en múltiples archivos basados en la fecha de publicación (Fecha_UTC). Es ideal para organizar grandes datasets en segmentos manejables por día.

### Requisitos

Para ejecutar este script, necesitas tener instalado Python y la librería **pandas**:

```bash
pip install pandas
```

### **Características Principales**

- **Preservación de IDs**: Configurado para leer la columna ID_Tweet como texto (string), evitando que identificadores largos se corrompan o se conviertan a notación científica.  
- **Manejo de Fechas ISO 8601**: Procesa automáticamente formatos de fecha con zonas horarias (ej. 2026-04-13T20:56:45.000Z).  
- **Salida Organizada**: Crea automáticamente una carpeta en el mismo directorio que el archivo original.  
- **Evita Sobrescritura**: Utiliza un *timestamp* (marca de tiempo) en el nombre de la carpeta de salida para permitir múltiples ejecuciones sin perder datos previos.  
- **Limpieza Automática**: Detecta y omite registros con fechas inválidas o vacías.

### **Uso**

El script se ejecuta desde la terminal pasando la ruta del archivo CSV como argumento:

python partition_tweets.py ruta/hacia/tu_fichero.csv

### **Estructura de Salida**

Si procesas un archivo llamado mis_tweets.csv, el script generará la siguiente estructura:

```bash
/directorio_del_csv/  
├── mis_tweets.csv  
└── mis_tweets_partition_20260509_163000/  
    ├── tweets_2026-04-13.csv  
    ├── tweets_2026-04-14.csv  
    └── ...
```

### **Argumentos**

| Argumento | Descripción |
| :---- | :---- |
| fichero | (Obligatorio) Ruta al archivo CSV que se desea particionar. |

### **Detalles Técnicos**

El script utiliza pandas.groupby para segmentar los datos de forma eficiente, lo que garantiza un rendimiento óptimo incluso con archivos de gran tamaño. Antes de exportar, elimina columnas auxiliares de cálculo para mantener la integridad del esquema original de tus datos.
