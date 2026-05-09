import asyncio
import csv
import os
import random
import time
from datetime import datetime, timedelta

from playwright.async_api import async_playwright

# --- CONFIGURACIÓN DE RUTA AL ESCRITORIO ---
ESCRITORIO = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
ARCHIVO_CSV = os.path.join(ESCRITORIO, "datos APPL.csv")

# --- CONFIGURACIÓN DE BÚSQUEDA ---
BUSQUEDA = "AAPL;Apple"  
OBJETIVO_TWEETS = 500          
DIAS_TOTALES = 7               
PERFIL_DIR = os.path.join(os.getcwd(), "perfil_X_DesktopApp")

async def ejecutar_captura():
    async with async_playwright() as p:
        try:
            context = await p.chromium.launch_persistent_context(
                user_data_dir=PERFIL_DIR,
                channel="chrome", 
                headless=False,
                args=[
                    f"--app=https://x.com/search?q={BUSQUEDA}&f=live",
                    "--window-size=1280,720",
                    "--disable-blink-features=AutomationControlled"
                ]
            )
            
            page = context.pages[0] if context.pages else await context.new_page()
            await page.wait_for_timeout(5000)

            # Verificación de Login
            if await page.query_selector('input[name="text"]'):
                print(">>> ESPERANDO LOGIN MANUAL...")
                await page.wait_for_selector('article[data-testid="tweet"]', timeout=0)

            tweets_capturados = []
            ids_vistos = set()
            intentos_scroll = 0

            while len(tweets_capturados) < OBJETIVO_TWEETS and intentos_scroll < 60:
                articulos = await page.query_selector_all('article[data-testid="tweet"]')
                
                for art in articulos:
                    try:
                        enlace = await art.query_selector('a[href*="/status/"]')
                        url_t = await enlace.get_attribute('href')
                        t_id = url_t.split('/')[-1]

                        if t_id not in ids_vistos:
                            # 1. Fecha
                            time_el = await art.query_selector('time')
                            fecha = await time_el.get_attribute('datetime')
                            
                            # 2. Texto
                            texto_el = await art.query_selector('div[data-testid="tweetText"]')
                            texto = " ".join((await texto_el.inner_text()).split()) 

                            # 3. Retweets (Compartidos)
                            # Buscamos el div que contiene el aria-label de retweets
                            rt_el = await art.query_selector('button[data-testid="retweet"]')
                            rt_text = await rt_el.get_attribute('aria-label')
                            # Extraemos solo el número del texto (ej: "5 retweets" -> "5")
                            retweets = rt_text.split()[0] if rt_text else "0"

                            # 4. Likes (Favoritos)
                            fav_el = await art.query_selector('button[data-testid="like"]')
                            fav_text = await fav_el.get_attribute('aria-label')
                            favoritos = fav_text.split()[0] if fav_text else "0"

                            tweets_capturados.append([t_id, fecha, texto, retweets, favoritos])
                            ids_vistos.add(t_id)
                            
                            if len(tweets_capturados) >= OBJETIVO_TWEETS:
                                break
                    except Exception as e:
                        print(f"Error al procesar un tweet: {e}")
                        continue

                await page.mouse.wheel(0, random.randint(1200, 2000))
                await page.wait_for_timeout(random.randint(2500, 4000))
                intentos_scroll += 1

            # Guardado con las nuevas columnas
            es_nuevo = not os.path.exists(ARCHIVO_CSV)
            with open(ARCHIVO_CSV, 'a', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f, quoting=csv.QUOTE_ALL)
                if es_nuevo:
                    writer.writerow(['ID_Tweet', 'Fecha_UTC', 'Contenido_Texto', 'Retweets', 'Favoritos'])
                writer.writerows(tweets_capturados)
            
            print(f">>> ÉXITO: {len(tweets_capturados)} tweets con métricas guardados.")
            await context.close()

        except Exception as e:
            print(f">>> ERROR: {e}")

def bucle_principal():
    fecha_fin = datetime.now() + timedelta(days=DIAS_TOTALES)
    while datetime.now() < fecha_fin:
        asyncio.run(ejecutar_captura())
        espera = 3600 + random.randint(0, 300)
        print("Dormido hasta la siguiente hora...")
        time.sleep(espera)

if __name__ == "__main__":
    bucle_principal()