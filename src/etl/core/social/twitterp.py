from __future__ import annotations

import random
from pathlib import Path
from typing import List

from playwright.async_api import async_playwright

from . import SocialPost, SocialSearchClient


class TwitterClient(SocialSearchClient):
    """Cliente de búsqueda social para X.com / Twitter.

    Implementa un contrato genérico para que pueda reemplazarse por otros
    clientes como Bluesky, Threads u otras APIs sociales.
    """

    platform_name = "twitter"

    def __init__(
        self, profile_dir: str = ".playwright_x_profile", headless: bool = False
    ):
        self.profile_dir = Path(profile_dir).expanduser().resolve()
        self.headless = headless
        self.profile_dir.mkdir(parents=True, exist_ok=True)

    async def search_posts(self, query: str, max_results: int) -> List[SocialPost]:
        return await self._extract_from_x(query, max_results)

    async def _extract_from_x(self, query: str, max_results: int) -> List[SocialPost]:
        results: List[SocialPost] = []
        ids_seen: set[str] = set()
        search_url = f"https://x.com/search?q={query}&f=live"

        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch_persistent_context(
                user_data_dir=str(self.profile_dir),
                headless=self.headless,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                ],
            )

            page = browser.pages[0] if browser.pages else await browser.new_page()
            await page.goto(search_url)
            await page.wait_for_timeout(4000)

            if await page.query_selector('input[name="text"]'):
                print(
                    ">>> Se ha detectado un login. Completa el inicio de sesión en X para continuar."
                )
                await page.wait_for_selector('article[data-testid="tweet"]', timeout=0)

            scroll_attempts = 0
            while len(results) < max_results and scroll_attempts < 60:
                articles = await page.query_selector_all('article[data-testid="tweet"]')

                for article in articles:
                    try:
                        post = await self._parse_article(article)
                        if not post or post["id"] in ids_seen:
                            continue

                        results.append(post)
                        ids_seen.add(post["id"])
                        if len(results) >= max_results:
                            break
                    except Exception as exc:
                        print(f"Error extrayendo tweet: {exc}")
                        continue

                await page.mouse.wheel(0, random.randint(1200, 2000))
                await page.wait_for_timeout(random.randint(2500, 4500))
                scroll_attempts += 1

            await browser.close()

        return results

    async def _parse_article(self, article) -> SocialPost | None:
        tweet_link = await article.query_selector('a[href*="/status/"]')
        if not tweet_link:
            return None

        href = await tweet_link.get_attribute("href")
        if not href:
            return None

        tweet_id = href.split("/")[-1]
        time_el = await article.query_selector("time")
        published_time = await time_el.get_attribute("datetime") if time_el else ""

        text_el = await article.query_selector('div[data-testid="tweetText"]')
        if not text_el:
            text_el = await article.query_selector("div[lang]")

        text = " ".join((await text_el.inner_text()).split()) if text_el else ""

        retweet_button = await article.query_selector('button[data-testid="retweet"]')
        retweet_label = (
            await retweet_button.get_attribute("aria-label") if retweet_button else "0"
        )
        reposts = retweet_label.split()[0] if retweet_label else "0"

        like_button = await article.query_selector('button[data-testid="like"]')
        like_label = (
            await like_button.get_attribute("aria-label") if like_button else "0"
        )
        likes = like_label.split()[0] if like_label else "0"

        return {
            "id": tweet_id,
            "source_url": f"https://x.com{href}",
            "published_time": published_time,
            "text": text,
            "reposts": reposts,
            "likes": likes,
            "raw": {},
        }
