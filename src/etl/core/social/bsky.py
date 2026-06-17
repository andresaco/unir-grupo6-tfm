from __future__ import annotations
import os
from typing import Dict, List
from atproto import Client
from . import SocialSearchClient


class BlueskyClient(SocialSearchClient):
    def __init__(self, handle: str | None = None, password: str | None = None):
        """
        Cliente para Bluesky utilizando AT Protocol.
        Lee las credenciales de las variables de entorno si no se proveen.
        """
        self.handle = handle or os.environ.get("BLUESKY_HANDLE")
        self.password = password or os.environ.get("BLUESKY_PASSWORD")

        if not self.handle or not self.password:
            raise ValueError(
                "Se requieren las credenciales de Bluesky (BLUESKY_HANDLE y BLUESKY_PASSWORD)"
            )

        self.client = Client()
        self.client.login(self.handle, self.password)

    async def search_posts(
        self, query: str, target_tweets: int
    ) -> List[Dict[str, any]]:
        """
        Busca publicaciones en Bluesky y las devuelve normalizadas.
        El contenido del texto se limpia para garantizar que sea un oneliner.
        """
        posts_normalized: List[Dict[str, any]] = []
        cursor = None
        remaining = target_tweets
        base_params = {"q": query, "limit": min(target_tweets, 100), "sort": "latest"}
        while remaining > 0:
            response = self.client.app.bsky.feed.search_posts(
                params={**base_params, "cursor": cursor if cursor else None}
            )

            if not response.posts:
                break

            for post in response.posts:
                raw_text = getattr(post.record, "text", "")

                # --- MODIFICACIÓN PARA ONELINER ---
                # 1. Reemplazamos retornos de carro y tabulaciones por espacios comunes
                # 2. Dividimos por cualquier espacio en blanco y volvemos a unir con un solo espacio
                # Esto elimina saltos de línea (\n, \r) y remueve espacios duplicados indeseados
                text_oneliner = " ".join(raw_text.replace("\r", " ").split())
                # ----------------------------------

                posts_normalized.append(
                    {
                        "id": post.uri,
                        "published_time": getattr(post.record, "created_at", ""),
                        "text": text_oneliner,  # Enviamos el texto ya aplanado
                        "reposts": str(getattr(post, "repost_count", 0)),
                        "likes": str(getattr(post, "like_count", 0)),
                    }
                )

            cursor = response.cursor
            remaining -= len(response.posts)

            if not cursor:
                break

        return posts_normalized[:target_tweets]
