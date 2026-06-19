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
        self, query: str, target_tweets: int, sort_type: str = "latest"
    ) -> List[Dict[str, any]]:
        """
        Busca publicaciones en Bluesky y las devuelve normalizadas.
        El contenido del texto se limpia para garantizar que sea un oneliner.

        Args:
            query: La cadena de búsqueda.
            target_tweets: Límite máximo de posts a recuperar.
            sort_type: Método de ordenación. Acepta "latest" (recientes) o "top" (más relevantes/importantes).
        """
        posts_normalized: List[Dict[str, any]] = []
        cursor = None
        remaining = target_tweets

        # Utilizamos el sort_type dinámico para poder buscar los más importantes ("top")
        base_params = {"q": query, "limit": min(target_tweets, 100), "sort": sort_type}

        while remaining > 0:
            response = self.client.app.bsky.feed.search_posts(
                params={**base_params, "cursor": cursor if cursor else None}
            )

            if not response.posts:
                break

            for post in response.posts:
                raw_text = getattr(post.record, "text", "")

                # --- MODIFICACIÓN PARA ONELINER ---
                # Eliminamos saltos de línea y tabulaciones
                text_oneliner = " ".join(raw_text.replace("\r", " ").split())
                # ----------------------------------

                posts_normalized.append(
                    {
                        "id": post.uri,
                        "published_time": getattr(post.record, "created_at", ""),
                        "text": text_oneliner,
                        "reposts": str(getattr(post, "repost_count", 0)),
                        "likes": str(getattr(post, "like_count", 0)),
                    }
                )

            cursor = response.cursor
            remaining -= len(response.posts)

            if not cursor:
                break

        return posts_normalized[:target_tweets]
