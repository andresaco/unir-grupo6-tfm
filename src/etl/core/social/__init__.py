from typing import Protocol, List, Dict, Any, TypedDict


class SocialPost(TypedDict, total=False):
    id: str
    source_url: str
    author: str
    published_time: str
    text: str
    reposts: str
    likes: str
    raw: Dict[str, Any]


class SocialSearchClient(Protocol):
    platform_name: str

    async def search_posts(self, query: str, max_results: int) -> List[SocialPost]: ...
