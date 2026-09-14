"""Fetch topic-related images from Wikipedia and build explore links."""

import logging
from urllib.parse import quote_plus

import requests


def fetch_topic_images(topic: str, max_images: int = 2) -> list:
    """Search Wikipedia for *topic* and return thumbnail URLs.

    Returns a list of dicts: ``{"url", "title", "caption", "width", "height"}``.
    Falls back to an empty list on any error.
    """
    if not topic:
        return []

    try:
        # Step 1 — find relevant Wikipedia pages.
        search_resp = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": topic,
                "srnamespace": "0",
                "srlimit": str(max_images + 3),
                "format": "json",
            },
            timeout=5,
        )
        results = search_resp.json().get("query", {}).get("search", [])
        if not results:
            return []

        # Step 2 — grab thumbnails for those pages.
        titles = "|".join(r["title"] for r in results[: max_images + 3])
        img_resp = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "titles": titles,
                "prop": "pageimages|pageterms",
                "piprop": "thumbnail",
                "pithumbsize": "480",
                "format": "json",
            },
            timeout=5,
        )
        pages = img_resp.json().get("query", {}).get("pages", {})

        images = []
        for page_id, page in pages.items():
            if page_id == "-1":
                continue
            thumb = page.get("thumbnail", {})
            if not thumb.get("source"):
                continue

            title = page.get("title", "")
            terms = page.get("terms", {})
            desc = terms["description"][0] if terms.get("description") else title

            images.append({
                "url": thumb["source"],
                "title": title,
                "caption": desc,
                "width": thumb.get("width", 480),
                "height": thumb.get("height", 360),
            })
            if len(images) >= max_images:
                break

        return images

    except Exception as exc:
        logging.warning("[IMAGE-FETCH] Failed: %s", exc)
        return []


def build_explore_links(topic: str) -> list:
    """Build a list of explore-more links for *topic*."""
    if not topic:
        return []

    enc = quote_plus(topic)
    return [
        {"label": "Google",    "icon": "🔍", "url": f"https://www.google.com/search?q={enc}"},
        {"label": "Wikipedia", "icon": "📖", "url": f"https://en.wikipedia.org/wiki/Special:Search/{enc}"},
        {"label": "News",      "icon": "📰", "url": f"https://news.google.com/search?q={enc}"},
        {"label": "Images",    "icon": "🖼️", "url": f"https://www.google.com/search?q={enc}&tbm=isch"},
    ]
