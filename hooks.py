from cat import hook, CheshireCat

from .main import WeaviateHandler


@hook(priority=0)
async def after_cat_bootstrap(cat: CheshireCat) -> None:
    if isinstance(cat.vector_memory_handler, WeaviateHandler):
        await cat.vector_memory_handler.connect()  # type: ignore[union-attr]
