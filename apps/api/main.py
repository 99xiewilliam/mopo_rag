from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from apps.api.routers.rag import router as rag_router
from apps.api.routers.health import router as health_router
from rag.utils.logging import setup_logging


def create_app() -> FastAPI:
    setup_logging()
    app = FastAPI(title="MOPO-RAG API", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_router)
    app.include_router(rag_router, prefix="/v1")
    return app


app = create_app()


