"""Main FastAPI Application Entrypoint."""
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import settings
from .api.routes import root_router, api_router, voice_router
from .utils.logger import setup_logger, get_logger

# Initialize application logger
setup_logger(
    name="insurance_rag",
    level=10 if settings.DEBUG else 20,
)
logger = get_logger("insurance_rag.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown procedures."""
    logger.info("=" * 60)
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"ChromaDB Persist Dir: {settings.CHROMA_PERSIST_DIRECTORY}")
    logger.info(f"ChromaDB Collection: {settings.CHROMA_COLLECTION_NAME}")
    logger.info(f"Knowledge Base Dir: {settings.DOCUMENT_DIRECTORY}")
    logger.info(f"Embedding Model: {settings.EMBEDDING_MODEL}")
    logger.info(f"Groq LLM Model: {settings.GROQ_MODEL}")
    logger.info(f"Groq API Key Configured: {'Yes' if settings.has_groq_key else 'No (set GROQ_API_KEY in .env)'}")
    logger.info(f"Allowed CORS Origin: {settings.FRONTEND_URL}")
    logger.info(f"Voice Input Enabled: {settings.ENABLE_VOICE_INPUT}")
    if settings.ENABLE_VOICE_INPUT:
        logger.info(f"Whisper Model: {settings.WHISPER_MODEL} ({settings.WHISPER_DEVICE}, {settings.WHISPER_COMPUTE_TYPE})")
        try:
            from .services.speech_service import get_speech_service
            get_speech_service().load_model()
        except Exception as e:
            logger.warning(f"Whisper model warm-up deferred: {str(e)}")
    logger.info("=" * 60)
    yield
    logger.info(f"Shutting down {settings.APP_NAME}")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Production-grade Retrieval-Augmented Generation (RAG) API for Insurance Q&A",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS Configuration
# Allow frontend URL from settings as well as standard local development variations
allowed_origins = list(
    filter(
        None,
        {
            settings.FRONTEND_URL.rstrip("/"),
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:3000",
            "http://localhost:8000",
        },
    )
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Fallback handler to prevent unhandled stack trace exposure."""
    logger.error(f"Unhandled exception on {request.method} {request.url}: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred. Please check server logs."},
    )


# Attach routers
app.include_router(root_router)
app.include_router(api_router)
app.include_router(voice_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
