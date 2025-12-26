"""
Main FastAPI application.
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.utils.logger import setup_logging, get_logger
from app.core.vector_store import get_vectorstore_manager
from app.api.routes import health, qa

# Setup logging
setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Handles startup and shutdown events.
    """
    # Startup
    logger.info(
        "Starting application",
        name=settings.APP_NAME,
        version=settings.APP_VERSION
    )
    
    # Initialize vector store on startup
    try:
        vectorstore_manager = get_vectorstore_manager()
        vectorstore_manager.initialize()
        logger.info("Vector store initialized on startup")
    except Exception as e:
        logger.warning(
            "Failed to initialize vector store on startup",
            error=str(e),
            message="This is normal if ingestion hasn't been run yet"
        )
    
    yield
    
    # Shutdown
    logger.info("Shutting down application")


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="RAG-based Question Answering API",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router)
app.include_router(qa.router)


@app.get("/ping")
async def ping():
    """Simple ping endpoint for Docker health checks."""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )