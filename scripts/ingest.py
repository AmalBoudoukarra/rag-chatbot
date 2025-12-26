#!/usr/bin/env python3
"""
Standalone script for ingesting documents.
Run this script to populate the vector database.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.ingestion import get_ingestion_service
from app.utils.logger import setup_logging, get_logger

# Setup logging
setup_logging()
logger = get_logger(__name__)


def main():
    """Main ingestion function."""
    try:
        logger.info("=" * 60)
        logger.info("Starting Document Ingestion")
        logger.info("=" * 60)
        
        ingestion_service = get_ingestion_service()
        result = ingestion_service.ingest_all_documents()
        
        logger.info("=" * 60)
        logger.info("Ingestion Complete")
        logger.info("=" * 60)
        logger.info("Results", **result)
        
        return 0
        
    except Exception as e:
        logger.error("Ingestion failed", error=str(e))
        return 1


if __name__ == "__main__":
    sys.exit(main())