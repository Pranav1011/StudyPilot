#!/usr/bin/env python3
"""
Database Setup Script

Initialize database, run migrations, and seed test data.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent))

from backend.core.database import (
    initialize_database, 
    close_database, 
    get_database_manager,
    DatabaseSeeder,
    DatabaseBackup
)
from backend.config import get_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def setup_database():
    """Setup database with migrations and test data."""
    try:
        # Get settings
        settings = get_settings()
        database_url = settings.database_url
        
        logger.info("Initializing database...")
        await initialize_database(database_url)
        
        db_manager = get_database_manager()
        
        # Health check
        health = await db_manager.health_check()
        logger.info(f"Database health: {health}")
        
        # Create tables
        logger.info("Creating database tables...")
        await db_manager.create_tables()
        
        # Run Alembic migrations
        logger.info("Running database migrations...")
        await run_migrations()
        
        # Seed test data if requested
        if os.getenv("SEED_TEST_DATA", "false").lower() == "true":
            logger.info("Seeding test data...")
            seeder = DatabaseSeeder(db_manager)
            await seeder.seed_test_data()
        
        logger.info("Database setup completed successfully!")
        
    except Exception as e:
        logger.error(f"Database setup failed: {e}")
        raise
    finally:
        await close_database()


async def run_migrations():
    """Run Alembic migrations."""
    try:
        import subprocess
        
        # Run alembic upgrade
        result = subprocess.run(
            ["alembic", "upgrade", "head"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )
        
        if result.returncode != 0:
            logger.error(f"Migration failed: {result.stderr}")
            raise Exception(f"Migration failed: {result.stderr}")
        
        logger.info("Migrations completed successfully")
        
    except Exception as e:
        logger.error(f"Failed to run migrations: {e}")
        raise


async def create_backup():
    """Create database backup."""
    try:
        settings = get_settings()
        database_url = settings.database_url
        
        await initialize_database(database_url)
        db_manager = get_database_manager()
        
        backup = DatabaseBackup(db_manager)
        backup_path = await backup.create_backup()
        
        logger.info(f"Database backup created: {backup_path}")
        
    except Exception as e:
        logger.error(f"Backup failed: {e}")
        raise
    finally:
        await close_database()


async def restore_backup(backup_path: str):
    """Restore database from backup."""
    try:
        settings = get_settings()
        database_url = settings.database_url
        
        await initialize_database(database_url)
        db_manager = get_database_manager()
        
        backup = DatabaseBackup(db_manager)
        await backup.restore_backup(backup_path)
        
        logger.info("Database restored successfully")
        
    except Exception as e:
        logger.error(f"Restore failed: {e}")
        raise
    finally:
        await close_database()


async def clear_test_data():
    """Clear test data from database."""
    try:
        settings = get_settings()
        database_url = settings.database_url
        
        await initialize_database(database_url)
        db_manager = get_database_manager()
        
        seeder = DatabaseSeeder(db_manager)
        await seeder.clear_test_data()
        
        logger.info("Test data cleared successfully")
        
    except Exception as e:
        logger.error(f"Failed to clear test data: {e}")
        raise
    finally:
        await close_database()


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Database setup and management")
    parser.add_argument("action", choices=["setup", "backup", "restore", "clear-test"], 
                       help="Action to perform")
    parser.add_argument("--backup-path", help="Path to backup file for restore")
    
    args = parser.parse_args()
    
    if args.action == "setup":
        asyncio.run(setup_database())
    elif args.action == "backup":
        asyncio.run(create_backup())
    elif args.action == "restore":
        if not args.backup_path:
            logger.error("Backup path required for restore action")
            sys.exit(1)
        asyncio.run(restore_backup(args.backup_path))
    elif args.action == "clear-test":
        asyncio.run(clear_test_data())


if __name__ == "__main__":
    main() 