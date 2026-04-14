"""Database connection management via SQLAlchemy."""

from typing import Generator

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.config.settings import DatabaseSettings


class DatabaseConnection:
    """Manages SQLAlchemy engine and session factory."""

    def __init__(self, settings: DatabaseSettings) -> None:
        self._settings = settings
        self._engine: Engine = create_engine(
            settings.connection_url,
            pool_pre_ping=True,
        )
        self._session_factory: sessionmaker[Session] = sessionmaker(
            bind=self._engine,
        )

    def get_session(self) -> Generator[Session, None, None]:
        """Yield a database session, ensuring proper cleanup."""
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def get_connection_url(self) -> str:
        """Return the connection URL string."""
        return self._settings.connection_url

    @property
    def engine(self) -> Engine:
        """Access the underlying SQLAlchemy engine."""
        return self._engine
