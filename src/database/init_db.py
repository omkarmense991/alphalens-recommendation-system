import src.database.models
from src.database.session import Base, engine
from src.utils.logger import logger


def init_db():
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created successfully")


if __name__ == "__main__":
    init_db()
