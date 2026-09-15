from app.database.connection import Base, engine
from app.database import models


def main() -> None:
    Base.metadata.create_all(bind=engine)
    print("MAI database tables created successfully.")


if __name__ == "__main__":
    main()