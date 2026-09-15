from sqlalchemy import text

from app.database.connection import engine


def main() -> None:
    try:
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            value = result.scalar()

        print(f"Database connection successful: {value}")

    except Exception as exc:
        print(f"Database connection failed: {exc}")


if __name__ == "__main__":
    main()