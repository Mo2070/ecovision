from models import init_db

if __name__ == "__main__":
    print("Creating SQLite database...")
    init_db()
    print("Database created successfully: ecovision.db")
