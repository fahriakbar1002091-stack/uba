# database.py - FULL VERSION DENGAN KONFIGURASI LENGKAP
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os

load_dotenv()

# DATABASE CONFIGURATION
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql://postgres:280908@localhost:5432/social_radar"
)

# Engine dengan konfigurasi performa
engine = create_engine(
    DATABASE_URL,
    pool_size=10,              # Jumlah koneksi maksimal
    max_overflow=20,           # Koneksi tambahan jika penuh
    pool_timeout=30,           # Timeout koneksi
    pool_recycle=1800,         # Recycle koneksi setiap 30 menit
    echo=False                 # Set True untuk debug SQL
)

# Session Factory
SessionLocal = sessionmaker(
    autocommit=False, 
    autoflush=False, 
    bind=engine
)

Base = declarative_base()

# Dependency Injection untuk FastAPI
def get_db():
    """Generator untuk database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Fungsi utilitas database
def init_db():
    """Inisialisasi database (create tables)"""
    from models import Base  # Import di sini untuk hindari circular import
    Base.metadata.create_all(bind=engine)
    print("✅ Database initialized successfully!")

def check_connection():
    """Cek koneksi database"""
    try:
        with engine.connect() as connection:
            print("✅ Database connection successful!")
            return True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False

# Optional: Fungsi untuk reset database (development only)
def reset_database():
    """Reset semua tabel (hanya untuk development)"""
    from models import Base
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    print("✅ Database has been reset!")

# Test connection saat import
if __name__ == "__main__":
    check_connection()