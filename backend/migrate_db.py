"""
Migration script to add new columns to existing database.
Run this to update the database schema after model changes.
"""
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "kanoonsathi.db"

def migrate():
    """Add new columns to case_laws table and create ingestion_logs table."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        # Check if columns already exist
        cursor.execute("PRAGMA table_info(case_laws)")
        columns = [row[1] for row in cursor.fetchall()]
        
        # Add new columns to case_laws if they don't exist
        new_columns = [
            ("ingestion_status", "VARCHAR(50) DEFAULT 'pending'"),
            ("ingestion_error", "TEXT"),
        ]
        
        for col_name, col_type in new_columns:
            if col_name not in columns:
                print(f"Adding column {col_name}...")
                cursor.execute(f"ALTER TABLE case_laws ADD COLUMN {col_name} {col_type}")
            else:
                print(f"Column {col_name} already exists, skipping...")
        
        # Create ingestion_logs table if it doesn't exist
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='ingestion_logs'
        """)
        if not cursor.fetchone():
            print("Creating ingestion_logs table...")
            cursor.execute("""
                CREATE TABLE ingestion_logs (
                    id INTEGER PRIMARY KEY,
                    filename VARCHAR(255) NOT NULL,
                    file_type VARCHAR(50) NOT NULL,
                    status VARCHAR(50) NOT NULL,
                    cases_processed INTEGER DEFAULT 0,
                    cases_failed INTEGER DEFAULT 0,
                    error_message TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
        else:
            print("ingestion_logs table already exists, skipping...")
        
        conn.commit()
        print("Migration completed successfully!")
    except Exception as e:
        conn.rollback()
        print(f"Migration failed: {e}")
        raise
    finally:
        conn.close()

if __name__ == "__main__":
    migrate()