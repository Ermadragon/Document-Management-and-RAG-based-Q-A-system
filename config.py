import os

DB_USERNAME = "postgres"
DB_PASSWORD = "database_password"  
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "postgres"

SQLALCHEMY_DATABASE_URI = f"postgresql://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
SQLALCHEMY_TRACK_MODIFICATIONS = False

