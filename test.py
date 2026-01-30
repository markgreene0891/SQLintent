import sqlite3
conn = sqlite3.connect("test.db")
cursor = conn.cursor()
cursor.execute(" SELECT SUM('')")