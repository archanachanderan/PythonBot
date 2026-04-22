import psycopg2

conn = psycopg2.connect(
    host="db.rdsukgjfqdobiltdbasn.supabase.co",
    database="postgres",
    user="postgres",
    password="Pythonbot@21042026",
    port=5432
)

cursor = conn.cursor()