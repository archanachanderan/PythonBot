import psycopg2

def get_connection():
    return psycopg2.connect(
        host="aws-1-ap-northeast-1.pooler.supabase.com",
        database="postgres",
        user="postgres.rdsukgjfqdobiltdbasn",
        password="Pythonbot@21042026",
        port=5432
    )