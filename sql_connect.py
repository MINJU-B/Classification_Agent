import psycopg2

def get_connection(
    *,
    db_dsn=None,
    host=None,
    port=5432,
    database=None,
    user=None,
    password=None,
):
    if db_dsn:
        return psycopg2.connect(db_dsn)
    if not all([host, database, user, password]):
        raise RuntimeError("db_dsn 또는 host/database/user/password가 필요합니다")
    return psycopg2.connect(
        host=host,
        port=port,
        database=database,
        user=user,
        password=password,
    )
