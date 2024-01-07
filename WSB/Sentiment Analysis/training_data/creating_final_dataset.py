import config
import psycopg2
import psycopg2.extras
import pandas as pd

connection = psycopg2.connect(host=config.DB_HOST, database=config.DB_NAME, user=config.DB_USER,
                              password=config.DB_PASS)
cursor = connection.cursor(cursor_factory=psycopg2.extras.DictCursor)

cursor.execute("""
    SELECT * 
    FROM mention
""")
file_csv = cursor.fetchall()
file_csv = pd.DataFrame(file_csv, columns=('stock_id', 'stock_symbol', 'dt', 'title', 
                                           'body', 'flair', 'post_id', 'score', 'upvote_ratio', 
                                           'num_comments', 'url', 'sentiment', 'is_train')).to_csv('wsb_dataset.csv', index=False)
