import config
import psycopg2
import psycopg2.extras
import random
import pandas as pd

connection = psycopg2.connect(host=config.DB_HOST, database=config.DB_NAME, user=config.DB_USER,
                              password=config.DB_PASS)
cursor = connection.cursor(cursor_factory=psycopg2.extras.DictCursor)

# we take all 231 samples from march 2020 crash (to train enough bearish posts), 
# 500 from january 2021
# and 500 from june 2021

cursor.execute("""
    SELECT stock_symbol, post_id, dt, title, body, flair, upvote_ratio 
    FROM mention
    WHERE dt::date BETWEEN '2020-02-13' AND '2020-03-27'
""")
sample1 = cursor.fetchall()
sample1 = pd.DataFrame(sample1).to_csv('/sample1.csv')

cursor.execute("""
    SELECT stock_symbol, post_id, dt, title, body, flair, upvote_ratio 
    FROM mention
    WHERE dt::date BETWEEN '2021-05-11' AND '2021-06-10'
""")
sample2 = cursor.fetchall()
random_sample2 = random.sample(sample2, 500)
random_sample2 = pd.DataFrame(random_sample2).to_csv('/sample2.csv')

cursor.execute("""
    SELECT stock_symbol, post_id, dt, title, body, flair, upvote_ratio 
    FROM mention
    WHERE dt::date BETWEEN '2021-01-04' AND '2021-02-03'
""")
sample3 = cursor.fetchall()
random_sample3 = random.sample(sample3, 500)
random_sample3 = pd.DataFrame(random_sample3).to_csv('/sample3.csv')
