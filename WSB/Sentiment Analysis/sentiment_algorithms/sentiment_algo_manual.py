import config
import psycopg2
import psycopg2.extras
import functools
import operator
import re
import emoji

connection = psycopg2.connect(host = config.DB_HOST, database = config.DB_NAME, user = config.DB_USER, password = config.DB_PASS)
cursor = connection.cursor(cursor_factory = psycopg2.extras.DictCursor)

cursor.execute(
    '''SELECT stock_id, stock_symbol, title, body, score, num_comments FROM mention'''
)
rows = cursor.fetchall()

def WSB_sentiment (title, body, score, num_comments):
    
    bullish = ['moon', 'mars', 'yolo', 'all', 'strong', "can't", 'tits', 'c', 'calls', 'call', 'btfd', 'undervalued', 'gains', 'gain', 'bull', 'bulls', 'bullish', 
               'buy', 'dip', 'fuel', 'fire', 'squeeze', 'squeezing', 'squoze', 'squozing', 'holding', 'bought', 'buying', 'hold', 'hodl', 'hodling', 'holding', 
               'yoloed', "yolo'ed", 'mooning',
               '💎', '🤲', '🤤', '🌝', '🙌🏻', '🙌', '👐', '✋', '🦍', '🚀', '🌙', '📈', '🌚'
              ]
    
    bearish = ['p', 'puts', 'put', 'losses', 'lose', 'loss', 'bear', 'bears', 'bearish', 'gay', 'sold', 'sell', 'selling',
               '🌈', '🐻', '😬'
              ]
    
    bullish_count = 0
    bearish_count = 0
    magnitude = score + num_comments
    
    text = title + body
    text_lowercase = text.lower()
    text_split_emoji = emoji.get_emoji_regexp().split(text_lowercase)
    text_split_whitespace = [substr.split() for substr in text_split_emoji]
    text_split = functools.reduce(operator.concat, text_split_whitespace)
    
    for word in text_split:
        if word in bullish:
            bullish_count += 1
        elif word in bearish:
            bearish_count += 1
    print(bullish_count)
    direction = bullish_count - bearish_count
        
    if direction > 1:
        return magnitude
    elif direction < 0:
        return magnitude * (-1)
    
for row in rows:
    result = WSB_sentiment(row['title'], row['body'], row['score'], row['num_comments'])
    post_id = row['post_id']
    stock_symbol = row['stock_symbol']

    try:
        cursor.execute('''
        UPDATE mention
        SET sentiment = %s
        WHERE post_id = %s AND stock_symbol = %s
        ''', (result, post_id, stock_symbol))
        connection.commit()

    except Exception as e:
        print(e)
        connection.rollback()
