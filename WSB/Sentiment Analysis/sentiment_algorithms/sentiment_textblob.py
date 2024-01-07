from textblob import TextBlob
import config
import psycopg2
import psycopg2.extras

connection = psycopg2.connect(host = config.DB_HOST, database = config.DB_NAME, user = config.DB_USER, password = config.DB_PASS)
cursor = connection.cursor(cursor_factory = psycopg2.extras.DictCursor)

cursor.execute(
    '''SELECT title, body FROM mention'''
)
rows = cursor.fetchall()

for row in rows:
    text = row['title'] + row['body']
    blob = TextBlob(text)
    
    # if the post has more sentences, it gets split
    for sentence in blob.sentences:
      
        #every gets a sentiment score from -1 (negative) to 1 (positive)
        sentiment = sentence.sentiment.polarity
        print(sentence)
        print(sentiment)
        
        # seeing the results, this method clearly doesn't work. 
        # Textblob is not fit at all!
