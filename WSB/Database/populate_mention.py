import config
import datetime as dt
from psaw import PushshiftAPI     # Pushshift is a database which contains all Reddit's posts
import praw                       # praw is the Reddit API, and it is useful to know the score and the number of comments of a post
import psycopg2
import psycopg2.extras

connection = psycopg2.connect(host = config.DB_HOST, database = config.DB_NAME, user = config.DB_USER, password = config.DB_PASS)
cursor = connection.cursor(cursor_factory = psycopg2.extras.DictCursor)
reddit = praw.Reddit(client_id =config.R_KEY, client_secret =config.R_SECRET, username =config.R_USERNAME, password =config.R_PASSWORD, user_agent = 'https://github.com/3m4c')
api = PushshiftAPI()


# creating two similar dictionaries where the keys are the stocks ids from the stock table
# and the values are the tickers with and without the $ sign (e.g. $GME)
# the third dictionary is the inverse of the second one: we will need it to access the ticker from the id
cursor.execute(
    '''SELECT * FROM stock'''
)
rows = cursor.fetchall()
stock_dict1 = {}
stock_dict2 = {}
stock_dict3 = {}
for row in rows:
    symbol = row['symbol']
    stock_dict1['$' + symbol] = row['id']
    stock_dict2[symbol] = row['id']
    stock_dict3[row['id']] = symbol

# scraping posts from start_time to end_time (we can choose any interval we want)
start_time = int(dt.datetime(2021, 1, 1).timestamp())
endtime = int(dt.datetime(2021, 6, 20).timestamp())
submissions = api.search_submissions(after = start_time,
                                         before = endtime,
                                         subreddit = 'wallstreetbets',
                                         filter = ['url', 'author', 'title', 'body', 'flair', 'subreddit', 'id']
                                          )

# iterating through the submissions (posts)
for submission in submissions:

    # splitting the title into words: any ticker gets temporarily stored in the tickers list
    words = submission.title.split()
    tickers = list(set(filter(lambda word: word.lower().startswith('$') or word in stock_dict2, words)))

    # many sumbmissions do not have a ticker, therefore we can discard them
    if len(tickers) > 0:

        # iterating through the tickers, if there are more than one in a single post
        for ticker in tickers:
            
            # we use Pushshift to extract the submission id, then we use Reddit to extract the score of that submission
            submission_id = submission.id
            sub_praw = reddit.submission(submission_id)
            score = sub_praw.score
            
            # we only need the posts with greater visibility
            if score > 100:
                
                #we need to make sure that the ticker is a real one
                if ticker in stock_dict1 or ticker in stock_dict2:
                
                    # finally, we can fetch all the data we need
                    submitted_time = dt.datetime.fromtimestamp(submission.created_utc).isoformat()
                    num_comments = sub_praw.num_comments
                    
                    #the ticker might be in the first or in the second dictionary, depending on the initial $
                    try:
                        stock_id = stock_dict1[ticker]
                    except:
                        stock_id = stock_dict2[ticker]
                        
                    stock_symbol = stock_dict3[stock_id]
                    body = sub_praw.selftext
                    flair = sub_praw.link_flair_text
                    upvote_ratio = sub_praw.upvote_ratio

                    # inserting the data into the 'mention' table
                    try:
                        cursor.execute('''
                        INSERT INTO mention (dt, stock_id, stock_symbol, title, body, flair, url, post_id, score, upvote_ratio, num_comments)
                        VAlUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ''', (
                        submitted_time, stock_id, stock_symbol, submission.title, body, flair,
                        submission.url, submission_id, score, upvote_ratio, num_comments))
                        connection.commit()

                    except Exception as e:
                        print(e)
                        connection.rollback()
