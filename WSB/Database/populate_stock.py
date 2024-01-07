import config                           #config contains the APIs keys and the database details, which include username and password
import alpaca_trade_api as tradeapi     #we may use any good finance API
import psycopg2                         #we need it to translate from Python to PostgreSQL
import psycopg2.extras


# creating a database session
connection = psycopg2.connect(host = config.DB_HOST, database = config.DB_NAME, user = config.DB_USER, password = config.DB_PASS)

# creating a connection with the database
cursor = connection.cursor(cursor_factory = psycopg2.extras.DictCursor) 

# Alpaca API authentication
api = tradeapi.REST(config.API_KEY, config.API_SECRET, base_url = config.API_URL)


assets = api.list_assets()
for asset in assets:
    #print(f'Inserting stock {asset.name} {asset.symbol}')
    cursor.execute(
        '''INSERT INTO stock (name, symbol, exchange, is_etf)
         VALUES (%s, %s, %s, %s)
         ''', (asset.name, asset.symbol, asset.exchange, False) #for now we don't need the is_etf column
    )

# executing the cursor.execute command onto the database
connection.commit()
