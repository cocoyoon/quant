
from scraping import Scraper
from bot import Telgram_Bot
from filter import filter


K_MARKETS = ['KOSPI','KOSDAQ']
TRADE_INFO = ['거래량/거래대금', '시가총액', '발행주식수/유동비율']

def main():

    scraper = Scraper(trade_info=TRADE_INFO)
    bot = Telgram_Bot()
    filtered_ticker = {}
    for market in K_MARKETS:

        df = scraper.collect(limit=300, market=market)
        filtered_ticker[market] = filter(data=df)
    
    bot.send_message(data=filtered_ticker)

if __name__ == "__main__":

    main()