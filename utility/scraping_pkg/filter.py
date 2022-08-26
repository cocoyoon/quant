
from unittest.util import sorted_list_difference

from h11 import Data
from data_manager import Data_Manager
from scraping import Scraper
from utils import Utils
import re

FILTERD_INFO = ['시가총액', '유동비율']
TRANSACTION_AMOUNT = 3e10
RETURN_UPPER_LIMIT = 29
TAIL_PRICE_LIMIT = 20

def filter(data) -> list:

    """
        Args
        - data: DataFrame of scrpaed data.

        Return
        - filtered_ticker: List of filtered ticker.

        Filtering Condition
            - 시총: 3000억 이상 제외. 
            - 3년 순이익률: 3년 순이익률이 모두 마이너스면 제외.
            - 거래 대금: 300억 미만 제외.
            - 상한가 여부: 1년중 상한가가 없으면 제외.
            - 긴꼬리 양봉 여부: 1년중 긴꼬리 양봉 업으면 제외.
    """

    filtered_ticker = []
    for index in range(len(data)):

        # market_cap
        market_cap = int(re.sub('[^\d]',"",data.loc[index]['시가총액']))
        if market_cap > 3000:
            continue
        
        # 유동비율
        current_ratio = int(re.sub('[^\d]',"",data.loc[index]['유동비율']))
        if current_ratio > 7000: 
            continue
            
        # 3년 순이익률
        profit_ratio_3years = data.loc[index]['3년 순이익률']
        try:
            profit_ratio = float(sorted(profit_ratio_3years)[-1])
            if  profit_ratio < 0.0:
                continue
        except Exception as e:
            print(f"You have error!. {e}")
        
        # 거래대금
        ticker = data.loc[index]['티커']
        tx_amount = Data_Manager.get_transaction_amount(ticker=ticker)
        if sorted(tx_amount)[-1] < TRANSACTION_AMOUNT:
            continue

        # 상한가 여부
        return_pct_change = Data_Manager.get_return_pct_change(ticker=ticker)
        if return_pct_change[-1] < RETURN_UPPER_LIMIT:
            continue

        # 긴꼬리 양봉 여부
        tail_change = Data_Manager.get_tail_change(ticker=ticker)
        if tail_change[-1] < TAIL_PRICE_LIMIT:
            continue

        print(f"Ticker {ticker}:{Data_Manager.get_ticker_name(ticker)}")
        filtered_ticker.append(ticker)  

    return filtered_ticker    


if __name__ == "__main__":

    TRADE_INFO = ['거래량/거래대금', '시가총액', '발행주식수/유동비율']
    scraper = Scraper(trade_info=TRADE_INFO)
    df = scraper.collect(limit=10, market="KOSPI")
    filtered_data = filter(data=df)
    

