
from bs4 import BeautifulSoup as bs
from bs4 import NavigableString, Tag
from h11 import Data
from selenium import webdriver
from dotenv import load_dotenv
from data_manager import Data_Manager
from pathlib import Path
import pandas as pd 
import os

load_dotenv()
DRIVER_DIR = Path.joinpath(Path.cwd(),'chromedriver')

class Scraper:

    def __init__(self, trade_info):

        self.market = None
        self.df = None
        self.trade_info = trade_info
        self.id = None 

    def collect(self, limit=None, market=None):
        
        """
            Args   
            - market
            - url

            Return
            - DataFrame
        """

        if self.market != market:
            print(f"Market: {market}")
            self.df = pd.DataFrame()    
        self.market = market

        driver = webdriver.Chrome(DRIVER_DIR)
        tickers = Data_Manager.get_ticker(market=market)[:limit] # for test
        for ticker in tickers:
            url = 'http://companyinfo.stock.naver.com/company/c1010001.aspx?cmp_cd={ticker}'.format(ticker=ticker)
            driver.get(url=url)
            html = driver.page_source
            new_data = self.parse_html(
                source=html, 
                ticker=ticker
            )
            self.df = pd.concat([self.df, new_data], ignore_index=True)
            
        driver.close()
        return self.df

    def parse_html(self, source, ticker):

        (df, items) = pd.DataFrame(), {}
        items['티커'] = ticker
        parser = bs(source, 'lxml')
        trade_table = parser.find('table', {'class': 'gHead'})
        scrape_data = trade_table.find_all('tr')
        for data in scrape_data:

            table_title = data.find('th', {'class': 'txt'}).text
            if table_title in self.trade_info:
                if table_title == '거래량/거래대금':
                    items['거래대금'] = data.find('td', {'class': 'num'}).text.split()[2].strip()
                elif table_title == '발행주식수/유동비율':
                    items['유동비율'] = data.find('td', {'class': 'num'}).text.split()[2].strip()
                else:
                    items[table_title] = data.find('td', {'class': 'num'}).text.strip()
        
        id_finder = parser.find('table', {'id': 'cTB00'})
        self.id = id_finder.find_next('div')['id']
        balance_table = parser.find('div', {'id': self.id})
        balance_table = balance_table.find('table', {'class': 'gHead01 all-width'})
        scrape_data = balance_table.find('tbody')
        
        for data in scrape_data.children:

            if isinstance(data, NavigableString):
                continue
            elif isinstance(data, Tag):
                
                profit_ratio = []
                if data.find('th', {'class': 'bg txt'}) != None:
                    balance_type = data.find('th', {'class': 'bg txt'}).text
                    if balance_type == '순이익률':
                        profit_3_years = data.find_all('td', {'class': 'num'})[1:4]
                        for profit in profit_3_years:
                            if profit.text == '':
                                continue
                            else:
                                profit_ratio.append(profit.text)

                        items['3년 순이익률'] = profit_ratio
        
        df = pd.DataFrame.from_dict(items, orient='index')

        return df.T

if __name__ == "__main__":

    scraper = Scraper(trade_info=['거래량/거래대금', '시가총액', '발행주식수/유동비율'])
    scraper.collect(market='KOSPI', ticker=['005930','035420', '035720'])

        