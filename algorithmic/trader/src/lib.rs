pub use market::{BinanceClient, EndPoint, get_market_data};

use reqwest::{Client, Response};

use dotenv;

pub struct Trader;