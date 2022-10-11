mod client;
mod endpoint;

/// Binance Module 
/// 1. Market Data
/// 2. 
use client::BinanceClient;
use endpoint::{EndPoint, MarketData};

pub mod binance {

    use super::*;

    pub async fn get_market_data() -> color_eyre::Result<()> {
        let client = BinanceClient::new(None, None);
        let base = String::from(EndPoint::Base);
        let end_point = String::from(EndPoint::MarketData(MarketData::ExhchangeInfo));
        let request_url = format!("{}{}", base, end_point.as_str());
        client.get(String::from(request_url), None).await?;

        Ok(())
    }
}

#[cfg(test)]
mod tests{
    use crate::binance::*;

    #[tokio::test]
    async fn get_market_data_works() -> color_eyre::Result<()> {
        let res = get_market_data().await?;
        assert_eq!((), res);
        Ok(())
    }
}