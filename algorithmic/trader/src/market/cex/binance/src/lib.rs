pub mod client;
pub mod endpoint;

/// Binance Module 
/// 1. Market Data
/// 2. 
pub use client::BinanceClient;
pub use endpoint::{EndPoint, MarketData};
pub use binance::*;

pub mod binance {

    use super::*;

    pub async fn get_market_data() -> color_eyre::Result<()> {
        let client = BinanceClient::new(None, None);
        let base = String::from(EndPoint::Base);
        let end_point = String::from(EndPoint::MarketData(MarketData::ExhchangeInfo));
        let request_url = format!("{}{}", base, end_point.as_str());
        println!("Requesting for -> {:?}", request_url);
        client.get(String::from(request_url), None).await?;

        Ok(())
    }
}
