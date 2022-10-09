
pub trait Api {
    
    fn base_api() -> String;

    fn spot_api() -> String; 

    fn future_api() -> String;
}

pub struct BinanceApi {}

impl Api for BinanceApi {
    fn base_api() -> String {
        String::from("https://api.binance.com")
    }

    fn spot_api() -> String {
        String::from("")
    }

    fn future_api() -> String {
        String::from("")
    }
}

