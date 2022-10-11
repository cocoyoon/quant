use reqwest::Client;

/// Binance Client
/// 
/// 1. GET Method
/// 2. POST Method
/// 3. Signing order
pub struct BinanceClient {
    pub public: Option<String>,
    pub secret: Option<String>,
    pub inner_client: Client
}

impl BinanceClient {

    pub fn new(public: Option<String>, secret: Option<String>) -> Self {
        Self {
            public,
            secret,
            inner_client: Client::new()
        }
    }

    pub async fn get(&self, request_url: String, params: Option<String>) -> color_eyre::Result<()> {

        let mut url = request_url.clone();
        if let Some(params) = params {
            url = format!("{}{}", url, params); 
        }
        println!("Requesting for -> {}", url);
        let res = self.inner_client.get(url).send().await?;
        println!("{:?}", res);

        Ok(())
    }
}