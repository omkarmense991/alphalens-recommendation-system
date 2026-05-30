import pandas as pd

from src.config.settings import RAW_DATA_DIR


def create_nifty50_universe():
    assets = [
        {
            "symbol": "RELIANCE.NS",
            "company_name": "Reliance Industries",
            "sector": "Energy",
            "industry": "Oil & Gas",
        },
        {
            "symbol": "TCS.NS",
            "company_name": "Tata Consultancy Services",
            "sector": "Information Technology",
            "industry": "IT Services",
        },
        {
            "symbol": "INFY.NS",
            "company_name": "Infosys",
            "sector": "Information Technology",
            "industry": "IT Services",
        },
        {
            "symbol": "HDFCBANK.NS",
            "company_name": "HDFC Bank",
            "sector": "Financial Services",
            "industry": "Private Bank",
        },
        {
            "symbol": "ICICIBANK.NS",
            "company_name": "ICICI Bank",
            "sector": "Financial Services",
            "industry": "Private Bank",
        },
        {
            "symbol": "SBIN.NS",
            "company_name": "State Bank of India",
            "sector": "Financial Services",
            "industry": "Public Bank",
        },
        {
            "symbol": "ITC.NS",
            "company_name": "ITC",
            "sector": "FMCG",
            "industry": "Consumer Goods",
        },
        {
            "symbol": "HINDUNILVR.NS",
            "company_name": "Hindustan Unilever",
            "sector": "FMCG",
            "industry": "Consumer Goods",
        },
        {
            "symbol": "LT.NS",
            "company_name": "Larsen & Toubro",
            "sector": "Construction",
            "industry": "Engineering",
        },
        {
            "symbol": "BHARTIARTL.NS",
            "company_name": "Bharti Airtel",
            "sector": "Telecommunication",
            "industry": "Telecom Services",
        },
        {
            "symbol": "ONGC.NS",
            "company_name": "Oil and Natural Gas Corporation",
            "sector": "Energy",
            "industry": "Oil & Gas",
        },
        {
            "symbol": "IOC.NS",
            "company_name": "Indian Oil Corporation",
            "sector": "Energy",
            "industry": "Oil & Gas",
        },
        {
            "symbol": "BPCL.NS",
            "company_name": "Bharat Petroleum Corporation",
            "sector": "Energy",
            "industry": "Oil & Gas",
        },
        {
            "symbol": "HINDPETRO.NS",
            "company_name": "Hindustan Petroleum Corporation",
            "sector": "Energy",
            "industry": "Oil & Gas",
        },
        {
            "symbol": "GAIL.NS",
            "company_name": "GAIL India",
            "sector": "Energy",
            "industry": "Gas Utilities",
        },
    ]

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(assets)
    output_path = RAW_DATA_DIR / "asset_universe.csv"
    df.to_csv(output_path, index=False)

    print(f"Asset universe saved to {output_path}")


if __name__ == "__main__":
    create_nifty50_universe()
