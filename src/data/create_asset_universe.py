import pandas as pd

from src.config.settings import RAW_DATA_DIR
from src.utils.logger import logger


def create_nifty50_universe():
    assets = [
        # Energy / Oil & Gas
        {
            "symbol": "RELIANCE.NS",
            "company_name": "Reliance Industries",
            "sector": "Energy",
            "industry": "Oil & Gas",
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
        {
            "symbol": "COALINDIA.NS",
            "company_name": "Coal India",
            "sector": "Energy",
            "industry": "Coal",
        },
        # IT
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
            "symbol": "HCLTECH.NS",
            "company_name": "HCL Technologies",
            "sector": "Information Technology",
            "industry": "IT Services",
        },
        {
            "symbol": "WIPRO.NS",
            "company_name": "Wipro",
            "sector": "Information Technology",
            "industry": "IT Services",
        },
        {
            "symbol": "TECHM.NS",
            "company_name": "Tech Mahindra",
            "sector": "Information Technology",
            "industry": "IT Services",
        },
        {
            "symbol": "PERSISTENT.NS",
            "company_name": "Persistent Systems",
            "sector": "Information Technology",
            "industry": "IT Services",
        },
        # Financial Services
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
            "symbol": "AXISBANK.NS",
            "company_name": "Axis Bank",
            "sector": "Financial Services",
            "industry": "Private Bank",
        },
        {
            "symbol": "KOTAKBANK.NS",
            "company_name": "Kotak Mahindra Bank",
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
            "symbol": "BAJFINANCE.NS",
            "company_name": "Bajaj Finance",
            "sector": "Financial Services",
            "industry": "NBFC",
        },
        {
            "symbol": "BAJAJFINSV.NS",
            "company_name": "Bajaj Finserv",
            "sector": "Financial Services",
            "industry": "Financial Holding",
        },
        {
            "symbol": "HDFCLIFE.NS",
            "company_name": "HDFC Life Insurance",
            "sector": "Financial Services",
            "industry": "Insurance",
        },
        {
            "symbol": "SBILIFE.NS",
            "company_name": "SBI Life Insurance",
            "sector": "Financial Services",
            "industry": "Insurance",
        },
        # FMCG
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
            "symbol": "NESTLEIND.NS",
            "company_name": "Nestle India",
            "sector": "FMCG",
            "industry": "Food Products",
        },
        {
            "symbol": "BRITANNIA.NS",
            "company_name": "Britannia Industries",
            "sector": "FMCG",
            "industry": "Food Products",
        },
        {
            "symbol": "TATACONSUM.NS",
            "company_name": "Tata Consumer Products",
            "sector": "FMCG",
            "industry": "Consumer Goods",
        },
        # Automobile
        {
            "symbol": "MARUTI.NS",
            "company_name": "Maruti Suzuki",
            "sector": "Automobile",
            "industry": "Passenger Vehicles",
        },
        {
            "symbol": "TVSMOTOR.NS",
            "company_name": "TVS Motor Company",
            "sector": "Automobile",
            "industry": "Two Wheelers",
        },
        {
            "symbol": "M&M.NS",
            "company_name": "Mahindra & Mahindra",
            "sector": "Automobile",
            "industry": "Passenger Vehicles",
        },
        {
            "symbol": "HEROMOTOCO.NS",
            "company_name": "Hero MotoCorp",
            "sector": "Automobile",
            "industry": "Two Wheelers",
        },
        {
            "symbol": "EICHERMOT.NS",
            "company_name": "Eicher Motors",
            "sector": "Automobile",
            "industry": "Two Wheelers",
        },
        # Pharma / Healthcare
        {
            "symbol": "SUNPHARMA.NS",
            "company_name": "Sun Pharmaceutical",
            "sector": "Healthcare",
            "industry": "Pharmaceuticals",
        },
        {
            "symbol": "CIPLA.NS",
            "company_name": "Cipla",
            "sector": "Healthcare",
            "industry": "Pharmaceuticals",
        },
        {
            "symbol": "DRREDDY.NS",
            "company_name": "Dr. Reddy's Laboratories",
            "sector": "Healthcare",
            "industry": "Pharmaceuticals",
        },
        {
            "symbol": "DIVISLAB.NS",
            "company_name": "Divi's Laboratories",
            "sector": "Healthcare",
            "industry": "Pharmaceuticals",
        },
        {
            "symbol": "APOLLOHOSP.NS",
            "company_name": "Apollo Hospitals",
            "sector": "Healthcare",
            "industry": "Hospitals",
        },
        # Construction / Capital Goods
        {
            "symbol": "LT.NS",
            "company_name": "Larsen & Toubro",
            "sector": "Construction",
            "industry": "Engineering",
        },
        {
            "symbol": "ULTRACEMCO.NS",
            "company_name": "UltraTech Cement",
            "sector": "Construction Materials",
            "industry": "Cement",
        },
        {
            "symbol": "GRASIM.NS",
            "company_name": "Grasim Industries",
            "sector": "Construction Materials",
            "industry": "Cement",
        },
        {
            "symbol": "SHREECEM.NS",
            "company_name": "Shree Cement",
            "sector": "Construction Materials",
            "industry": "Cement",
        },
        # Metals
        {
            "symbol": "TATASTEEL.NS",
            "company_name": "Tata Steel",
            "sector": "Metals",
            "industry": "Steel",
        },
        {
            "symbol": "JSWSTEEL.NS",
            "company_name": "JSW Steel",
            "sector": "Metals",
            "industry": "Steel",
        },
        {
            "symbol": "HINDALCO.NS",
            "company_name": "Hindalco Industries",
            "sector": "Metals",
            "industry": "Aluminium",
        },
        # Telecom
        {
            "symbol": "BHARTIARTL.NS",
            "company_name": "Bharti Airtel",
            "sector": "Telecommunication",
            "industry": "Telecom Services",
        },
        # Consumer / Retail
        {
            "symbol": "TITAN.NS",
            "company_name": "Titan Company",
            "sector": "Consumer Discretionary",
            "industry": "Jewellery & Watches",
        },
        {
            "symbol": "ASIANPAINT.NS",
            "company_name": "Asian Paints",
            "sector": "Consumer Discretionary",
            "industry": "Paints",
        },
        {
            "symbol": "DMART.NS",
            "company_name": "Avenue Supermarts",
            "sector": "Consumer Discretionary",
            "industry": "Retail",
        },
        # Power / Utilities
        {
            "symbol": "NTPC.NS",
            "company_name": "NTPC",
            "sector": "Utilities",
            "industry": "Power Generation",
        },
        {
            "symbol": "POWERGRID.NS",
            "company_name": "Power Grid Corporation",
            "sector": "Utilities",
            "industry": "Power Transmission",
        },
        {
            "symbol": "ADANIGREEN.NS",
            "company_name": "Adani Green Energy",
            "sector": "Utilities",
            "industry": "Renewable Energy",
        },
    ]

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(assets)
    output_path = RAW_DATA_DIR / "asset_universe.csv"
    df.to_csv(output_path, index=False)

    logger.info(f"Asset universe saved to {output_path}")
    logger.info(f"Asset universe shape: {df.shape}")


if __name__ == "__main__":
    create_nifty50_universe()
