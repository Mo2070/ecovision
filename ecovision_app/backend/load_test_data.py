import requests

BASE_URL = "http://127.0.0.1:8000"

# Sample countries data
countries_data = [
    {
        "country_code": "US",
        "country_name": "United States",
        "region": "North America",
        "income_level": "High income",
        "population": 331000000,
        "gdp_usd": 21427700000000,
        "currency_code": "USD",
        "capital_city": "Washington D.C."
    },
    {
        "country_code": "DE",
        "country_name": "Germany",
        "region": "Europe",
        "income_level": "High income",
        "population": 83200000,
        "gdp_usd": 3845630000000,
        "currency_code": "EUR",
        "capital_city": "Berlin"
    },
    {
        "country_code": "JP",
        "country_name": "Japan",
        "region": "Asia",
        "income_level": "High income",
        "population": 125800000,
        "gdp_usd": 5064870000000,
        "currency_code": "JPY",
        "capital_city": "Tokyo"
    }
]

# Sample indicators data
indicators_data = [
    {
        "indicator_code": "GDP",
        "indicator_name": "Gross Domestic Product",
        "country_code": "US",
        "value": 21427700000000,
        "unit": "USD",
        "year": 2023,
        "quarter": "Q4",
        "data_source": "World Bank",
        "last_updated": "2025-08-01",
        "category": "GDP"
    },
    {
        "indicator_code": "GDP",
        "indicator_name": "Gross Domestic Product",
        "country_code": "DE",
        "value": 3845630000000,
        "unit": "USD",
        "year": 2023,
        "quarter": "Q4",
        "data_source": "World Bank",
        "last_updated": "2025-08-01",
        "category": "GDP"
    },
    {
        "indicator_code": "POP",
        "indicator_name": "Population",
        "country_code": "US",
        "value": 331000000,
        "unit": "people",
        "year": 2023,
        "quarter": None,
        "data_source": "UN Data",
        "last_updated": "2025-08-01",
        "category": "Demographics"
    }
]

def post_data(endpoint, data):
    url = f"{BASE_URL}{endpoint}"
    response = requests.post(url, json=data)
    if response.status_code == 200:
        print(f"✅ Successfully loaded data into {endpoint}")
    else:
        print(f"❌ Failed to load {endpoint}: {response.status_code} - {response.text}")

if __name__ == "__main__":
    print("📤 Loading countries...")
    post_data("/api/countries/bulk", countries_data)

    print("📤 Loading indicators...")
    post_data("/api/indicators/bulk", indicators_data)
