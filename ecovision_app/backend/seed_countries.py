from sqlalchemy.orm import Session
from backend.models import Country, init_db, SessionLocal

# ISO + World Bank region & income level dataset
COUNTRIES_DATA = [
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
        "population": 83000000,
        "gdp_usd": 3845630000000,
        "currency_code": "EUR",
        "capital_city": "Berlin"
    },
    {
        "country_code": "CN",
        "country_name": "China",
        "region": "Asia",
        "income_level": "Upper middle income",
        "population": 1444216107,
        "gdp_usd": 14342903000000,
        "currency_code": "CNY",
        "capital_city": "Beijing"
    },
    # Add more countries as needed
]

def seed_countries():
    init_db()  # Ensure tables exist
    db: Session = SessionLocal()

    # Clear existing countries (optional)
    db.query(Country).delete()

    # Insert new data
    for country in COUNTRIES_DATA:
        db.add(Country(**country))

    db.commit()
    db.close()
    print(f" ✅ Inserted {len(COUNTRIES_DATA)} countries into database.")

if __name__ == "__main__":
    seed_countries()
