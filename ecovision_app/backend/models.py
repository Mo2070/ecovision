from sqlalchemy import create_engine, Column, Integer, String, Float, Date, UniqueConstraint, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

DATABASE_URL = "sqlite:///./ecovision.db"

Base = declarative_base()

class Country(Base):
    __tablename__ = "countries"
    id = Column(Integer, primary_key=True, index=True)
    country_code = Column(String, unique=True, index=True, nullable=False)
    country_name = Column(String, nullable=False)
    region = Column(String, nullable=True, index=True)
    income_level = Column(String, nullable=True)
    population = Column(Float, nullable=True)
    gdp_usd = Column(Float, nullable=True)
    currency_code = Column(String, nullable=True)
    capital_city = Column(String, nullable=True)

class EconomicIndicator(Base):
    __tablename__ = "economic_indicators"
    id = Column(Integer, primary_key=True, index=True)
    indicator_code = Column(String, nullable=False, index=True)
    indicator_name = Column(String, nullable=False)
    country_code = Column(String, nullable=False, index=True)
    country_name = Column(String, nullable=False)
    region = Column(String, nullable=True, index=True)
    value = Column(Float, nullable=True)
    unit = Column(String, nullable=True)
    year = Column(Integer, nullable=False, index=True)
    quarter = Column(String, nullable=True)
    data_source = Column(String, nullable=True)
    last_updated = Column(Date, nullable=True)
    category = Column(String, nullable=True)

    __table_args__ = (
        UniqueConstraint('indicator_code', 'country_code', 'year', 'quarter', name='uq_indicator_country_year_quarter'),
    )

# Create engine and session
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)
