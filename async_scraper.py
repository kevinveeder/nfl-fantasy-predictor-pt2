"""
Async web scraping module for parallel data collection
"""
import asyncio
import aiohttp
import pandas as pd
import time
from typing import List, Dict, Optional
from bs4 import BeautifulSoup
import logging

logger = logging.getLogger(__name__)

class AsyncFantasyScraper:
    """Async scraper for fantasy football data with concurrent requests"""
    
    def __init__(self, max_concurrent_requests: int = 4, request_delay: float = 0.5):
        self.max_concurrent_requests = max_concurrent_requests
        self.request_delay = request_delay
        self.session = None
        
    async def __aenter__(self):
        connector = aiohttp.TCPConnector(limit=self.max_concurrent_requests)
        timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(connector=connector, timeout=timeout)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def fetch_url(self, url: str, semaphore: asyncio.Semaphore) -> Optional[str]:
        """Fetch a single URL with rate limiting"""
        async with semaphore:
            try:
                await asyncio.sleep(self.request_delay)  # Rate limiting
                async with self.session.get(url) as response:
                    if response.status == 200:
                        return await response.text()
                    else:
                        logger.warning(f"HTTP {response.status} for {url}")
                        return None
            except Exception as e:
                logger.error(f"Error fetching {url}: {e}")
                return None
    
    async def scrape_position_async(self, position: str) -> Optional[pd.DataFrame]:
        """Scrape a single position asynchronously"""
        url = f"https://www.fantasypros.com/nfl/projections/{position.lower()}.php"
        
        semaphore = asyncio.Semaphore(1)  # One request at a time per position
        html = await self.fetch_url(url, semaphore)
        
        if not html:
            return None
            
        try:
            # Parse HTML and extract table
            soup = BeautifulSoup(html, 'html.parser')
            table = soup.find('table', {'id': 'data'})
            
            if not table:
                logger.warning(f"No data table found for {position}")
                return None
            
            # Use pandas to parse the table efficiently
            df = pd.read_html(str(table))[0]
            
            # Clean column names vectorized
            df.columns = [col.replace('\\n', ' ').strip() if isinstance(col, str) else col for col in df.columns]
            
            # Add position column
            df['Position'] = position.upper()
            
            # Vectorized numeric conversion for all stat columns
            stat_columns = [col for col in df.columns if col not in ['Player', 'Position']]
            df[stat_columns] = df[stat_columns].apply(pd.to_numeric, errors='coerce')
            
            logger.info(f"Scraped {len(df)} {position} players")
            return df
            
        except Exception as e:
            logger.error(f"Error parsing {position} data: {e}")
            return None
    
    async def scrape_all_positions_async(self, positions: List[str] = None) -> Optional[pd.DataFrame]:
        """Scrape all positions concurrently"""
        if positions is None:
            positions = ['QB', 'RB', 'WR', 'TE']
        
        logger.info(f"Starting async scraping for {len(positions)} positions...")
        
        # Create semaphore for overall rate limiting
        semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        
        # Create tasks for all positions
        tasks = [self.scrape_position_async(position) for position in positions]
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter successful results
        successful_dfs = []
        for i, result in enumerate(results):
            if isinstance(result, pd.DataFrame) and not result.empty:
                successful_dfs.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Error scraping {positions[i]}: {result}")
        
        if not successful_dfs:
            logger.error("No positions scraped successfully")
            return None
        
        # Concatenate all DataFrames efficiently
        combined_df = pd.concat(successful_dfs, ignore_index=True, sort=False)
        logger.info(f"Successfully scraped {len(combined_df)} total players")
        
        return combined_df

async def scrape_historical_data_async(years: List[int], max_concurrent: int = 3) -> Optional[pd.DataFrame]:
    """Async scraping for historical data from multiple years"""
    
    async def scrape_year(session: aiohttp.ClientSession, year: int, semaphore: asyncio.Semaphore) -> Optional[pd.DataFrame]:
        """Scrape data for a single year"""
        url = f"https://www.pro-football-reference.com/years/{year}/fantasy.htm"
        
        async with semaphore:
            try:
                await asyncio.sleep(0.5)  # Rate limiting
                async with session.get(url) as response:
                    if response.status == 200:
                        html = await response.text()
                        
                        # Use pandas read_html which is already optimized
                        df = pd.read_html(html, header=1)[0]
                        
                        # Vectorized cleaning
                        df = df[df['Rk'] != 'Rk']  # Remove repeated headers
                        df['Year'] = year
                        
                        # Vectorized numeric conversion
                        numeric_columns = [
                            'FantPt', 'G', 'Att', 'Tgt', 'Rec', 'Cmp', 'Att.1', 'Yds', 'TD', 'Int', 
                            'Yds.1', 'TD.1', 'Yds.2', 'TD.2', 'FL', 'Fmb'
                        ]
                        existing_numeric_cols = [col for col in numeric_columns if col in df.columns]
                        df[existing_numeric_cols] = df[existing_numeric_cols].apply(pd.to_numeric, errors='coerce')
                        
                        logger.info(f"Scraped {len(df)} players from {year}")
                        return df
                        
            except Exception as e:
                logger.error(f"Error scraping {year}: {e}")
                return None
    
    # Set up async session and semaphore
    connector = aiohttp.TCPConnector(limit=max_concurrent)
    timeout = aiohttp.ClientTimeout(total=60)
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        # Create tasks for all years
        tasks = [scrape_year(session, year, semaphore) for year in years]
        
        # Execute concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter successful results
        successful_dfs = []
        for i, result in enumerate(results):
            if isinstance(result, pd.DataFrame) and not result.empty:
                successful_dfs.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Error scraping {years[i]}: {result}")
        
        if not successful_dfs:
            return None
        
        # Concatenate efficiently
        combined_df = pd.concat(successful_dfs, ignore_index=True, sort=False)
        logger.info(f"Total historical data: {len(combined_df)} player-seasons")
        
        return combined_df

# Synchronous wrapper for backward compatibility
def scrape_all_positions_sync(positions: List[str] = None) -> Optional[pd.DataFrame]:
    """Synchronous wrapper for async scraping"""
    async def run_async():
        async with AsyncFantasyScraper() as scraper:
            return await scraper.scrape_all_positions_async(positions)
    
    return asyncio.run(run_async())

def scrape_historical_data_sync(years: List[int]) -> Optional[pd.DataFrame]:
    """Synchronous wrapper for async historical scraping"""
    return asyncio.run(scrape_historical_data_async(years))