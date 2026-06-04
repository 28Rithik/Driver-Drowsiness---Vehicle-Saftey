import requests
from bs4 import BeautifulSoup
import json
import csv
import re
import time
from urllib.parse import urljoin
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OBDScraper:
    def __init__(self):
        self.base_url = "https://obdguide.com/en"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        self.usd_to_inr = 83.0
        self.categories = {
            'powertrain': 'category/powertrain',
            'body': 'category/body',
            'chassis': 'category/chassis',
            'network': 'category/network'
        }
        self.all_codes = []

    def normalize_text(self, text):
        if not text:
            return ""
        return re.sub(r"\s+", " ", text).strip()

    def extract_section_items(self, soup, heading_name):
        heading = soup.find(lambda tag: tag.name in ['h2', 'h3'] and self.normalize_text(tag.get_text(" ", strip=True)).lower() == heading_name.lower())
        if not heading:
            return []

        section_list = heading.find_next(['ul', 'ol'])
        if not section_list:
            return []

        return [self.normalize_text(li.get_text(" ", strip=True)) for li in section_list.find_all('li')]

    def extract_label_value(self, page_text, label, stop_labels=None):
        stop_labels = stop_labels or []
        escaped_label = re.escape(label)
        escaped_stops = "|".join(re.escape(item) for item in stop_labels)
        pattern = rf"{escaped_label}\s*(.*?)(?:\s*(?:{escaped_stops})\s*|$)" if escaped_stops else rf"{escaped_label}\s*(.*?)(?:$)"
        match = re.search(pattern, page_text, flags=re.IGNORECASE)
        if not match:
            return ""
        return self.normalize_text(match.group(1))

    def extract_repair_cost(self, page_text):
        match = re.search(r"(?:Approx\.\s*)?repair cost\s*~?\$([\d,]+(?:\.\d+)?)", page_text, flags=re.IGNORECASE)
        if not match:
            match = re.search(r"\$([\d,]+(?:\.\d+)?)", page_text)
        if not match:
            return "", ""

        usd_value = float(match.group(1).replace(',', ''))
        inr_value = round(usd_value * self.usd_to_inr)
        repair_cost_usd = f"${usd_value:,.2f}".replace('.00', '')
        repair_cost_inr = f"₹{inr_value:,.0f}"
        return repair_cost_usd, repair_cost_inr

    def extract_title_and_description(self, soup):
        h1 = soup.find('h1')
        title = ""
        description = ""

        if h1:
            title_text = self.normalize_text(h1.get_text(" ", strip=True))
            title = title_text

            paragraph = h1.find_next('p')
            if paragraph:
                description = self.normalize_text(paragraph.get_text(" ", strip=True))

        return title, description
        
    def fetch_page(self, url):
        """Fetch a page with error handling"""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            logger.error(f"Error fetching {url}: {e}")
            return None
    
    def scrape_category_listing(self, category_name, category_path):
        """Scrape all OBD codes from a category listing page"""
        url = f"{self.base_url}/{category_path}"
        logger.info(f"Scraping category: {category_name} - {url}")
        
        html = self.fetch_page(url)
        if not html:
            return []
        
        soup = BeautifulSoup(html, 'html.parser')
        codes = []
        seen_codes = set()
        
        # Find all code links in the category page
        # Adjust selectors based on actual HTML structure
        code_elements = soup.find_all('a', href=True)
        
        for element in code_elements:
            href = element.get('href', '')
            text = element.get_text(strip=True)

            # Filter for OBD code patterns (P, B, C, U followed by 4 hexadecimal characters)
            match = re.search(r"/en/([pbcu][0-9a-f]{4})/?$", href, flags=re.IGNORECASE)
            if match:
                code = match.group(1).upper()
                if code not in seen_codes:
                    seen_codes.add(code)
                    codes.append({
                        'code': code,
                        'category': category_name,
                        'url': urljoin(self.base_url, href),
                        'title': text
                    })
        
        logger.info(f"Found {len(codes)} codes in {category_name}")
        return codes
    
    def scrape_code_details(self, code_info):
        """Scrape detailed information for a specific OBD code"""
        url = code_info['url']
        logger.info(f"Scraping code details: {code_info['code']} - {url}")
        
        html = self.fetch_page(url)
        if not html:
            return code_info
        
        soup = BeautifulSoup(html, 'html.parser')

        title, description = self.extract_title_and_description(soup)
        page_text = self.normalize_text(soup.get_text(" ", strip=True))

        severity_match = re.search(r"Severity\s*(✅\s*Low|⚠️\s*Medium|🔴\s*High|Low|Medium|High)", page_text, flags=re.IGNORECASE)
        severity = self.normalize_text(severity_match.group(1)) if severity_match else ""

        can_drive = self.extract_label_value(page_text, "Can you drive?", ["Approx. repair cost", "Symptoms", "Causes", "How to Fix", "FAQ"])
        repair_cost_usd, repair_cost_inr = self.extract_repair_cost(page_text)

        symptoms = self.extract_section_items(soup, "Symptoms")
        causes = self.extract_section_items(soup, "Causes")
        fixes = self.extract_section_items(soup, "How to Fix")

        summary_match = re.search(rf"{re.escape(code_info['code'])}\s+(.+?)\s*(?:✅\s*Low|⚠️\s*Medium|🔴\s*High)", page_text, flags=re.IGNORECASE)
        short_description = self.normalize_text(summary_match.group(1)) if summary_match else ""
        
        # Update the code info with extracted details
        code_info.update({
            'title': title or code_info.get('title', ''),
            'short_description': short_description,
            'description': description,
            'severity': severity,
            'can_drive': can_drive,
            'repair_cost_usd': repair_cost_usd,
            'repair_cost_inr': repair_cost_inr,
            'symptoms': symptoms,
            'causes': causes,
            'how_to_fix': fixes
        })
        
        return code_info
    
    def save_to_json(self, filename='obd_codes.json'):
        """Save all scraped data to JSON file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.all_codes, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(self.all_codes)} codes to {filename}")
    
    def save_to_csv(self, filename='obd_codes.csv'):
        """Save all scraped data to CSV file"""
        if not self.all_codes:
            logger.warning("No codes to save")
            return
        
        keys = self.all_codes[0].keys()

        def serialize(value):
            if isinstance(value, list):
                return " | ".join(value)
            return value

        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for row in self.all_codes:
                writer.writerow({key: serialize(row.get(key, "")) for key in keys})
        logger.info(f"Saved {len(self.all_codes)} codes to {filename}")
    
    def run(self, max_codes_per_category=None, delay=0.5):
        """Run the complete scraping process"""
        logger.info("Starting OBD code scraping...")
        
        try:
            for category_name, category_path in self.categories.items():
                # Scrape category listings
                category_codes = self.scrape_category_listing(category_name, category_path)
                
                # Limit codes if specified
                if max_codes_per_category:
                    category_codes = category_codes[:max_codes_per_category]
                
                # Scrape details for each code
                for code_info in category_codes:
                    code_info = self.scrape_code_details(code_info)
                    self.all_codes.append(code_info)
                    time.sleep(delay)  # Be respectful to the server
                
                # Add delay between categories
                time.sleep(1)
            
            logger.info(f"Scraping complete! Total codes: {len(self.all_codes)}")
            
            # Save results
            self.save_to_json()
            self.save_to_csv()
            
            return self.all_codes
            
        except Exception as e:
            logger.error(f"Error during scraping: {e}")
            return self.all_codes


if __name__ == "__main__":
    # Create scraper instance
    scraper = OBDScraper()
    
    # Run with optional limits (remove or adjust these parameters as needed)
    # max_codes_per_category: limit codes per category (None = all)
    # delay: seconds between requests (0.5 = respectful)
    scraper.run(max_codes_per_category=None, delay=0.5)
    
    # Display summary
    print("\n" + "="*50)
    print(f"Total codes scraped: {len(scraper.all_codes)}")
    print("Files saved:")
    print("  - obd_codes.json")
    print("  - obd_codes.csv")
    print("="*50)
