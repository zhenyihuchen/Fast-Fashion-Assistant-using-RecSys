# Target Result: Sex, Category, Item Name, Price, Image URL, Color, Reference Number,Description
# https://www.actowizsolutions.com/scrape-product-information-zara-selenium-python.php


# Importing the required libraries
import re
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from time import sleep
from csv import writer
from selenium.webdriver.common.by import By
from selenium.common.exceptions import (
    TimeoutException,
    NoSuchElementException,
    ElementClickInterceptedException,
    StaleElementReferenceException,
)


print("Starting web scraper...")

DEFAULT_TIMEOUT = 3
IMAGE_TIMEOUT = 6

# Categories with direct URLs
category_urls = {
    "JACKETS": "https://www.zara.com/es/en/woman-jackets-l1114.html?v1=2536936",
    "COATS": "https://www.zara.com/es/en/woman-outerwear-l1184.html?v1=2419032",
    "DRESSES": "https://www.zara.com/es/en/woman-dresses-l1066.html?v1=2420896",
    "TOPS": "https://www.zara.com/es/en/woman-tops-l1322.html?v1=2419940",
    "SHIRTS": "https://www.zara.com/es/en/woman-shirts-l1217.html?v1=2420369",
    "BODIES": "https://www.zara.com/es/en/woman-body-l1057.html?v1=2420490",
    "JEANS": "https://www.zara.com/es/en/woman-jeans-l1119.html?v1=2419185",
    "TROUSERS": "https://www.zara.com/es/en/woman-trousers-l1335.html?v1=2420795",
    "T-SHIRTS": "https://www.zara.com/es/en/woman-tshirts-l1362.html?v1=2420417",
    "CARDIGANS": "https://www.zara.com/es/en/woman-cardigans-sweaters-l8322.html?v1=2419844",
    "SKIRTS": "https://www.zara.com/es/en/woman-skirts-l1299.html?v1=2420454",
    "SWEATSHIRTS": "https://www.zara.com/es/en/woman-sweatshirts-l1320.html?v1=2467841",
    "SHOES": "https://www.zara.com/es/en/woman-shoes-l1251.html?v1=2419160",
    "BAGS": "https://www.zara.com/es/en/woman-bags-l1024.html?v1=2417728"
}

def create_driver():
    chrome_options = Options()
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)



# Direct navigation to category URL
def navigate_to_category(category_name, category_url, driver):
    print(f"Navigating to {category_name}...")
    driver.get(category_url)
    sleep(2)
    
    # Handle cookie consent popup if present
    try:
        accept_cookies = driver.find_element(By.XPATH, '//button[contains(text(), "Accept") or contains(text(), "Aceptar")]')
        accept_cookies.click()
        sleep(1)
    except:
        pass
    
    print(f"Navigated to {category_name}!")

# COMMENTED OUT, Old menu navigation method of manually clickling through menu - not very efficient 
# def navigate_to_category(category_name):
#     print(f"Navigating to {category_name}...")
#     
#     for attempt in range(3):  # Retry up to 3 times
#         try:
#             driver.get('https://www.zara.com/es/en/')
#             sleep(5)  # Increased wait time
#             
#             # Click menu icon
#             print("Looking for menu icon...")
#             menu_icon = WebDriverWait(driver, 15).until(EC.element_to_be_clickable((By.XPATH, "//*[@id='theme-app']//header//button")))
#             driver.execute_script("arguments[0].click();", menu_icon)  # Use JS click
#             sleep(3)
#             print("Menu opened")
#             
#             # Click WOMAN
#             print("Looking for WOMAN section...")
#             woman_selectors = [
#                 '//span[@class="layout-categories-category-name" and text()="WOMAN"]',
#                 '//span[contains(@class,"layout-categories-category-name") and text()="WOMAN"]',
#                 '//span[contains(text(), "WOMAN")]'
#             ]
#             
#             woman_clicked = False
#             for selector in woman_selectors:
#                 try:
#                     woman_section = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, selector)))
#                     driver.execute_script("arguments[0].click();", woman_section)  # Use JS click
#                     sleep(3)
#                     print("WOMAN clicked")
#                     woman_clicked = True
#                     break
#                 except TimeoutException:
#                     continue
#             
#             if not woman_clicked:
#                 print("Could not find WOMAN section, trying direct URL...")
#                 driver.get('https://www.zara.com/es/en/woman-l1180.html')
#                 sleep(5)
#             
#             # Click NEW COLLECTION (optional step)
#             print("Looking for NEW COLLECTION...")
#             try:
#                 new_collection = WebDriverWait(driver, 8).until(
#                     EC.element_to_be_clickable((By.XPATH, '//span[@class="layout-categories-category-name" and contains(text(), "NEW COLLECTION")]'))
#                 )
#                 driver.execute_script("arguments[0].click();", new_collection)
#                 sleep(3)
#                 print("NEW COLLECTION clicked")
#             except TimeoutException:
#                 print("NEW COLLECTION not found, continuing...")
#             
#             # Click category
#             print(f"Looking for {category_name}...")
#             category_section = WebDriverWait(driver, 15).until(
#                 EC.element_to_be_clickable((By.XPATH, f'//span[@class="layout-categories-category-name" and text()="{category_name}"]'))
#             )
#             driver.execute_script("arguments[0].click();", category_section)  # Use JS click
#             sleep(5)
#             print(f"Navigated to {category_name}!")
#             return  # Success, exit retry loop
#             
#         except Exception as e:
#             print(f"Attempt {attempt + 1} failed: {e}")
#             if attempt == 2:  # Last attempt
#                 raise
#             sleep(3)

# Function to scroll and collect product links
def collect_product_links(driver):
    print("Scrolling to load all products...")
    # Scroll 3 times quickly instead of waiting for no height change
    for i in range(3):
        driver.execute_script("window.scrollTo(0,document.body.scrollHeight)")
        sleep(2)
        print(f"Scroll #{i+1}")
    
    print("Collecting product links...")
    product_links = []
    page_product_links = driver.find_elements(By.XPATH, '//div[@class="product-grid-product__figure"]/a')
    
    for product in page_product_links:
        product_link = product.get_attribute('href')
        # Validate URL format immediately
        if product_link and product_link.startswith('https://www.zara.com'):
            product_links.append(product_link)
            print(f"Valid URL collected: {product_link}")
        else:
            print(f"Invalid URL skipped: {product_link}")
    
    # Remove duplicates while preserving order
    unique_links = list(dict.fromkeys(product_links))
    duplicates_removed = len(product_links) - len(unique_links)
    
    print(f"Found {len(product_links)} total products")
    print(f"Removed {duplicates_removed} duplicate URLs")
    print(f"Final unique products: {len(unique_links)}")
    
    return unique_links

# Extracting product name
# def get_product_name():
#     try:
#         product_name = driver.find_element(By.XPATH, '//h1[@class="product-detail-info__header-name"]').text
#     except Exception as e:
#         print(f"Product name extraction error: {e}")
#         product_name = "Not available"
#     return product_name

def get_product_name(driver, timeout=DEFAULT_TIMEOUT):
    try:
        el = WebDriverWait(driver, timeout).until(
            EC.visibility_of_element_located((By.XPATH, "//h1[contains(@class,'product-detail-info__header-name')]"))
        )
        return el.text.strip() or "Not available"
    except TimeoutException as e:
        print(f"Product name extraction error: {e}")
        return "Not available"

def get_price(driver, timeout=DEFAULT_TIMEOUT):
    try:
        el = WebDriverWait(driver, timeout).until(
            EC.visibility_of_element_located((By.XPATH, "//span[contains(@class,'money-amount__main')]"))
        )
        return el.text.strip() or "Not available"
    except TimeoutException as e:
        print(f"Price extraction error: {e}")
        return "Not available"


# Extracting product color with new XPath
# def get_color_new():
#     try:
#         color_full = driver.find_element(By.XPATH, '//p[@class="product-color-extended-name product-detail-color-selector__selected-color-name"]').text
#         color = color_full.split(' | ')[0] if ' | ' in color_full else color_full
#     except Exception as e:
#         print(f"Color extraction error: {e}")
#         color = "Not available"
#     return color

def get_color_new(driver, timeout=DEFAULT_TIMEOUT):
    xpaths = [
        "//p[@data-qa-qualifier='product-detail-info-color']",
        "//*[self::p or self::span][contains(@class,'product-detail-color-selector__selected-color-name')]",
        "//*[self::p or self::span][contains(@class,'product-color-extended-name')]",
    ]

    for xp in xpaths:
        try:
            el = WebDriverWait(driver, timeout).until(
                EC.visibility_of_element_located((By.XPATH, xp))
            )
            text = el.text.strip()
            if not text:
                continue
            if " | " in text:
                text = text.split(" | ", 1)[0].strip()
            return text
        except TimeoutException:
            continue
        except StaleElementReferenceException:
            continue

    return "Not available"

def get_reference_number(driver, timeout=DEFAULT_TIMEOUT):
    xpaths = [
        "//button[contains(@class,'product-color-extended-name__copy-action')]",
        "//*[self::button or self::span or self::p][contains(@class,'product-color-extended-name__copy-action')]",
    ]
    for xp in xpaths:
        try:
            el = WebDriverWait(driver, timeout).until(EC.visibility_of_element_located((By.XPATH, xp)))
            txt = el.text.strip()
            if txt:
                return txt
        except TimeoutException:
            continue
        except StaleElementReferenceException:
            continue

    return "Not available"

def get_desc(driver, timeout=DEFAULT_TIMEOUT):
    try:
        el = WebDriverWait(driver, timeout).until(
            EC.visibility_of_element_located(
                (By.XPATH, "//div[contains(@class,'expandable-text__inner-content')]//p")
            )
        )
        return el.text.strip() or "Not available"
    except TimeoutException as e:
        print(f"Description extraction error: {e}")
        return "Not available"

def _largest_from_srcset(srcset: str) -> str | None:
    if not srcset:
        return None
    parts = [p.strip() for p in srcset.split(",") if p.strip()]
    best_url, best_w = None, -1
    for part in parts:
        segs = part.split()
        url = segs[0].strip()
        w = -1
        if len(segs) > 1:
            m = re.match(r"(\d+)w", segs[1])
            if m:
                w = int(m.group(1))
        if w > best_w:
            best_w, best_url = w, url
    return best_url

def get_product_image(driver, timeout=IMAGE_TIMEOUT):
    try:
        # Clear any existing state and ensure fresh page load
        driver.execute_script("window.scrollTo(0, 0);")
        sleep(1)
        
        # Scroll to load extra images (same as isolated test)
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight/2);") #scrolls halfway down the page to trigger lazy loading of the extra images section.
        sleep(2)
        
        # Wait for extra images section to load
        WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((By.XPATH, "//ul[@class='product-detail-view__extra-images']//picture"))
        )
        
        # Get all picture elements
        pictures = driver.find_elements(By.XPATH, "//ul[@class='product-detail-view__extra-images']//picture[@class='media-image']")
        print(f"Found {len(pictures)} picture elements to check")
        
        # First pass: look for front view images
        for i, picture in enumerate(pictures):
            try:
                img = picture.find_element(By.XPATH, ".//img")
                alt = (img.get_attribute("alt") or "")
                alt_norm = " ".join(alt.split()).lower()
                print(f"Picture {i+1} img alt: '{alt}'")
                
                if "front view" in alt_norm:
                    print(f"Found front view in picture {i+1}")
                    # Try to get from srcset first
                    sources = picture.find_elements(By.XPATH, ".//source[@srcset]")
                    for s in sources:
                        url = _largest_from_srcset((s.get_attribute("srcset") or "").strip())
                        if url:
                            print(f"Extracted URL from srcset: {url[:50]}...")
                            return url
                    
                    # Fallback to img src
                    src = img.get_attribute("src")
                    if src and "transparent-background.png" not in src:
                        print(f"Using img src: {src[:50]}...")
                        return src
                        
            except Exception:
                continue
        
        print("No front view found")
        
        print("Product image extraction error: No valid images found")
        return "Not available"
        
    except Exception as e:
        print(f"Product image extraction error: {e}")
        return "Not available"


# Scrape single category function for parallel processing
def scrape_category(category_name, category_url):
    driver = create_driver()
    driver.maximize_window()
    
    try:
        navigate_to_category(category_name, category_url, driver)
        product_links = collect_product_links(driver)
        
        category_data = []
        processed_count = 0
        success_count = 0
        error_count = 0
        print(f"Starting data extraction for {category_name}...")
        
        for i, product in enumerate(product_links, 1):
            processed_count += 1
            try:
                print(f"Processing {category_name} {i}/{len(product_links)}: {product}")
                driver.get(product)
                sleep(1)  # Reduced sleep time
                
                product_name = get_product_name(driver)
                price = get_price(driver)
                color = get_color_new(driver)
                reference_number = get_reference_number(driver)
                desc = get_desc(driver)
                product_image = get_product_image(driver)
                
                # Validate all extracted data before creating record
                if not product or not product.startswith('https://www.zara.com'):
                    print(f"✗ Invalid product URL: {product}")
                    error_count += 1
                    continue
                    
                # Ensure no field contains newlines or commas that could break CSV
                product_name = product_name.replace('\n', ' ').replace('\r', ' ') if product_name else "Not available"
                desc = desc.replace('\n', ' ').replace('\r', ' ') if desc else "Not available"
                
                extracted_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
                record = [product, product_name, price, color, reference_number, desc, product_image, category_name, extracted_at]
                
                # Final validation - ensure all fields are strings
                record = [str(field) if field is not None else "Not available" for field in record]
                
                category_data.append(record)
                success_count += 1
                print(f"✓ {product_name} - SUCCESS")
                
            except Exception as e:
                print(f"✗ Error processing product {i}: {e}")
                error_count += 1
                continue
        
        print(f"Completed {category_name}:")
        print(f"  - Products found: {len(product_links)}")
        print(f"  - Products processed: {processed_count}")
        print(f"  - Successful extractions: {success_count}")
        print(f"  - Errors/skipped: {error_count}")
        return category_data
        
    except Exception as e:
        print(f"Error processing {category_name}: {e}")
        return []
    finally:
        driver.quit()



# Main execution with parallel processing
if __name__ == "__main__":
    print("Starting parallel data extraction...")
    
    # Use ThreadPoolExecutor for parallel processing (1-2 threads to avoid overwhelming the server)
    with ThreadPoolExecutor(max_workers=1) as executor:
        # Submit all category scraping tasks
        future_to_category = {
            executor.submit(scrape_category, category, url): category 
            for category, url in category_urls.items()
        }
        
        # Collect results and write to CSV
        total_found = 0
        total_written = 0
        
        with open('women_all_categories_data.csv', 'w', newline='', encoding='utf-8') as f:
            theWriter = writer(f)
            heading = ['row_id', 'product_url', 'product_name', 'price', 'color', 'reference_number', 'description', 'image_url', 'product_category', 'extracted_at']
            theWriter.writerow(heading)
            
            # Process completed tasks as they finish
            for future in future_to_category:
                category = future_to_category[future]
                try:
                    category_data = future.result()
                    total_found += len(category_data)
                    
                    for record in category_data:
                        total_written += 1
                        theWriter.writerow([total_written] + record)
                    print(f"✓ Completed writing {category} data to CSV")
                except Exception as e:
                    print(f"✗ Error with {category}: {e}")
        
        print(f"\n=== FINAL SUMMARY ===")
        print(f"Total products extracted: {total_found}")
        print(f"Total products written to CSV: {total_written}")
    
    print("Done! All categories processed.")
