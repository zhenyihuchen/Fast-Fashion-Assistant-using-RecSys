# Target Result: Sex, Category, Item Name, Price, Image URL, Color, Reference Number, Description

# Importing the required libraries
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import ElementClickInterceptedException, TimeoutException
from webdriver_manager.chrome import ChromeDriverManager
from time import sleep
from csv import writer
from selenium.webdriver.common.by import By

print("Starting web scraper...")
print("Initializing Chrome driver...")

# Chrome options for stability
chrome_options = Options()
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("--disable-blink-features=AutomationControlled")
chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
chrome_options.add_experimental_option('useAutomationExtension', False)

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
wait = WebDriverWait(driver, 10)

# Helper function for safe clicking
def safe_click(element):
    try:
        driver.execute_script("arguments[0].scrollIntoView(true);", element)
        sleep(0.5)
        element.click()
    except ElementClickInterceptedException:
        driver.execute_script("arguments[0].click();", element)

# Helper function to find element by text
def find_element_by_text(css_selector, text_options):
    elements = driver.find_elements(By.CSS_SELECTOR, css_selector)
    for element in elements:
        element_text = element.text.strip()
        if any(text.upper() in element_text.upper() for text in text_options):
            return element
    return None

print("Navigating to Zara website...")
driver.get('https://www.zara.com/es/en/')
driver.maximize_window()
print("Page loaded!")

# Handle cookie consent popup with OneTrust selectors
try:
    print("Handling cookie consent...")
    # Wait for overlay to disappear first
    wait.until(EC.invisibility_of_element_located((By.CSS_SELECTOR, ".onetrust-pc-dark-filter")))
    
    # Try OneTrust accept button first
    try:
        accept_cookies = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "#onetrust-accept-btn-handler")))
        safe_click(accept_cookies)
        print("OneTrust cookies accepted!")
    except TimeoutException:
        # Fallback to generic accept buttons
        accept_buttons = driver.find_elements(By.CSS_SELECTOR, "button")
        for button in accept_buttons:
            if any(text in button.text.upper() for text in ["ACCEPT", "ACEPTAR"]):
                safe_click(button)
                print("Generic cookies accepted!")
                break
    sleep(2)
except:
    print("No cookie popup found or already handled")

# Categories to scrape
categories = [
    "TOPS",
]

# Function to navigate to category
def navigate_to_category(category_name):
    print(f"Navigating to {category_name}...")
    driver.get('https://www.zara.com/es/en/')
    
    # Wait for page load
    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "header")))
    
    # Click menu icon
    menu_icon = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "#theme-app header button, header button[aria-label*='menu'], header button[aria-label*='Menu']")))
    safe_click(menu_icon)
    
    # Click WOMAN
    woman_element = find_element_by_text(".layout-categories-category-name", ["WOMAN", "MUJER"])
    if woman_element:
        wait.until(EC.element_to_be_clickable(woman_element))
        safe_click(woman_element)
    
    # Click NEW COLLECTION
    collection_element = find_element_by_text(".layout-categories-category-name", ["NEW COLLECTION", "NUEVA COLECCIÓN"])
    if collection_element:
        wait.until(EC.element_to_be_clickable(collection_element))
        safe_click(collection_element)
    
    # Click category
    category_element = find_element_by_text(".layout-categories-category-name", [category_name])
    if category_element:
        wait.until(EC.element_to_be_clickable(category_element))
        safe_click(category_element)
    
    print(f"Navigated to {category_name}!")

# Function to scroll and collect product links
def collect_product_links():
    print("Scrolling to load all products...")
    height = driver.execute_script("return document.body.scrollHeight")
    scroll_count = 0
    while True:
        driver.execute_script("window.scrollTo(0,document.body.scrollHeight)")
        wait.until(lambda d: d.execute_script("return document.readyState") == "complete")
        sleep(2)
        new_height = driver.execute_script("return document.body.scrollHeight")
        scroll_count += 1
        print(f"Scroll #{scroll_count}")
        
        if height == new_height:
            break
        height = new_height
    
    print("Collecting product links...")
    product_links = []
    
    # Wait for product grid to load
    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ".product-grid-product__figure a, [class*='product-grid'] a, [class*='product'] a[href*='/p/']")))
    
    page_product_links = driver.find_elements(By.CSS_SELECTOR, ".product-grid-product__figure a, [class*='product-grid'] a, [class*='product'] a[href*='/p/']")
    
    for product in page_product_links:
        product_link = product.get_attribute('href')
        if product_link and product_link.startswith('http') and '/p/' in product_link:
            product_links.append(product_link)
    
    print(f"Found {len(product_links)} products")
    return product_links

# Extracting product name
def get_product_name():
    try:
        selectors = [
            "h1.product-detail-info__header-name",
            "h1[class*='product-detail-info']",
            "h1[class*='header-name']",
            ".product-detail-info h1"
        ]
        for selector in selectors:
            try:
                product_name = driver.find_element(By.CSS_SELECTOR, selector).text
                return product_name
            except:
                continue
        return "Not available"
    except Exception as e:
        print(f"Product name extraction error: {e}")
        return "Not available"

# Extracting product price
def get_mrp():
    try:
        selectors = [
            ".money-amount__main",
            "[class*='money-amount'] [class*='main']",
            ".price [class*='main']",
            "[class*='price'] span"
        ]
        for selector in selectors:
            try:
                mrp = driver.find_element(By.CSS_SELECTOR, selector).text
                return mrp
            except:
                continue
        return "Not available"
    except Exception as e:
        print(f"Price extraction error: {e}")
        return "Not available"

# Extracting product color
def get_color_new():
    try:
        selectors = [
            'p[data-qa-qualifier="product-detail-info-color"]',
            "p.product-detail-color-selector__selected-color-name",
            ".product-color-extended-name",
            "p[class*='product-color-extended-name']",
            "p[class*='color-selector']"
        ]
        for selector in selectors:
            try:
                color_full = driver.find_element(By.CSS_SELECTOR, selector).text
                color = color_full.split(' | ')[0] if ' | ' in color_full else color_full
                return color.strip()
            except:
                continue
        return "Not available"
    except Exception as e:
        print(f"Color extraction error: {e}")
        return "Not available"

# Extracting reference number
def get_reference_number():
    try:
        selectors = [
            'button[data-qa-action="product-detail-info-color-copy"]',
            "button.product-color-extended-name__copy-action",
            "button[class*='copy-action']",
            "p[class*='color'] button"
        ]
        for selector in selectors:
            try:
                reference = driver.find_element(By.CSS_SELECTOR, selector).text
                return reference
            except:
                continue
        return "Not available"
    except Exception as e:
        print(f"Reference number extraction error: {e}")
        return "Not available"

# Extracting product description
def get_desc():
    try:
        selectors = [
            ".expandable-text__inner-content p",
            "[class*='expandable-text'] p",
            ".product-detail-description p",
            "[class*='description'] p"
        ]
        for selector in selectors:
            try:
                desc = driver.find_element(By.CSS_SELECTOR, selector).text
                return desc
            except:
                continue
        return "Not available"
    except Exception as e:
        print(f"Description extraction error: {e}")
        return "Not available"

# Extracting front view image URL
def get_front_view_image():
    try:
        # Find all product images
        images = driver.find_elements(By.CSS_SELECTOR, "img.media-image__image, img[class*='media-image'], img[class*='media__wrapper']")
        
        for img in images:
            alt_text = img.get_attribute('alt') or ""
            if "front view" in alt_text.lower():
                return img.get_attribute('src')
        
        # Fallback to first product image
        if images:
            return images[0].get_attribute('src')
        
        return "Not available"
    except Exception as e:
        print(f"Front view image extraction error: {e}")
        return "Not available"

# Writing to a CSV File
print("Starting data extraction for all categories...")
with open('women_all_categories_data_css.csv','w',newline='', encoding='utf-8') as f:
    theWriter = writer(f)
    heading = ['product_url', 'product_name', 'mrp', 'color', 'reference_number', 'description', 'front_view_image', 'product_category']
    theWriter.writerow(heading)
    
    for category in categories:
        try:
            navigate_to_category(category)
            product_links = collect_product_links()
            
            print(f"Starting data extraction for {category}...")
            for i, product in enumerate(product_links, 1):
                try:
                    print(f"Processing {category} {i}/{len(product_links)}")
                    driver.get(product)
                    
                    # Wait for product page to load
                    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "h1, .product-detail-info, [class*='product-detail']")))
                    
                    product_name = get_product_name()
                    mrp = get_mrp()
                    color = get_color_new()
                    reference_number = get_reference_number()
                    desc = get_desc()
                    front_view_image = get_front_view_image()
                    record = [product, product_name, mrp, color, reference_number, desc, front_view_image, category]
                    theWriter.writerow(record)
                    print(f"✓ {product_name}")
                except Exception as e:
                    print(f"Error processing product {i}: {e}")
                    continue
            
            print(f"Completed {category} - {len(product_links)} products")
        except Exception as e:
            print(f"Error processing {category}: {e}")
            continue

print("Done!")
driver.quit()