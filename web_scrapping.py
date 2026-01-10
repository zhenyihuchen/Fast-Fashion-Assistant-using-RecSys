# Target Result: Sex, Category, Item Name, Price, Image URL, Color, Reference Number,Description
# https://www.actowizsolutions.com/scrape-product-information-zara-selenium-python.php


# Importing the required libraries
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

print("Navigating to Zara website...")
driver.get('https://www.zara.com/es/en/')
driver.maximize_window()
sleep(5)
print("Page loaded!")

# Handle cookie consent popup
try:
    print("Handling cookie consent...")
    accept_cookies = driver.find_element(By.XPATH, '//button[contains(text(), "Accept") or contains(text(), "Aceptar")]')
    accept_cookies.click()
    sleep(2)
    print("Cookies accepted!")
except:
    print("No cookie popup found or already handled")

# Categories to scrape
categories = [
    # "JACKETS | BLAZERS",
    # "COATS",
    # "DRESSES",
    "TOPS",
    # "SHIRTS",
    # "BODIES"
    # "JEANS",
    # "TROUSERS",
    # "T-SHIRTS",
    # "CARDIGANS | JUMPERS",
    # "SKIRTS | BERMUDAS",
    # "SWEATSHIRTS | JOGGERS",
    # "SHOES",
    # "BAGS"
]

# Function to navigate to category
def navigate_to_category(category_name):
    print(f"Navigating to {category_name}...")
    driver.get('https://www.zara.com/es/en/')
    sleep(3)
    
    try:
        # Click menu icon
        print("Looking for menu icon...")
        menu_icon = wait.until(EC.element_to_be_clickable((By.XPATH, "//*[@id='theme-app']//header//button")))
        menu_icon.click()
        sleep(3)
        print("Menu opened")
        
        # Click WOMAN
        print("Looking for WOMAN section...")
        # Try multiple selectors for WOMAN
        woman_selectors = [
            '//span[@class="layout-categories-category-name" and text()="WOMAN"]',
            '//span[contains(@class,"layout-categories-category-name") and text()="WOMAN"]',
            '//span[contains(text(), "WOMAN")]',
            '//a[contains(text(), "WOMAN")]',
            '//*[text()="WOMAN"]'
        ]
        
        woman_clicked = False
        for selector in woman_selectors:
            try:
                woman_section = wait.until(EC.element_to_be_clickable((By.XPATH, selector)))
                woman_section.click()
                sleep(3)
                print("WOMAN clicked")
                woman_clicked = True
                break
            except TimeoutException:
                continue
        
        if not woman_clicked:
            print("Could not find WOMAN section, trying direct URL...")
            driver.get('https://www.zara.com/es/en/woman-l1180.html')
            sleep(3)
        
        # Click NEW COLLECTION
        print("Looking for NEW COLLECTION...")
        collection_selectors = [
            '//span[@class="layout-categories-category-name" and contains(text(), "NEW COLLECTION")]',
            '//span[contains(@class,"layout-categories-category-name") and contains(text(), "NEW COLLECTION")]',
            '//span[contains(text(), "NEW COLLECTION")]',
            '//a[contains(text(), "NEW COLLECTION")]',
            '//*[contains(text(), "NEW COLLECTION")]'
        ]
        
        collection_clicked = False
        for selector in collection_selectors:
            try:
                new_collection = wait.until(EC.element_to_be_clickable((By.XPATH, selector)))
                new_collection.click()
                sleep(3)
                print("NEW COLLECTION clicked")
                collection_clicked = True
                break
            except TimeoutException:
                continue
        
        if not collection_clicked:
            print("Could not find NEW COLLECTION, skipping...")
        
        # Click category
        print(f"Looking for {category_name}...")
        category_section = wait.until(EC.element_to_be_clickable((By.XPATH, f'//span[@class="layout-categories-category-name" and text()="{category_name}"]')))
        category_section.click()
        sleep(3)
        print(f"Navigated to {category_name}!")
    except TimeoutException as e:
        print(f"Navigation timeout error: {e}")
        raise
    except Exception as e:
        print(f"Navigation error: {e}")
        raise

# Function to scroll and collect product links
def collect_product_links():
    print("Scrolling to load all products...")
    height = driver.execute_script("return document.body.scrollHeight")
    scroll_count = 0
    while True:
        driver.execute_script("window.scrollTo(0,document.body.scrollHeight)")
        sleep(5)
        new_height = driver.execute_script("return document.body.scrollHeight")
        scroll_count += 1
        print(f"Scroll #{scroll_count}")
        
        if height == new_height:
            break
        height = new_height
    
    print("Collecting product links...")
    product_links = []
    page_product_links = driver.find_elements(By.XPATH, '//div[@class="product-grid-product__figure"]/a')
    
    for product in page_product_links:
        product_link = product.get_attribute('href')
        if product_link and product_link.startswith('http'):
            product_links.append(product_link)
    
    print(f"Found {len(product_links)} products")
    return product_links

# Extracting product name
# def get_product_name():
#     try:
#         product_name = driver.find_element(By.XPATH, '//h1[@class="product-detail-info__header-name"]').text
#     except Exception as e:
#         print(f"Product name extraction error: {e}")
#         product_name = "Not available"
#     return product_name

def get_product_name(timeout=8):
    try:
        el = WebDriverWait(driver, timeout).until(
            EC.visibility_of_element_located((By.XPATH, "//h1[contains(@class,'product-detail-info__header-name')]"))
        )
        return el.text.strip() or "Not available"
    except TimeoutException as e:
        print(f"Product name extraction error: {e}")
        return "Not available"



# Extracting product mrp
# def get_mrp():
#     try:
#         mrp = driver.find_element(By.XPATH, '//span[@class="money-amount__main"]').text
#     except Exception as e:
#         print(f"Price extraction error: {e}")
#         mrp = "Not available"
#     return mrp

def get_mrp(timeout=8):
    try:
        # There can be multiple prices; take the first visible main amount
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

def get_color_new(timeout=8):
    """
    Robust Zara color extraction using XPath.
    Tries multiple page variants and avoids exact class matching.
    """
    xpaths = [
        # Most stable (QA attribute)
        "//p[@data-qa-qualifier='product-detail-info-color']",

        # Selected color component (some templates)
        "//*[self::p or self::span][contains(@class,'product-detail-color-selector__selected-color-name')]",

        # Generic fallback
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

# Extracting reference number
# def get_reference_number():
#     try:
#         reference = driver.find_element(By.XPATH, '//button[@class="product-color-extended-name__copy-action"]').text
#     except Exception as e:
#         print(f"Reference number extraction error: {e}")
#         reference = "Not available"
#     return reference 

def get_reference_number(timeout=8):
    """
    Some pages show a 'copy action' button containing the reference text.
    Use contains(@class,...) instead of exact class match.
    """
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

# # Extracting product color
# def get_color():
#     try:
#         color = driver.find_element(By.XPATH, '//p[@class="product-color-extended-name product-detail-info__color"]').text
#     except Exception as e:
#         color = "Not available"
#     return color

# Extracting product description
# def get_desc():
#     try:
#         desc = driver.find_element(By.XPATH, '//div[@class="expandable-text__inner-content"]/p').text
#     except Exception as e:
#         print(f"Description extraction error: {e}")
#         desc = "Not available"
#     return desc

def get_desc(timeout=8):
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

# Extracting front view image URL
# def get_front_view_image():
#     try:
#         front_image = driver.find_element(By.XPATH, '//img[@class="media-image__image media__wrapper--media" and starts-with(@alt, "Front view")]')
#         front_image_url = front_image.get_attribute('src')
#     except Exception as e:
#         print(f"Front view image extraction error: {e}")
#         front_image_url = "Not available"
#     return front_image_url

# //*[@id="main"]/div/div[2]/div[1]/div[1]/button/div/div/picture/img

# //*[@id="main"]/div/div[2]/ul/li[3]/button/div/div/picture/img

# //*[@id="main"]/div/div[2]/ul/li[6]/button/div/div/picture/img
# //*[@id="main"]/div/div[2]/ul/li[7]/button/div/div/picture/img
# //*[@id="main"]/div/div[2]/ul/li[5]/button/div/div/picture/img

# def get_front_view_image(timeout=8):
#     """
#     Zara image alts vary by language. Try multiple strategies.
#     """
#     xpaths = [
#         # Your original (English)
#         "//img[contains(@class,'media-image__image') and starts-with(@alt, 'Front view')]",
#         # Spanish / more generic: any media image in gallery (first one)
#         "(//img[contains(@class,'media-image__image')])[1]",
#     ]
#     for xp in xpaths:
#         try:
#             el = WebDriverWait(driver, timeout).until(EC.presence_of_element_located((By.XPATH, xp)))
#             src = el.get_attribute("src")
#             if src:
#                 return src
#         except TimeoutException:
#             continue
#         except StaleElementReferenceException:
#             continue

#     return "Not available"

def get_product_image(timeout=8):
    try:
        WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((By.XPATH, "//img[contains(@class,'media-image__image media__wrapper--media')]"))
        )
        images = driver.find_elements(By.XPATH, "//img[contains(@class,'media-image__image media__wrapper--media')]")
        
        for img in images:
            src = img.get_attribute("src")
            if src and "transparent-background.png" not in src:
                return src
        
        return "Not available"
    except Exception as e:
        print(f"Product image extraction error: {e}")
        return "Not available"



# Writing to a CSV File
print("Starting data extraction for all categories...")
with open('women_all_categories_data.csv','w',newline='', encoding='utf-8') as f:
    theWriter = writer(f)
    heading = ['product_url', 'product_name', 'mrp', 'color', 'reference_number', 'description', 'product_image', 'product_category']
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
                    sleep(2)
                    product_name = get_product_name()
                    mrp = get_mrp()
                    color = get_color_new()
                    reference_number = get_reference_number()
                    desc = get_desc()
                    product_image = get_product_image()
                    record = [product, product_name, mrp, color, reference_number, desc, product_image, category]
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