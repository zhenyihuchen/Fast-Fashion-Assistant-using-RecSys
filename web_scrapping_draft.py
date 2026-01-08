#https://www.blog.datahut.co/post/web-scraping-zara

# Target Result: Sex, Category, Item Name, Price, Image or Image URL, Color, Reference Number, Material or Composition, Description

# First approach using Selenium 

# Importing the required libraries
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from time import sleep
from csv import writer
from selenium.webdriver.common.by import By

print("Starting web scraper...")
print("Initializing Chrome driver...")
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

print("Navigating to Zara website...")
driver.get('https://www.zara.com/es/en/search?searchTerm=women%20jackets&section=WOMAN')
driver.maximize_window()
print("Page loaded!")

# Scrolling the web page
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

# Getting the product elements
page_product_links = driver.find_elements(By.XPATH, '//div[@class="product-grid-product__figure"]/a')

# Getting the product links
for product in page_product_links:
    product_link = product.get_attribute('href')
    product_links.append(product_link)

print(f"Found {len(product_links)} products")

# Extracting product name
def get_product_name():
    try:
        product_name = driver.find_element(By.XPATH, '//h1[@class="product-detail-info__header-name"]').text
    except Exception as e:
        product_name = "Not available"
    return product_name

# Extracting product mrp
def get_mrp():
    try:
        mrp = driver.find_element(By.XPATH, '//span[@class="money-amount__main"]').text
    except Exception as e:
        mrp = "Not available"
    return mrp

# Extracting product color
def get_color():
    try:
        color = driver.find_element(By.XPATH, '//p[@class="product-color-extended-name product-detail-info__color"]').text
    except Exception as e:
        color = "Not available"
    return color

# Extracting product description
def get_desc():
    try:
        desc = driver.find_element(By.XPATH, '//div[@class="expandable-text__inner-content"]/p').text
    except Exception as e:
        desc = "Not available"
    return desc

# Writing to a CSV File
print("Starting data extraction...")
with open('women_jacket_data.csv','w',newline='', encoding='utf-8') as f:
    theWriter = writer(f)
    heading = ['product_url', 'product_name', 'mrp', 'color', 'description']
    theWriter.writerow(heading)
    
    for i, product in enumerate(product_links, 1):
        print(f"Processing {i}/{len(product_links)}")
        driver.get(product)
        sleep(5)
        product_name = get_product_name()
        mrp = get_mrp()
        color = get_color()
        desc = get_desc()
        record = [product, product_name, mrp, color, desc]
        theWriter.writerow(record)
        print(f"✓ {product_name}")

print("Done!")
driver.quit()