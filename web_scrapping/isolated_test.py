"""
Isolated test (Zara): extract Front-view (packshot) image URL from:
https://www.zara.com/es/en/knit-peplum-belted-top-p04192141.html

Important:
- The "Front view of ..." alt is in the MAIN gallery picture, not in the extra-images thumbnails.
- So we scan ALL <picture class="media-image"> and check the inner <img alt=...>.
"""

import re
from time import sleep

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, StaleElementReferenceException
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd

#urls = ["https://www.zara.com/es/en/knit-peplum-belted-top-p04192141.html", "https://www.zara.com/es/en/satin-halter-top-p02231581.html?v1=502175053"]
# urls = ["https://www.zara.com/es/en/short-satin-scarf-dress-p02116341.html"]

urls = ["https://www.zara.com/es/en/wool-blend-bomber-jacket-zw-collection-p07522040.html"]

# # Read women_all_categories_dataset.csv file
# woman_data = pd.read_csv("women_all_categories_data.csv")

# # Check how many rows has in the product_image column the value "Not available"
# na_image_rows = woman_data[woman_data["product_image"] == "Not available"]
# print(f"Number of rows with 'Not available' in product_image column: {len(na_image_rows)}")

# # Save in a list all the product URLs of the rows with "Not available" in the product_image column
# urls = na_image_rows["product_url"].tolist()

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


def handle_onetrust_cookies(driver, timeout=6):
    """Accept OneTrust cookies if present."""
    try:
        btn = WebDriverWait(driver, timeout).until(
            EC.element_to_be_clickable((By.ID, "onetrust-accept-btn-handler"))
        )
        btn.click()
    except TimeoutException:
        pass

    # Wait overlay gone if present
    try:
        WebDriverWait(driver, 3).until(
            EC.invisibility_of_element_located((By.CSS_SELECTOR, ".onetrust-pc-dark-filter"))
        )
    except TimeoutException:
        pass


def debug_print_picture_alts(driver, limit=30):
    # Check extra images section specifically
    extra_imgs = driver.find_elements(By.XPATH, "//ul[@class='product-detail-view__extra-images']//img")
    print(f"\n[DEBUG] extra images found: {len(extra_imgs)}")
    for i, img in enumerate(extra_imgs[:limit], 1):
        try:
            alt = img.get_attribute("alt")
            src = img.get_attribute("src")
            print(f"[DEBUG] {i:02d} alt={repr(alt)} src={repr(src[:60] if src else None)}")
        except Exception:
            print(f"[DEBUG] {i:02d} alt=<error>")
    
    # Also check all picture elements
    pics = driver.find_elements(By.XPATH, "//picture[contains(@class,'media-image')]")
    print(f"\n[DEBUG] total picture.media-image found: {len(pics)}")
    for i, pic in enumerate(pics[:limit], 1):
        try:
            img = pic.find_element(By.XPATH, ".//img")
            alt = img.get_attribute("alt")
            print(f"[DEBUG] {i:02d} alt={repr(alt)}")
        except Exception:
            print(f"[DEBUG] {i:02d} alt=<error>")


def get_front_or_rear_packshot_url(driver, timeout=15) -> str:
    """
    Look specifically in the extra images section for Front view images.
    """
    wait = WebDriverWait(driver, timeout)

    # Wait for the extra images section to load
    try:
        wait.until(EC.presence_of_element_located((By.XPATH, "//ul[@class='product-detail-view__extra-images']//img")))
    except TimeoutException:
        return "Not available"

    # Look in the extra images section first
    extra_imgs = driver.find_elements(By.XPATH, "//ul[@class='product-detail-view__extra-images']//img[contains(@class,'media-image__image')]")
    
    for img in extra_imgs:
        try:
            alt = (img.get_attribute("alt") or "")
            alt_norm = " ".join(alt.split()).lower()
            
            if alt_norm.startswith("front view"):
                # Get the parent picture element to extract from srcset
                picture = img.find_element(By.XPATH, "./ancestor::picture[1]")
                sources = picture.find_elements(By.XPATH, ".//source[@srcset]")
                for s in sources:
                    url = _largest_from_srcset((s.get_attribute("srcset") or "").strip())
                    if url:
                        return url
                
                # Fallback to img src
                src = img.get_attribute("src")
                if src and "transparent-background.png" not in src:
                    return src
        except Exception:
            continue

    return "Not available"


def main():
    chrome_options = Options()
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option("useAutomationExtension", False)

    try:
        # Force fresh ChromeDriver download
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
    except Exception as e:
        print(f"ChromeDriver setup failed: {e}")
        print("Trying alternative setup...")
        # Fallback: try without service specification
        driver = webdriver.Chrome(options=chrome_options)

    try:
        for i, url in enumerate(urls, 1):
            print(f"\n=== Testing URL {i}/{len(urls)} ===")
            print(f"URL: {url}")
            
            driver.get(url)
            driver.maximize_window()
            sleep(1.5)  # initial hydration

            handle_onetrust_cookies(driver)

            # Scroll to load extra images
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight/2);")
            sleep(2)
            debug_print_picture_alts(driver, limit=25)

            image_url = get_front_or_rear_packshot_url(driver)
            print(f"\nFront/Rear packshot image URL: {image_url}\n")
            
            #Save the extracted image URL back to the dataframe
            # woman_data.loc[woman_data["product_url"] == url, "product_image"] = image_url

    finally:
        driver.quit()


if __name__ == "__main__":
    main()
