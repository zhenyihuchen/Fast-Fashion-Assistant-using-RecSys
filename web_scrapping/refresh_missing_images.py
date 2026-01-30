import csv
import re
from time import sleep

from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from web_scrapping.web_scrapping import create_driver


INPUT_CSV = "women_all_categories_data_updated1.csv"
OUTPUT_CSV = "women_all_categories_data_updated2.csv"
IMAGE_TIMEOUT = 6


def _is_missing_image(value: str) -> bool:
    return (value or "").strip() == "Not available"

def _accept_cookies(driver) -> None:
    try:
        accept = driver.find_element(
            "xpath", '//button[contains(text(), "Accept") or contains(text(), "Aceptar")]'
        )
        accept.click()
        sleep(1)
    except Exception:
        pass


def _largest_from_srcset(srcset: str) -> str | None: # returns the best image URL from a srcset string or None.
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
    return best_url # return the largest-width URL


def _find_pictures(driver, timeout: int): # helper to wait for and collect picture elements
    try:
        WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located(
                (By.XPATH, "//ul[@class='product-detail-view__extra-images']//picture")
            ) # wait until at least one picture appears in the extra-images list.
        )
    except TimeoutException: # if it never appears in time…  return empty list
        return []
    return driver.find_elements(
        By.XPATH,
        "//ul[@class='product-detail-view__extra-images']//picture[@class='media-image']",
    ) # collect all <picture class="media-image"> elements in that list


def _front_view_from_pictures(pictures) -> str | None: # scan picture elements for a front view URL
    first_rear_url = None
    rear_count = 0
    for i, picture in enumerate(pictures):
        try:
            img = picture.find_element(By.XPATH, ".//img") # get the <img> inside the picture
            alt = (img.get_attribute("alt") or "") #  read alt text safely
            alt_norm = " ".join(alt.split()).lower() # normalize whitespace and lowercase
            print(f"Picture {i+1} img alt: '{alt}'")

            if "front view" in alt_norm:
                print(f"Found front view in picture {i+1}")
                sources = picture.find_elements(By.XPATH, ".//source[@srcset]") # get <source> nodes with srcset
                for s in sources:
                    url = _largest_from_srcset((s.get_attribute("srcset") or "").strip())
                    if url:
                        print(f"Extracted URL from srcset: {url[:50]}...")
                        return url
                    
                src = img.get_attribute("src") # fallback to the img src
                if src and "transparent-background.png" not in src: # ensure it’s a real image
                    print(f"Using img src: {src[:50]}...")
                    return src
            if alt_norm.startswith("rear view"):
                rear_count += 1
                if first_rear_url is None:
                    sources = picture.find_elements(By.XPATH, ".//source[@srcset]")
                    for s in sources:
                        url = _largest_from_srcset((s.get_attribute("srcset") or "").strip())
                        if url:
                            first_rear_url = url
                            break
                    if first_rear_url is None:
                        src = img.get_attribute("src")
                        if src and "transparent-background.png" not in src:
                            first_rear_url = src
            # if no front view is found but two or more images have alt text starting with “rear view”, it returns the first rear view image (assumed mislabeled).
        except Exception:
            continue
    if rear_count >= 2 and first_rear_url:
        print("Falling back to first rear view image")
        return first_rear_url
    return None


def _front_view_from_main(driver) -> str | None: # check the main-content section for a front view
    try:
        img = driver.find_element(
            By.XPATH,
            "//div[contains(@class,'product-detail-view__main-content')]"
            "//img[contains(@class,'media-image__image')]",
        )
        alt = (img.get_attribute("alt") or "")
        alt_norm = " ".join(alt.split()).lower()
        if "front view" in alt_norm:
            src = img.get_attribute("src")
            if src and "transparent-background.png" not in src:
                print("Found front view in main content")
                return src
    except Exception:
        pass
    return None


def _front_view_from_secondary(driver) -> str | None: # check the secondary-content section for a front view
    try:
        img = driver.find_element(
            By.XPATH,
            "//div[contains(@class,'product-detail-view__secondary-content')]"
            "//img[contains(@class,'media-image__image')]",
        ) # grab the first matching image there
        alt = (img.get_attribute("alt") or "")
        alt_norm = " ".join(alt.split()).lower()
        if "front view" in alt_norm:
            src = img.get_attribute("src")
            if src and "transparent-background.png" not in src: # ensure it’s not a placeholder
                print("Found front view in secondary content")
                return src
    except Exception:
        pass
    return None


def get_product_image(driver, timeout=IMAGE_TIMEOUT): # main extractor with timeout
    try:
        # Start at top and check before any scroll.
        driver.execute_script("window.scrollTo(0, 0);") # scroll to top
        sleep(0.5)

        url = _front_view_from_main(driver) # try main section first
        if url:
            return url # return if found

        url = _front_view_from_secondary(driver) # try secondary section next
        if url:
            return url # return if found

        pictures = _find_pictures(driver, timeout=3)# try extra-images at top with short wait
        if pictures:
            print(f"Found {len(pictures)} picture elements to check")
            url = _front_view_from_pictures(pictures) # scan for front view
            if url:
                return url

        driver.execute_script("window.scrollTo(0, document.body.scrollHeight/2);") # scroll halfway down
        sleep(1)
        url = _front_view_from_main(driver) # re-check main section
        if url:
            return url

        url = _front_view_from_secondary(driver) # re-check secondary section.
        if url:
            return url

        pictures = _find_pictures(driver, timeout=timeout) # re-check extra-images
        if pictures:
            print(f"Found {len(pictures)} picture elements to check")
            url = _front_view_from_pictures(pictures)
            if url:
                return url

        for _ in range(4): # up to 4 more scroll attempts
            driver.execute_script("window.scrollBy(0, window.innerHeight/2);")
            sleep(1)
            url = _front_view_from_main(driver) # check main again
            if url:
                return url

            url = _front_view_from_secondary(driver) # check secondary again
            if url:
                return url

            pictures = _find_pictures(driver, timeout=timeout)
            if pictures:
                print(f"Found {len(pictures)} picture elements to check")
                url = _front_view_from_pictures(pictures) # final extra-images check
                if url:
                    return url

        print("No front view found")
        return "Not available"
    except Exception as e:
        print(f"Product image extraction error: {e}")
        return "Not available"


def main() -> None:
    with open(INPUT_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise RuntimeError("CSV has no header row")
        rows = list(reader)

    driver = create_driver()
    driver.maximize_window()
    driver.set_page_load_timeout(45)
    updated = 0

    try:
        for i, row in enumerate(rows, 1):
            if not _is_missing_image(row.get("image_url", "")):
                continue

            product_url = (row.get("product_url") or "").strip()
            if not product_url:
                print(f"Row {i}: missing product_url, skipping")
                continue

            print(f"Row {i}: retrying image for {product_url}")
            for attempt in range(3):
                try:
                    driver.get(product_url)
                    sleep(1)
                    _accept_cookies(driver)
                    break
                except (TimeoutException, WebDriverException) as e:
                    print(f"Row {i}: load failed (attempt {attempt + 1}/3): {e}")
                    if attempt == 2:
                        raise
                    sleep(2)

            image_url = get_product_image(driver)
            if image_url != "Not available":
                row["image_url"] = image_url
                updated += 1
                print(f"Row {i}: updated image_url ✅")
            else:
                print(f"Row {i}: still not available ❌")
    finally:
        driver.quit()

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Done. Updated {updated} rows.")
    print(f"Wrote {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
