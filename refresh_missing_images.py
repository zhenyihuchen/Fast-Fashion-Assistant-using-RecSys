import csv
from time import sleep

from web_scrapping import create_driver, get_product_image


INPUT_CSV = "women_all_categories_data_good.csv"
OUTPUT_CSV = "women_all_categories_data_good_updated.csv"


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


def main() -> None:
    with open(INPUT_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise RuntimeError("CSV has no header row")
        rows = list(reader)

    driver = create_driver()
    driver.maximize_window()
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
            driver.get(product_url)
            sleep(1)
            _accept_cookies(driver)

            image_url = get_product_image(driver)
            if image_url != "Not available":
                row["image_url"] = image_url
                updated += 1
                print(f"Row {i}: updated image_url")
            else:
                print(f"Row {i}: still not available")
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
