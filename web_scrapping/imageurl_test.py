import pandas as pd
from pathlib import Path
import requests 
from PIL import Image
from io import BytesIO

base_dir = Path(__file__).resolve().parent
data_dir = base_dir.parent / "data" / "web_scrapping_runs"
df = pd.read_csv(data_dir / "women_all_categories_data.csv")
image_url = df["product_image"][1]

response = requests.get(image_url) #Load the image temporarily just ot view it. it doeesnt download it
image = Image.open(BytesIO(response.content))
image.show()
