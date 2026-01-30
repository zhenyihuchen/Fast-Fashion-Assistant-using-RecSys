import pandas as pd
import requests 
from PIL import Image
from io import BytesIO

df = pd.read_csv("women_all_categories_data.csv")
image_url = df["product_image"][1]

response = requests.get(image_url) #Load the image temporarily just ot view it. it doeesnt download it
image = Image.open(BytesIO(response.content))
image.show()