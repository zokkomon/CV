#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 21:03:00 2023

@author: zok
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
import time
import urllib

options = Options()
options.headless = True
driver = webdriver.Firefox()

# Target webpage URL
url = "https://www.pinterest.com/tomoyuen/resume-design/"

# Function to extract and save images
def extract_and_save_images(img_elements):
    src = []
    for img in img_elements:
        src.append(img.get_attribute("src"))
        try:
            for i in range(10):    
                urllib.request.urlretrieve(str(src[i]), "sample_data/resume{}.jpg".format(i))
            print("down")
        except Exception as e:
            print(f"Error downloading {src}: {e}")

# Load the webpage
driver.get(url)

# Wait for images to load
driver.execute_script("window.scrollTo(0,document.body.scrollHeight);")
time.sleep(5)

# Find all image elements
img_elements = driver.find_elements(By.XPATH,"//img[contains(@class, 'XiG ho- sLG zI7 iyn Hsu')]")

# Extract and save images
extract_and_save_images(img_elements)

# Close the browser
driver.quit()
