import os
import time
from selenium import webdriver


class SeleniumConnect:

    def __init__(self,username='Gigante_',password='mypassssl3'):
        self.USERNAME = username
        self.PASSWORD = password
        self.CHROMEDRIVER_PATH = 'chromedriver.exe' # Insert the path of chromedriver (to be downloaded from "https://sites.google.com/a/chromium.org/chromedriver/downloads")
        prefs={}
        prefs["download.default_directory"]=os.path.join(os.getcwd(),"weekend")
        self.options = webdriver.ChromeOptions()
        self.options.add_experimental_option("prefs", prefs)

    def download_files(self):
        driver = webdriver.Chrome(chrome_options=self.options,executable_path=self.CHROMEDRIVER_PATH)
        driver.get('https://footystats.org/login');
        time.sleep(5) # Let the user actually see something!
        search_box = driver.find_element_by_name('username')
        search_box.send_keys(self.USERNAME)
        search_box = driver.find_element_by_name('password')

        search_box.send_keys(self.PASSWORD)

        driver.find_element_by_id('register_account').submit()

        time.sleep(5) # Let the user actually see something!

        driver.get('https://footystats.org/c-dl.php?type=matches&comp=4889');  # Sample download 1
        driver.get('https://footystats.org/c-dl.php?type=matches&comp=4673')
        driver.get('https://footystats.org/c-dl.php?type=matches&comp=4759')

        time.sleep(5)

        driver.quit()