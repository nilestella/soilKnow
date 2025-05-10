import os
import time
import mysql.connector
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from datetime import datetime

options = Options()
options.add_argument('--ignore-certificate-errors')
options.add_argument('--ignore-ssl-errors')
options.add_argument('--disable-extensions')
prefs = {
    "download.prompt_for_download": False,
    "download.directory_upgrade": True,
    "safebrowsing.enabled": True,
    "plugins.always_open_pdf_externally": True
}
options.add_experimental_option("prefs", prefs)

driver_path = 'C:\\Program Files\\Google\\Chrome\\Application\\chromedriver.exe'
service = Service(driver_path)
driver = webdriver.Chrome(service=service, options=options)

url = 'http://www.nmc.cn/publish/agro/disastersmonitoring/Agricultural_Drought_Monitoring.htm'

driver.get(url)
time.sleep(3)

# 获取当前日期
today = datetime.now().strftime("%m/%d")

# 只选择包含当天日期的元素
time_elements = driver.find_elements(By.CSS_SELECTOR, '.col-xs-12.time')
for element in time_elements:
    timestamp = element.find_element(By.CSS_SELECTOR, 'div').text.strip()


    # if today in timestamp:# 只处理当天数据
    img_url = element.get_attribute('data-img')
    print(f"Time: {timestamp}, Image URL: {img_url}")

    try:
        conn = mysql.connector.connect(
            host="**.***.**.**",
            user='root',
            password='*****',
            database='farmland202504'
        )
        cursor = conn.cursor()

        # 先检查是否已存在相同时间的数据
        cursor.execute("SELECT * FROM droughtmonitor WHERE time = %s", (timestamp,))
        if not cursor.fetchone():  # 如果不存在则插入
            cursor.execute("""
                           INSERT INTO droughtmonitor (time, imgSrc)
                           VALUES (%s, %s)
                           """, (timestamp, img_url))
            conn.commit()
            print(f"数据已插入: {timestamp}")
        else:
            print(f"数据已存在，跳过: {timestamp}")

    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()

driver.quit()