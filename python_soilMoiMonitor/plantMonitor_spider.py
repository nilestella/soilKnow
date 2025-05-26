import os
import time
from datetime import datetime

import mysql.connector
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

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

plants = {
    '大豆': 'http://nmc.cn/publish/agro/information/soybean.html',
    '油菜': 'http://nmc.cn/publish/agro/information/oilseedrape.html',
    '马铃薯': 'http://nmc.cn/publish/agro/information/potato.html',
    '夏玉米': 'http://nmc.cn/publish/agro/information/summer-corn.html',
    '冬小麦': 'http://nmc.cn/publish/agro/information/winter-wheat.html',
    '春小麦': 'http://nmc.cn/publish/agro/information/spring-wheat.html',
    '棉花': 'http://nmc.cn/publish/agro/information/cotton.html',
    '早稻': 'http://nmc.cn/publish/agro/information/earlyrice.html',
    '春玉米': 'http://nmc.cn/publish/agro/information/spring-corn.html',
    '一季稻':'http://nmc.cn/publish/agro/information/rice-quarter.html'

}

for plant, url in plants.items():
    print(f"Processing plant: {plant}")
    driver.get(url)
    time.sleep(3)
    today = datetime.now().strftime("%m/%d")
    time_elements = driver.find_elements(By.CSS_SELECTOR, '.col-xs-12.time')

    for element in time_elements:
        img_url = element.get_attribute('data-img')
        timestamp = element.find_element(By.CSS_SELECTOR, 'div').text.strip()
        # if today in timestamp:#加入条件判断，即可只获取当天数据；删去，子内容取消一次缩进，即可获取近8天数据
        print(f"Plant: {plant}, Time: {timestamp}, Image URL: {img_url}")
        try:
            conn = mysql.connector.connect(
                host="47.111.98.49",
                user='root',
                password='123456',
                database='farmland202504'
            )
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM plantmonitor WHERE imgSrc = %s", (img_url,))
            if cursor.fetchone()[0] == 0:
                cursor.execute("""
                               INSERT INTO plantmonitor (plant, time, imgSrc)
                               VALUES (%s, %s, %s)
                               """, (plant, timestamp, img_url))
                conn.commit()
        except Exception as e:
            print(f"Error occurred: {e}")
        finally:
            if 'conn' in locals() and conn.is_connected():
                cursor.close()
                conn.close()

driver.quit()