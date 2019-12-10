import time

import requests
from bs4 import BeautifulSoup
from selenium import webdriver

from source.utils import connect_to_data_base


def _get_html_source_code(url):
	_html = ''
	resp = requests.get(url)
	if resp.status_code == 200:
		_html = resp.text
	return _html

### because naver datalab page using ajax, we need a brower to execute ajax command
### using Chrome web browser driver
# chrome_options = webdriver.ChromeOptions()
#
# prefs = {"profile.default_content_setting_values.notifications": 2}
# chrome_options.add_experimental_option("prefs", prefs)

# create a new Chrome session
driver = webdriver.Chrome('source/chromedriver/chromedriver_win.exe')
driver.implicitly_wait(30)
driver.get("https://datalab.naver.com/")

cates = ['패션의류', '패션잡화', '화장품/미용', '디지털/가전', '가구/인테리어', '출산/육아', '식품', '스포츠/레저', '생활/건강', '여행/문화', '면세점']

cates_input_option = driver.find_element_by_class_name('select_btn')
options = driver.find_elements_by_class_name('option')
db = connect_to_data_base()
cursor = db.cursor()

for cate in cates:
	### get ajax page of each category above
	for opt in options:
		### naver trigger ajax call by option click, then we need click con disire option value
		if (opt.get_attribute('innerText') == cate):
			driver.execute_script("arguments[0].click();", opt)
			time.sleep(0.5)
			### load html code to beautiful soup
			his_source_code = driver.page_source
			## parser html code
			his_soup = BeautifulSoup(his_source_code, 'html.parser')
			keywords_container = his_soup.find_all('div', class_='keyword_rank')
			for container in keywords_container:
				keyword_span = container.find_all('span',class_='title')
				for keyword in keyword_span:
					# query = 'insert into keywords(keyword,category) values ("'+ keyword.text +'","'+cate+'") on duplicate key update keyword = "'+ keyword.text +'", category= "'+cate+'";'
					query = 'insert into naver_keywords(keyword,category) values ("'+ keyword.text +'","'+cate+'") on duplicate key update insert_date = current_timestamp;'
					try:
						cursor.execute(query)
						db.commit()
					except:
						db.rollback()
						raise Exception('Could not execute the query, check statement or connection to database..')
cursor.close()
db.close()