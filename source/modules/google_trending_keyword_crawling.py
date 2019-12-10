import pandas as pd
from pytrends.request import TrendReq

import MySQLdb as sql


def connect_to_data_base():
	try:
		db = sql.connect(host='164.125.154.217', user='root', passwd='2848', db='locs_keywords',
						 port=3306, charset='utf8')
	except:
		raise ConnectionError('Could not connect to database server, check internet connection and database detail..')
	return db

#
# db = connect_to_data_base()
# cursor = db.cursor()
#
#
#
# # google_trends = TrendReq(hl='ko', geo='kr', timeout=(10,25), proxies=['https://34.203.233.13:80','https://35.201.123.31:880'], retries=2, backoff_factor=0.1)
# google_trends = TrendReq(hl='ko', geo='kr')
# new_keywords = google_trends.trending_searches(pn='south_korea')
# for row in new_keywords.itertuples():
# 	keyword = row[1]
# 	print(keyword)
# 	related = google_trends.suggestions(keyword=keyword)
# 	# related_df = pd.DataFrame(related)
# 	for dict_ in related:
# 		kw = dict_['title']
# 		cate = dict_['type']
# 		query = 'insert into google_keywords(keyword,category) values ("'+ kw +'","'+cate+'") on duplicate key update insert_date = current_timestamp;'
# 		try:
# 			cursor.execute(query)
# 			db.commit()
# 		except Exception as e:
# 			print(e)
# 			db.rollback()
# 			continue
# cursor.close()
# db.close()
#
#
if __name__ == '__main__':

	db = connect_to_data_base()
	cursor = db.cursor()
	i = 0
	while i <= 3000:
		# google_trends = TrendReq(hl='ko', geo='kr', timeout=(10,25), proxies=['https://34.203.233.13:80','https://35.201.123.31:880'], retries=2, backoff_factor=0.1)
		google_trends = TrendReq(hl='ko', geo='kr', timeout=(3, 5))
		i += 1
		query = 'select keyword from google_keywords order by insert_date DESC limit 1000;'
		cursor.execute(query)
		data = cursor.fetchall()
		# data_ = pd.DataFrame(data)
		for row in data:
			keyword = row[0]
			try:
				related = google_trends.suggestions(keyword=keyword)
			except Exception as e:
				print(e)
				continue
			related_df = pd.DataFrame(related)
			for dict_ in related:
				kw = dict_['title']
				cate = dict_['type']
				query = 'insert into google_keywords(keyword,category) values ("' + kw + '","' + cate + '") on duplicate key update insert_date = current_timestamp;'
				try:
					cursor.execute(query)
					db.commit()
					print('success with {}'.format(keyword))
				except Exception as e:
					print(e)
					db.rollback()
					continue
	cursor.close()
	db.close()
