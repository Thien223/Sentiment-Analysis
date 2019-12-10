#-*- coding: utf-8 -*-
import os
import sys
import urllib.request
### 검색어의 ratio(ranking) 출력할뿐임

client_id = "6YcBbXhQjc5u5KSBBy71"
client_secret = "NBTn1gh4c0"

url = "https://openapi.naver.com/v1/datalab/search"
body = "{\"startDate\":\"2017-01-01\",\"endDate\":\"2020-04-30\",\"timeUnit\":\"month\",\"keywordGroups\":[{\"groupName\":\"패션의류\"},{\"groupName\":\"패션잡화\"}],\"device\":\"pc\",\"ages\":[\"1\",\"2\"],\"gender\":\"f\"}"

request = urllib.request.Request(url)
request.add_header("X-Naver-Client-Id",client_id)
request.add_header("X-Naver-Client-Secret",client_secret)
request.add_header("Content-Type","application/json")
response = urllib.request.urlopen(request, data=body.encode("utf-8"))
rescode = response.getcode()
if(rescode==200):
    response_body = response.read()
    print(response_body.decode('utf-8'))
else:
    print("Error Code:" + rescode)