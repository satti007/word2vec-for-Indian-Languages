import re
import os
import requests
from bs4 import BeautifulSoup

# Data crawling
url_count  = 0                      # number of urls crawled
line_count = 0                      # Num of lines (split by '.') -- just to get an idea
word_count = 0                      # Num of words (split by ' ') -- just to get an idea
start_url_id = 1000
end_url_id   = 620000
for i in range(start_url_id,end_url_id+1):
    url  = "http://www.andhrajyothy.com/artical?SID={}".format(i)
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    content = soup.find_all(id="ContentPlaceHolder1_lblStoryDetails")
    if len(content) == 1:
        text = content[0].text.encode('utf-8')
        if len(text) > 0:
            lines = text.split('.')
            line_count += len(lines)
            word_count += len(text.split(' '))
            url_file = open('corpora/{}.txt'.format(i),'w',0)
            url_file.write(text)
            url_file.close()
    
    url_count +=1
    print ('{}th crawled, URLS parsing done:{}, NUM_OF_SEN:{}, NUM_OF_WORDS:{}'.format(i,url_count,line_count,word_count))

all_file.close()