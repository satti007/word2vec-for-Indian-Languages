import re
import os
import requests
from bs4 import BeautifulSoup

# The below code is for crawling data from "www.andhrajyothy.com"
# Each article has very simple url : "http://www.andhrajyothy.com/artical?SID=id"
# Crawling all artciles (1000<=id<=620000) which are on the website as of Aug 2018
################################################################################################################################
def doCrawl(start_id,end_id):
    url_count  = 0                              # number of urls crawled      -- to check the progress
    line_count = 0                              # Num of lines (split by '.') -- just to get an idea of how much content is scrapped
    word_count = 0                              # Num of words (split by ' ') -- just to get an idea of how much content is scrapped
    save_dir = '../../data/corpora/andhrajyothy/articles/{}k_to_{}k/'.format(start_id/1000,end_id/1000) # directory to save articles
    for i in range(start_id,end_id+1):
        url  = "http://www.andhrajyothy.com/artical?SID={}".format(i)
        page = requests.get(url)
        soup = BeautifulSoup(page.content, 'html.parser')
        content = soup.find_all(id="ContentPlaceHolder1_lblStoryDetails") # A <p> element with this id has all the content of article
        if len(content) == 1:
            text = content[0].text.encode('utf-8')
            if len(text) > 0:
                lines = text.split('.')
                line_count += len(lines)
                word_count += len(text.split(' '))
                url_file = open(save_dir+'{}.txt'.format(i),'w')          # Opening a file named id.txt....
                url_file.write(text)                                      # ... to write the article contnet
                url_file.close()
        
        url_count +=1
        print ('{}th crawled, URLS parsing done:{}, NUM_OF_SEN:{}, NUM_OF_WORDS:{}'.format(i,url_count,line_count,word_count))
        # if url_count == 10:
        #     break

def crawlAJ():
    start = 1000                # First article id 
    end   = 620000              # Last  article id as of Aug 2018 
    doCrawl(start,100000)       # call each of these(4) fuctions one at a time
    # doCrawl(100000,200000)    # each to monitor & check progress, and also...  
    # doCrawl(200000,400000)    # ... crawling time of each article increases significantly.. 
    # doCrawl(400000,end)       # ... if you do the crawling for a long time continuously 

# crawlAJ()
#######################################################################################################################################


# The below code is for crawling data from "http://telugu.webdunia.com"
#######################################################################################################################################
home_url = 'http://telugu.webdunia.com/'                # base directorty for saving files
home_dir = '../../data/corpora/webdunia/'               # home_page of the website 

# Given an article url, scrape the content of that article and save it in a file
def getConent(article_urls,save_dir,start,end):
    url_count  = 0
    line_count = 0
    word_count = 0
    crawl_list = article_urls[start:end]
    for idx,url in enumerate(crawl_list):
        addr    = home_url + url[1:-1]
        page    = requests.get(addr)
        soup    = BeautifulSoup(page.content, 'html.parser')
        try:
            content = soup.find(itemprop="articleBody")
            if len(content) != 0:
                text = content.get_text()
                if len(text) > 0:
                    lines = text.split('.')
                    line_count += len(lines)
                    word_count += len(text.split(' '))
                    url_file = open(save_dir+'articles/{}k_to_{}k/{}.txt'.format(int(start/1000),int(end/1000),idx+start),'w')
                    url_file.write(text)
                    url_file.close()
                    url_count  += 1 
            print ('{}th crawled, URLS parsing done:{}, NUM_OF_SEN:{}, NUM_OF_WORDS:{}'.format(idx+start,url_count,line_count,word_count))
        except:
            print(addr)
            print("##ERROR##")
            continue
        # if url_count == 10:
        #     break

def crawlContent(home_url,save_dir,file):
    article_urls = open(save_dir+file,'r').readlines()
    # getConent(article_urls,save_dir,0,50000)   # call each of these fuctions one at a time
    # getConent(article_urls,save_dir,50000,100000)  # each to monitor & check progress, and also...  
    getConent(article_urls,save_dir,100000,150000) # ... crawling time of each article increases significantly.. 
    # getConent(article_urls,save_dir,150000,231565) # ... if you do the crawling for a long time continuously 


# A function to sort the lines in a file
def sortFile(file_name):
    lines = open(home_dir+file_name,'r').readlines()
    lines.sort()
    doc   = open(home_dir+file_name,'w')
    for line in lines:
        doc.write(line)
    
    doc.close()

# A function to make sure all the elements in the given dict are distinct
def updateDict(list_,dict_,file):
    for l in list_:
        if l not in dict_:
            dict_[l] = 1
            file.write(l+'\n')
    return dict_

# For a give section,check next page exists
def checkNxt(soup,count):
    if count == 1:
        return 1
    else:
        try:
            bts = len(soup.find('div', class_="btnBlock").find_all('a'))
        except:
            return 0
        if bts != 2:
            return 0
        return 1

# given a section name, return all the articles urls present under that section
def getArticleUrls(url,article_urls,file):
    count     = 1
    next_bt   = 1
    sec_count = 0
    init_len  = len(article_urls)
    while (next_bt):
        urls = []
        link = url+'/'+str(count)
        page = requests.get(link)
        soup = BeautifulSoup(page.content, 'html.parser')
        all_links =soup.find_all('a')
        for l in all_links:
            h = l.get('href').strip()
            if '/article/' in h:
                urls.append(h)
                # print (h)        
        article_urls = updateDict(urls,article_urls,file)
        sec_count    = len(article_urls) - init_len
        print('page_count: {}, sec_articles:{}, total_articles: {}'.format(count,sec_count,len(article_urls)))
        
        next_bt  = checkNxt(soup,count)
        count    = count + 1
    
    return article_urls

# given a class name, return all the section urls present in the div with that class
def getSectionUrls(soup,class_name,section_urls,file):
    urls = []
    for divs in soup.find_all('div', class_=class_name):
        for a in divs.find_all('a'):
            if len(a.get('href')) > 0:
                urls.append(a.get('href'))
    
    section_urls = updateDict(urls,section_urls,file)
    
    return section_urls

def crawlWD():
    page = requests.get(home_url)
    soup = BeautifulSoup(page.content, 'html.parser')
    
    section_urls     = {}
    section_urls_doc = open(home_dir+'section_urls.txt','w')
    print('Getting section_urls...')
    section_urls = getSectionUrls(soup,'sectionFooter fLt w145 ',section_urls,section_urls_doc)
    section_urls = getSectionUrls(soup,'sectionFooter fLt w145 bdrRNone',section_urls,section_urls_doc)
    print('The total num of section_urls: {}'.format(len(section_urls)))
    section_urls_doc.close()
    sortFile('section_urls.txt')
    
    article_urls     = {}
    article_urls_doc = open(home_dir+'article_urls.txt','w')
    for idx, url in enumerate(section_urls):
        print('Getting article_urls from section:{}, {}...'.format(idx,url.split('/')[-1]))
        article_urls = getArticleUrls(url,article_urls,article_urls_doc)
    
    article_urls_doc.close()
    sortFile('article_urls.txt')

# crawlWD()                                          # Collecting all the articles urls under the webdunia domian
crawlContent(home_url,home_dir,'article_urls.txt') # After collecting all the articles urls, scrape the content in each of them
#######################################################################################################################################



'''
def getArticleUrls(url,article_urls,file):
    count     = 1
    next_bt   = 1
    sec_count = 0
    init_len  = len(article_urls)
    while (next_bt):
        urls = []
        link = url+'/'+str(count)
        page = requests.get(link)
        soup = BeautifulSoup(page.content, 'html.parser')
        try:
            urls.append(soup.find('h1', class_="headMain").find('a').get('href'))
        except:
            print ('No headMain')
        
        try:
            for li in soup.find('ul' , class_="itemList").find_all('a'):
               urls.append(li.get('href'))
        except:
            print ('No itemList')
                
        try:
            head = soup.find_all('h2', class_="headSub")
            for h in head:
                urls.append(h.find('a').get('href'))
        except:
            print ('No headSub')
        
        article_urls = updateDict(urls,article_urls,file)
        sec_count    = len(article_urls) - init_len
        print('page_count: {}, sec_articles:{}, total_articles: {}'.format(count,sec_count,len(article_urls)))
        
        next_bt  = checkNxt(soup,count)
        count    = count + 1
    
    return article_urls
'''