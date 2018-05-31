#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 09:37:11 2018

@author: LEIHAO
"""

import sqlite3
import numpy as np
#from scipy.sparse import csr_matrix, save_npz
from nltk.corpus import wordnet
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
import time
import TXTnlp
import matplotlib.pyplot as plt
from matplotlib.dates import datestr2num
import NYSE_tradingdays as tday
import datetime



directory='/Users/leihao/Downloads/'
sqlite_file=directory+'nasdaq.db'
conn=sqlite3.connect(sqlite_file)
c=conn.cursor()
titl=c.execute("""Select DATE(date) AS art_date, COUNT(title) AS art_cnt_per_day 
               FROM articles GROUP BY art_date""")
titl_tuple=titl.fetchall()
conn.close()
# remove the 1st count which is 0
titl_tuple.pop(0)
# remove 2 outliers, which have index 618 and 619
titl_tuple.pop(618)
titl_tuple.pop(619)

#data for trading days
tdays = list(tday.NYSE_tradingdays(datetime.datetime(2006,4,2),datetime.datetime(2017,4,1)))
trad_dates=[item[0] for item in titl_tuple if datetime.datetime.strptime(item[0],'%Y-%m-%d') in tdays]
trad_cnts= [item[1] for item in titl_tuple if datetime.datetime.strptime(item[0],'%Y-%m-%d') in tdays]
trad_num_dates=[datestr2num(item) for item in trad_dates]

#remove outliers
trad_cnts.pop(467)
trad_num_dates.pop(467)
trad_dates.pop(467)

#data for non-trading days
notrad_dates=[item[0] for item in titl_tuple if datetime.datetime.strptime(item[0],'%Y-%m-%d') not in tdays]
notrad_cnts= [item[1] for item in titl_tuple if datetime.datetime.strptime(item[0],'%Y-%m-%d') not in tdays]
notrad_num_dates=[datestr2num(item) for item in notrad_dates]


plt.plot_date(trad_num_dates, trad_cnts, 'b.')
plt.plot_date(notrad_num_dates, notrad_cnts, 'r.')
plt.ylabel('Number of Articles')
plt.xlabel('Daily')





titl_dates=[item[0] for item in titl_tuple]
new_dates=[datestr2num(item) for item in titl_dates]
titl_cnts=[item[1] for item in titl_tuple]
#  outlier_remove()
outlier_ind1=np.argmax(titl_cnts)
titl_cnts.pop(outlier_ind1)
titl_dates.pop(outlier_ind1)
new_dates.pop(outlier_ind1)

outlier_ind2=np.argmax(titl_cnts)
titl_cnts.pop(outlier_ind2)
titl_dates.pop(outlier_ind2)
new_dates.pop(outlier_ind2)

plt.plot_date(new_dates, titl_cnts, 'bo')

# -----------------
# Get the MPI rank
# -----------------
#comm = MPI.COMM_WORLD
#size = comm.Get_size()
#rank = comm.Get_rank()
cnt=[32,602]
for year in range(2010,2018):
    start_date, end_date=str(year)+'-01-01', str(year)+'-12-31' #if rank==0 else ('2016-01-03', '2016-01-04')
    conn=sqlite3.connect(sqlite_file)
    c=conn.cursor()
    titl=c.execute("SELECT title FROM articles WHERE date BETWEEN ? AND ?", (start_date, end_date))
    titl_tuple=titl.fetchall()
    conn.close()
    cnt.append(len(titl_tuple))
    
    
import matplotlib.pyplot as plt
plt.plot(range(2008,2018),cnt,'-o')
#plt.axis([2008,2017,0,200000])
#plt.show()


from datetime import datetime
#import random
 
year = np.random.randint(2009, 2016, size=10)
month =np.random.randint(1, 12,size=10)
pre_month=list(zip(year,month))
post_month=[str(item1)+'-'+str(item2) for item1, item2 in pre_month]

day = np.random.randint(1, 28,100)
pre_date=list(zip(year,month,day))
rand_date = [datetime(item[0],item[1],item[2]) for item in pre_date]

test_date=tuple(item.strftime('%Y-%m-%d') for item in rand_date[:2])
titl_tuple=[]
conn=sqlite3.connect(sqlite_file)
c=conn.cursor()
for date in test_date:
    titl=c.execute("SELECT article FROM articles WHERE date BETWEEN ? AND ?",('2012-08-04',))
    titl_tuple2=titl.fetchall()
conn.close()

year = np.random.randint(2009, 2016, size=12)
month =np.random.randint(1, 12,size=12)
pre_month=list(zip(year,month))

post_month2=[]
for item1, item2 in pre_month:
    #if len(item2)==1:
    dat=str(item1)+'-0'+str(item2) if item2<10 else str(item1)+'-'+str(item2)
    post_month2.append(dat)    
#post_month2=[str(item1)+'-'+str(item2) for item1, item2 in pre_month if len(item2)==2 else str(item1)+'-0'+str(item2) ]

def word_cnt(article_tuple):
    wd_cnt=[]
    for item in article_tuple:
            txt = TXTnlp.TextBlob(item[0])
            ### Tokens and POS tags
            txttok = TXTnlp.token(txt)
            wd_cnt.append(len(txttok))
    return wd_cnt
all_wd_cnt=[]
conn=sqlite3.connect(sqlite_file)
c=conn.cursor()
for mth in post_month2:
    #mth='2016-01'
    start_date, end_date=mth+'-01', mth+'-31'
    titl=c.execute("SELECT article FROM articles WHERE date BETWEEN ? AND ?",(start_date,end_date))
    titl_tuple2=titl.fetchall()
    all_wd_cnt += word_cnt(titl_tuple2)
conn.close()


import matplotlib.pyplot as plt




#count number of articles for preliminary result
start_date, end_date='2016-01-01', '2016-06-31' #if rank==0 else ('2016-01-03', '2016-01-04')
conn=sqlite3.connect(sqlite_file)
c=conn.cursor()
art_cnt=c.execute("SELECT title FROM articles WHERE date BETWEEN ? AND ?", (start_date, end_date))
art_cnt_tuple=art_cnt.fetchall()
conn.close()
