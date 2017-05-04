import re
import os
from collections import Counter
import datetime

case2judge = pd.read_pickle('../rawdata/cluster2songername.pkl')
casefreq = pd.read_pickle('../rawdata/case_level/casefreqs-akd.pkl')

judges = np.unique(list(case2judge.values()))
judge2case = [[v,k] for k, v in case2judge.items()]


## return opinion year dicionary
p = re.compile(r'\d{1,2}?  \d{4}?')
case2year = {}
folders = glob('../rawdata/text/*')
for folder in folders:
    members = glob(folder + '/*txt')
    print(folder,len(members))
    if len(members) == 0:
        continue
    for fname in members:
        if not fname.endswith('txt'):
            continue
        caseid = fname.split('/')[-1][:-4]
        text = open(fname).read()    
        normtext = re.sub('[^a-z0-9 ]',' ',text.lower())
        years = p.findall(normtext)
        proyear = []
        for year in years:
            try :
                date = datetime.datetime.strptime(year,"%d  %Y")
                if date.year > 1900 and date.year < 2017:
                    proyear.append(date.year)
            except ValueError as err:
                continue
        if len(proyear) > 0:
            case2year[caseid] = np.max(proyear)
pd.to_pickle(case2year,'datasets/case2year.pkl')

## return the judge-year opnions dictionary
judge2year_case = {}
for judge in judges:
    cases = [v for k,v in judge2case if k == judge]
    year2case = {}
    for case in cases:
        try:
            year = case2year[str(case)]
            if year2case.get(year) == None:
                year2case[year] = []
            else:
                year2case[year].append(case)
        except KeyError as err:
            continue
    judge2year_case[judge] = year2case 
pd.to_pickle(judge2year_case,'datasets/judge2year_case.pkl')
