import re
import string
import pprint
from collections import OrderedDict

frequency = {}
document_text = open(r'C:\Users\golde\OneDrive\Desktop\SAC Resume Docs\TJD.txt', 'r')
text_string = document_text.read().lower()
match_pattern = re.findall(r'\b[a-z]{3,15}\b', text_string)
 
for word in match_pattern:
    count = frequency.get(word,0)
    frequency[word] = count + 1


###sorted_list=sorted(frequency.items(), key=lambda x: x[0])

sorted_frequency = {k: frequency[k] for k in sorted(frequency)}

##.split("[\\W&&[^\"']]")
pp = pprint.PrettyPrinter(indent=4)

for words in sorted_frequency.keys():
    if sorted_frequency[words] > 5:
        pp.pprint(sorted_frequency)

##            words, sorted_frequency[words], indent=4)
##

frequency=OrderedDict(sorted(frequency.items()))
     
frequency_list = frequency.keys()
 
for words in frequency_list:
    if frequency[words] > 5:
        print(words, frequency[words])
##    else:
##        print("complete")

sorted_dict = {k: disordered[k] for k in sorted(disordered)}

pp = pprint.PrettyPrinter(indent=4)
pp.pprint(stuff)

##import json
## import pprint
## from urllib.request import urlopen
## with urlopen('https://pypi.org/pypi/sampleproject/json') as resp:
##     project_info = json.load(resp)['info']
