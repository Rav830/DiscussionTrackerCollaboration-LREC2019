
import config
import parse_transcript as pt
import os
import pandas as pd
import numpy as np
import nltk
import json
from unidecode import unidecode
from collections import Counter
from nltk.corpus import stopwords

nltk.download('punkt')

stopWords = set(stopwords.words("english"))
keep_words = set([])

speakerList = set([])
speakerRemove = set([])
def extract_data(dataPath, x, y, group, groupVal, transcript, transVal):
    #this convert transcript is exactly what was given to me in 
    data = pt.convert_transcript(dataPath)
    
    #need to do a data compression pass
    rows = []
    for index, row in data.iterrows():
        if(row['Turn'] == ''):
            rows[-1]['Talk2'] += ' ' + row['Talk2']
        else:
            rows.append(row)
    #now filter out all of the teacher talks
    
    rowsKeep = []
    for row in rows:
        #first make sure something is being said
        if(row['Talk2'].strip() == '' and row['Collaboration Code'] == ''):
            #if nothing is being said and there is no collab code
            #None
            speakerRemove.add(row['Student'])
        else:
            #check if the speaker is not a teacher or adult in some way
            clean = row['Student'].lower().strip()
            #boolean logic 
            #teacher is not in the name (eg teacher or St. Teacher) and
            #not of the form "t" and
            #not of the form tX where X is any digit and 
            #not with the name adult and
            #not of the form T 'stuff'
            if(('teacher' not in clean) and 
                (clean != "t") and 
                (not (clean.startswith('t') and clean[1].isdigit())) and 
                ("adult" not in clean) and 
                (not (clean.startswith('t') and '?' in clean))):
                #if they are not a teacher check that there is a collaboration code present maybe
                rowsKeep.append(row)
            else:
                #print(dataPath)
                #print(row, end ="\n+++++++=\n")
                #input()
                speakerRemove.add(row['Student'])
    #    if(row['Collaboration Code'] != ''):
   #         rowsKeep.append(row)
    #    else:
            
        
            #this could be a teacher or student
            
            
            
         #   else:
                #this should be not be considered let's check appropriately
    
    #print("++++++++++++++++++++++++++++++++++++++++++")
    
    #now append to the running lists 
    for row in rowsKeep:
        speakerList.add(row['Student'])
        
        if(row['Collaboration Code'] == ''):
            print(dataPath)
            print(row, end ="\n+++++++=\n")
        x.append(row['Talk2'])
        y.append(row['Collaboration Code'])
        group.append(groupVal)
        transcript.append(transVal)
        
def compress_parens(x, left, right):
    while x.count(left) > 0:
        j = x.index(left)
        closeIdx = x.index(right, j)
        temp = ' '.join(x[j:closeIdx+1])
        x[j] = temp
        del x[j+1:closeIdx+1]
       
counting = Counter() 
def retrieveVal(cvSplits, path):
	for k in cvSplits:
		for test in cvSplits[k]['test']:
			if(test in path):
				counting[k] += 1
				return int(k)
	
	print("We did not find this path in the test sets and that is a fatal error")
	print(path+"\n\n\n\n")
	print(cvSplits)
	raise SystemExit

def pre_process_data(dataPathList, isTrain = True, cvSplits=None):
    
    x = []
    y = []
    group = []
    transcript = []
    print("Reading in files...")
    groupVal = 0
    transVal = 0
    for p in dataPathList:
        #print(p)
        transVal += 1
        if(cvSplits):
        	# the groupval will be the fold that this file name is in the test set
        	groupVal = retrieveVal(cvSplits, p)
        else:
        	groupVal += 1
        extract_data(p, x, y, group, groupVal, transcript, transVal)
    
    #print(Counter(y))
    #print(len(y))
    #print(speakerList)
    #print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    #print(speakerRemove)
    #print("Done Reading in files")
    #raise SystemExit
    print("PreProcessing...")
    for i in range(len(x)):
        # ascii normalize the turn
        x[i] = unidecode(x[i])
        #tokenize the text
        x[i] = nltk.word_tokenize(x[i].lower())
        
        #remove stop words
        x[i] = [p for p in x[i] if p not in stopWords]
        
        
        #compress the contents between parenthesis
        #compress_parens(x[i], '(', ')')
        #compress_parens(x[i], '{', '}')
        #compress_parens(x[i], '[', ']')

    #if this set of data is part of the training data we create the unknown word list
    #following jurafsky's approach of single words being unknown
    if(isTrain):
        flat_word_val = [item for sublist in x for item in sublist]
        countObj = Counter(flat_word_val)
        #countObj.update()
        
        for key in countObj.keys():
            if(countObj[key] != 1):
                keep_words.add(key)

    if(len(keep_words) == 0):
        print("Need to run this in train mode first")
        raise SystemExit

    #now do an unknown word filter pass
    for i in range(len(x)):
        unkn_filtered = []
        for word in x[i]:
            if(word not in keep_words):
                unkn_filtered.append('<UKN>')
            else:
                unkn_filtered.append(word)
        x[i] = unkn_filtered

    return (x,y,group,transcript)

def build_vocab(ls, posTagged = False):
    vocab = set([])
    for doc in ls:
        for word in doc:
            if(posTagged):
                vocab.add(word[0])
            else:
                vocab.add(word)
    return vocab

def count_words(ls, vocab, posTagged=False):
    #prep dict
    counts = {}
    #prefill this list with zeroes
    for word in vocab:
        counts[word] = [0] * len(ls)
    
    #go through each text, count the words,
    for i in range(len(ls)):
        #print("%d/%d" %(i,len(df['text'])))
        #pdIndex = df.index[i]
        if(posTagged):
            toCount = [x[0] for x in ls[i]]
        else:
            toCount = ls[i]
        word_counts = Counter(toCount)
        
        #now go through each word in counts and check in the Counter object
        for vocab_word in word_counts.keys():
            if(vocab_word in vocab):
                counts[vocab_word][i] = word_counts[vocab_word]
    
    return counts

def build_vectored_df(x, y, yName, group, transcript):
    tempDF = count_words(x, build_vocab(x))
    tempDF[yName] = y
    tempDF['|-group'] = group
    tempDF['|-transcript'] = transcript
    
    return pd.DataFrame(data = tempDF)

def y_conversion(a):
    if a == '' or a == 'Non' or a == '??W??':
        return 0
    elif a == 'challenge':
        return 3
    elif a == 'agree':
        return 2
    elif a == 'new-idea':
        return 1
    elif a == 'extension':
        return 4
    else:
        print(a)
        return -1
            
conversion = ['Non', 'new-idea', 'agree', 'challenge', 'extension']
def conversion_y(a):
    return conversion[a]


def handle_labels(y):
    #convert all empty to non and then number the rest
    
    return [ y_conversion(x) for x in y]


def filterNon(x, y, group, transcript):
	for i in range(len(y)):
		if(y[i] == y_conversion('Non')):
			x[i] = None
			y[i] = None
			group[i] = None
			transcript[i] = None

	return [i for i in x if i != None], [i for i in y if i != None], [i for i in group if i != None], [i for i in transcript if i != None]

def gen_data_and_df(fList, removeNon = False, isTrain = True, cvSplits = None):
    
    x, y, group, transcript = pre_process_data(fList, isTrain, cvSplits)
    y = handle_labels(y)
    
    if(removeNon):
    	x, y, group, transcript = filterNon(x, y, group, transcript)
    
    df = build_vectored_df(x,y,"|-Collab", group, transcript)
    #print("Counting: " + str(counting))
    return (x,y,df)


def df_transform(dict_form):
	
	for key in dict_form.keys():
		if(key not in ["|-Collab", "|-group", "|-transcript"]):
			numDocs = len(dict_form[key])
			numFreq = len([x for x in dict_form[key]  if x != 0  ])
			idf = np.log(numDocs/numFreq)
			for i in range(numDocs):
				dict_form[key][i] *= idf 

	return dict_form
def tf_transform(dict_form):
	#print(dict_form)
	
	#compute sum for each row
	sum_list = [0]*len(dict_form['|-Collab'])
	for key in dict_form.keys():
		if(key not in ["|-Collab", "|-group", "|-transcript"]):
			for i in range(len(dict_form[key])):
				sum_list[i] += dict_form[key][i]
				if(sum_list[i] == 0):
					sum_list[i] = 1
	
	for key in dict_form.keys():
		if(key not in ["|-Collab", "|-group", "|-transcript"]):
			for i in range(len(dict_form[key])):
				dict_form[key][i] = dict_form[key][i]/sum_list[i] 
	return dict_form

def to_tf_idf(df):
	dict_form = df.to_dict('list')
	dict_form = tf_transform(dict_form)

	dict_form = df_transform(dict_form)
	
	retval = pd.DataFrame(data = dict_form)
	return retval

# df['Collab']
#gDF = df['group']

if __name__ == "__main__":
    
    dataPath = "../Data/EAGER/"
    di = [dataPath+x for x in os.listdir(dataPath) if x.endswith(".xlsx") or x.endswith(".xls")]

    x,y,df = gen_data_and_df(di, removeNon = False)
    print(set(df['|-Collab']))
    print(Counter(df['|-Collab']))
    
    #print(x[26])
    #for idx, i in enumerate(x):
     #   print(str(idx) + str(i))
    
    print(df)

    print(df['group'])
    
    print(df['transcript'])























