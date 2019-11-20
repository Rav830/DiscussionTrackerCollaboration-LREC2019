#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import config
import json
import os
import pandas as pd
import numpy as np
import re

def collaboration_converter(a):
    #print(a)
    a = a.upper().strip()
    if a == 'I':
        return 'initiation'
    elif a == 'C':
        return 'challenge'
    elif a == 'R':
        return 'rebuttal'
    elif a == 'SE':
        return 'self-extension'
    elif a == 'OE':
        return 'other-extension'
    elif a == 'A':
        return 'agree'
    elif a == 'P':
        return 'probe'
    elif a == 'NON':
        return 'Non'
    elif a == 'N':
        return 'new-idea'
    elif a == 'E':
        return 'extension'
    elif len(a) == 1:
        return '??'+a+'??'
    else:
        #possible the labeler put more info, attempt to guess the first char
        print("Doing best guess " + a)
        temp = collaboration_converter(a[0])
        if('?' in temp):
            return '??' + a + '??'
        else:
            return temp

def student_converter(a):
    if len(a) > 0:
        if a[0].lower() == 't' or a.lower() == 'teacher':
            return 'teacher'
        elif "?" in a:
            return ""
        else:
            return re.sub("[^0-9]","",a)
    else:
        return ''
        

def turn_converter(a):
    if len(a.split(' ')) > 1:
        # print('Splitting ref - old ref:')
        # print(a)
        a = a.split(' ')[0]
        # print('new ref:')
        # print(a)
    if len(a.split(',')) > 1:
        # print('Splitting ref - old ref:')
        # print(a)
        a = a.split(',')[0]
        # print('new ref:')
        # print(a)
    return a.split('.')[-1].strip()

def talk_converter(a):
    if isinstance(a, str):
        return a.strip()
    else:
        return str(a).strip()


def convert_transcript(file):
    converter = {#'Claim': (lambda x: 'claim' if x else ''),
                # 'Evidence': (lambda x: 'evidence' if x else ''),
                # 'Warrant': (lambda x: 'explanation' if x else ''),
               #  'Low': (lambda x: 'low' if x else ''),
               #  'Med': (lambda x: 'med' if x else ''),
               #  'High': (lambda x: 'high' if x else ''),
                 #'Textual': (lambda x: 'textual' if x else ''),
                 #'Intertextual': (lambda x: 'textual' if x else ''),
                 #'Experiential': (lambda x: 'experiential' if x else ''),
                 'Collaboration Code': (lambda x: collaboration_converter(x) if x else ''),
    #             'Disc id':  (lambda x: turn_converter(x) if x else ''),
                 'Disc id': turn_converter,
                 'Turn of Reference': turn_converter,
                 'Argument Segmentation': talk_converter,
                 #'Sp id': student_converter,
                 #'Factual': (lambda x: 'f' if x else ''),
                 #'Interp': (lambda x: 'i' if x else '')
                 }
    

    d = pd.read_excel(file, header=0, usecols=range(23), keep_default_na=False, converters=converter)
    
    #print(json.dumps(d, indent = 2))

    #d['Argmove'] = d.Claim.str.cat(d.Evidence).str.cat(d.Warrant)
    #d['Specificity'] = d['Low'].str.cat(d['Med']).str.cat(d['High'])
    #d['Domain'] = d['Textual'].str.cat(d['Intertextual']).str.cat(d['Experiential'])
    #d['Question'] = "Question"#d['Factual/Literal'].str.cat(d['Interpretive'])

    cols = ['Disc id', 'Sp id', 'Argument Segmentation','Collaboration Code', 'Turn of Reference']


    data = pd.DataFrame.copy(d)
    data = data[cols]
    data = data.replace('', np.nan)
    data = data.dropna(axis=0, how='all').replace(np.nan, '')
    data.rename(index=str, columns={'Disc id': 'Turn', 'Sp id':'Student', 'Argument Segmentation':'Talk2',
                                    'Collaboration Coding':'Collaboration', 'Turn of Reference':'Reference'}, inplace=True)
    return data



