# coding=utf-8
import json
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("dataset", help ="The dataset to run the models on", choices=("atlas", "eager", "combined")) 
parser.add_argument("--rebuild-data", help="rebuild the dataframe we use", action = "store_true")
parser.add_argument("--num-files", help="the number of files to process", type=int, default=-1)
parser.add_argument("--tf-idf", help="if set, it will build the dataset using tf-idf", action='store_true')
parser.add_argument("--remove-non", help="removes the non category from the data", action = 'store_true')
parser.add_argument("--use-cv", help="path to a file that contains the cross validation splits", default=None) 
args = parser.parse_args()

if(args.dataset == "combined"):
	print("message from config, this is not implemented yet!!")
	raise SystemExit
	
args.dataset = args.dataset.upper()
args.tf_idf = "_TF-IDF" if args.tf_idf else ""
args.remove_non = "_filtered" if args.remove_non else ""
args.use_cv = json.load(open(args.use_cv)) if args.use_cv else None

import sys
if(sys.stdout.encoding != 'utf-8'):
   sys.stdout = open(1, 'w', encoding = 'utf-8', closefd = False)


#gotten from http://web.archive.org/web/20141016185743/https://mail.python.org/pipermail/python-list/2007-May/460639.html
class Tee(object):
     def __init__(self, name, mode):
         self.file = open(name, mode)
         self.stdout = sys.stdout
         sys.stdout = self
     def close(self):
         if self.stdout is not None:
             sys.stdout = self.stdout
             self.stdout = None
         if self.file is not None:
             self.file.close()
             self.file = None
     def write(self, data):
         self.file.write(data)
         self.stdout.write(data)
         self.flush()
     def flush(self):
         self.file.flush()
         self.stdout.flush()
     def __del__(self):
         self.close()
