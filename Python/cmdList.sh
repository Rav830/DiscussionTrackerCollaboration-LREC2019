for i in "eager --use-cv ../Data/EAGER_10_fold_crossvalidation.json"
do
    #optional argument, use tf-idf
	for j in "" "--tf-idf"
	do  
	    #optional argument remove the non elements
	    for k in "" "--remove-non"
	    do
		    echo "Starting dummy models" &&
		    python3.7 model_Dummy.py $i $j $k --rebuild-data &&
		    python3.7 model_Dummy_regroup.py $i $j $k;


	    done
        python3.7 model_DetectThree_Dummy.py $i $j;
	    python3.7 model_DetectNon_Dummy.py $i $j;
	done
done
echo "Done with dummy models";

for i in "eager --use-cv ../Data/EAGER_10_fold_crossvalidation.json"
do
    #optional argument, use tf-idf
	for j in "" "--tf-idf"
	do  
	    #optional argument remove the non elements
	    for k in "" "--remove-non"
	    do
		    echo "Multinomial Naive Bayes Model" &&
            python3.7 model_Naive_Bayes.py $i $j $k &&
            python3.7 model_Naive_Bayes_regroup.py $i $j $k &&
            python3.7 model_Naive_Bayes_GroupAC.py $i $j $k;
	    done
	    python3.7 model_DetectNon_Naive_Bayes.py  $i $j;
	done
done
echo "Done with Naive Bayes models";

for i in "eager --use-cv ../Data/EAGER_10_fold_crossvalidation.json"
do
    #optional argument, use tf-idf
	for j in "" "--tf-idf"
	do  
	    #optional argument remove the non elements
	    for k in "" "--remove-non"
	    do
		    echo "Complement Naive Bayes Model" &&
		    python3.7 model_Complement_Naive_Bayes.py  $i $j $k &&
		    python3.7 model_Gaussian_Naive_Bayes.py  $i $j $k;
            python3.7 model_Complement_Naive_Bayes_GroupAC.py $i $j $k;
            python3.7 model_Gaussian_Naive_Bayes_GroupAC.py $i $j $k;
	    done
	    python3.7 model_DetectNon_Complement_Naive_Bayes.py $i $j &&
	    python3.7 model_DetectNon_Gaussian_Naive_Bayes.py  $i $j;
	    python3.7 model_DetectThree_Gaussian_Naive_Bayes.py  $i $j;
	done
done
echo "Done with alternate Naive Bayes models";

