# Automatic Summerization of CNN News Using TextRank Algorithm
			

**Technology: PySpark, Databricks.<br>
• Implemented the TextRank Algorithm on the CNN news Summary dataset from DeepMind using PySpark.<br> 
• Evaluated the results of our summarization with the given data using F1 measure, precision and recall metrics and also wrote a research paper on the same.<br>**

For our project Text Summarization Using TextRank algorithm we have used the dataset: CNN-News Summary from the Kaggle website: https://cs.nyu.edu/~kcho/DMQA/

The input dataset has been loaded on AWS S3 at 

Stories 	-> s3://cs6350proj20/cnn100/stories


Please save it on your own bucket in S3 if you need to execute the code and copy the S3 key credentials in the appropriate code section.  

Steps to execute code:
1. Install required libraries:
	1. sklearn
	2. networkx
	3. pandas
	4. py-rouge
	5. IPython
	6. nltk punkt

2. run in AWS notebook with spark and hadoop installed cluster


3. change to PySpark kernel


4. set input data path and number of summary sentence to be extracted


5. run all cells 


	




