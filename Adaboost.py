#thsi program performs decision tree depth-1 stump to train the classifier. The decision stump is called T times (T iterations) and at the the 
#of T iterations, all the classifiers are summed up to give a final H classifer that's used on the testing data to predict labels.

#To run:
#python adaboost.py 10 train test

import sys,math
from collections import Counter

T=sys.argv[1]
trainfile=sys.argv[2]
testfile=sys.argv[3]

#reads data from the training file into a list of lists
def readdata():
	with open (trainfile,'r') as f:
		lines = f.read().splitlines()
		data=[]
		for line in lines:
			#print(line)
			features=line.split('\t')
			dataline=[]
			for f in features:
				dataline.append(f);
			data.append(dataline)
	#print(data)
	return data

#perfoems decision stump and updates weights on D vector for each iteration
def decisionstump(data,D):
	totalfeatures=len(data[0])-1
	sampleclass1=0
	sampleclass2=0
	for (i,j) in zip(data,D):
		if i[0]=='e':
			sampleclass1+=j
		else:
			sampleclass2+=j
	#count of number of records that belong to class 1, class 2 and the total number of training samples
	totalsamples=sampleclass1+sampleclass2
	#print("total samples and featires=",totalsamples,totalfeatures)
	entropyS=-(sampleclass1/totalsamples)*math.log(sampleclass1/totalsamples,2)-(sampleclass2/totalsamples)*math.log(sampleclass2/totalsamples,2)
	#print("entropyS=",entropyS) -- entropyS for the entire dataset
	gain=[0]
	i=1
	#holds the dictionary of each attribute, each value and class that it most likely predicts
	decisionmap={}
	#calculate the gain for each feature
	while(i<=totalfeatures):
		counts = {}
		featurecounts = {}
		linecount=1
		for (line,j) in zip(data,D):
			linecount+=1
			entry = counts.setdefault((line[0],line[i]),[0])
			entry[0] += j
			feature=featurecounts.setdefault(line[i],[0])
			feature[0]+=j
		#print("count=",counts[listkey])
		featureentropy=0

		#calculate entropy and gain for each value of a given attribute to predict which value tends to predict which class
		for f in featurecounts:
			#print(f,featurecounts[f])
			listkey=('e',f)
			if(listkey in counts):
				sampleclass1=counts[listkey][0]
			else:
				sampleclass1=0
			listkey=('p',f)
			if(listkey in counts):
				sampleclass2=counts[listkey][0]
			else:
				sampleclass2=0

			a=[]
			if(sampleclass1>=sampleclass2):
				a.append(f)
				a.append('e')
				if(i in decisionmap):
					decisionmap[i].append(a)
				else:
					decisionmap[i]=[a]
			else:
				a.append(f)
				a.append('p')
				if(i in decisionmap):
					decisionmap[i].append(a)
				else:
					decisionmap[i]=[a]
			#print(sampleclass1,sampleclass2,featurecounts[f][0])
			#calculate weighted entropy for each attribute
			if(sampleclass1!=0 and sampleclass2!=0):
				entropySV=-(sampleclass1/featurecounts[f][0])*math.log(sampleclass1/featurecounts[f][0],2)-(sampleclass2/featurecounts[f][0])*math.log(sampleclass2/featurecounts[f][0],2)
				entropySV=(featurecounts[f][0]/totalsamples)*entropySV
			else:
				entropySV=0
			featureentropy+=entropySV
		#print(featureentropy)
		#print(i,"=",decisionmap[i])
		#calculate inforation gain for each feature
		gain.append(entropyS-featureentropy)
		i+=1
	#print(gain)
	#for the current classifier, keep tab of the feature that gives max gain
	maxval=max(gain)
	maxindex=gain.index(maxval)
	#print(maxindex)
	#print(decisionmap[maxindex])
	D,alpha,d=makedecision(decisionmap[maxindex],maxindex,data,D)
	return D,alpha,d,maxindex

#this function performs the adaboost weighted avergae updates and calculates error rate and strngth of each decision stump classifier
#in this function, correct predictions weight goes down and wrong predicitons weights go up
def makedecision(decisionmap,maxindex,data,D):
	d=dict(decisionmap)
	y=[]
	for line in data:
		y.append(d[line[maxindex]])
	#print("prediction=",y)
	error=0
	i=0
	for line in data:
		if(line[0]!=y[i]):
			error+=D[i]
		i+=1
	if(error>0):
		alpha=0.5*math.log((1-error)/error)
	else:
		alpha=0
	#print("error rate=",error," alpha=",alpha)
	#print(alpha)
	i=0
	for line in data:
		if(line[0]=='e'):
			yi=1
		else:
			yi=-1
		if(y[i]=='e'):
			hxi=1
		else:
			hxi=-1
		D[i]=D[i]*math.exp(-alpha*yi*hxi)
		i+=1
	#print("unnormalized=",D)
	D = [float(i)/sum(D) for i in D]
	#print("New D=",D)
	return D,alpha,d

#call to adaboost train function, in return calls decision stump algorithm T times
def adaboosttrain(data):
	DAll=[]
	alphaall=[]
	decision=[]
	maxindices=[]
	D=[1.0/len(data)]*len(data)
	#print(D)
	for i in range(int(T)):
		#print("iteration ",i)
		D,a,d,maxindex=decisionstump(data,D)
		#print(a)
		DAll.append(D)
		alphaall.append(a)
		decision.append(d)
		maxindices.append(maxindex)
	#print(DAll)
	#print(alphaall)
	#print(decision)
	#print(maxindices)
	return alphaall,decision,maxindices

#after adaboost has been trained, test data is sent to this function
def adaboosttest(decision,alphaall,maxindices):
	with open (testfile,'r') as f:
		lines = f.read().splitlines()
		data=[]
		for line in lines:
			#print(line)
			features=line.split('\t')
			dataline=[]
			for f in features:
				dataline.append(f);
			data.append(dataline)
		#print(data)
	finalprediction=[]
	correctcount=0.0
	#calculate accuracy for the prediciotns made
	for line in data:
		y=finalHypothesis(line,decision,alphaall,maxindices)
		if(y==line[0]):
			correctcount+=1.0
	#print(correctcount)
	#print(len(data))
	print(correctcount/len(data)*100.0)
	for a in alphaall:
		print(a)

#run the final lassifier (sum of all classifiers using alpha and sign of the prediciton class) on test data to return results
def finalHypothesis(x,decision,alphaall,maxindices):
	hypotheses=[]
	for d,i in zip(decision,maxindices):
		if (x[i] in d):
			if(d[x[i]]=='e'): 
				hypotheses.append(1)
			else:
				hypotheses.append(-1)
		else:
			hypotheses.append(1)
	#print("prediction=",hypotheses)
	total=sum(a * h for (a, h) in zip(alphaall, hypotheses))
	#print(total)
	if total>0:
		return 'e';
	else:
		return 'p';


if __name__ == "__main__":
   data=readdata()
   alphaall,decision,maxindices=adaboosttrain(data)
   adaboosttest(decision,alphaall,maxindices)