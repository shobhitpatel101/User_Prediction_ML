from django.shortcuts import render,render_to_response
from django.http import HttpResponse,HttpResponseBadRequest
from django.core.files.base import ContentFile
from django.core.files.storage import FileSystemStorage
from django.core.files import File

import numpy as np
import pandas as pd
from sklearn import preprocessing 

import re


def home(request):
	return render(request,'musicapp/home.html')




def foo2(request,s):
	if request.method != 'POST':
		return HttpResponseBadRequest('<p align="center"><h1>Unkown File Error!</h1></p>')


	f_txt=open('/home/sumit/locallib/files/%s.txt' %s[:-4],'r')

	f_csv=open('/home/sumit/locallib/files/csv/%s.tsv' %s[:-4],'w')
	f_trash=open('/home/sumit/locallib/files/trash/%s.txt' %s[:-4],'w')

	for line in f_txt:
		
		line=line.replace(' - ','\t',1)

		pos=line.find(':')
		new_line=line[:pos+1]
		new_line=new_line.replace(', ','\t',1)
		line=line[pos+1:]
		n_pos=line.find(':')

		tab_line=line[n_pos:]
		tab_line2=line[:n_pos]
		tab_line=tab_line.replace('\t',' ')

		tab_line=tab_line.replace(':','\t',1)
		tab_line=re.sub('[^a-zA-Z\t\n0-9]',' ',tab_line)

		tab_line=" ".join(tab_line.split())
		tab_line='\t'+tab_line+'\n'

		line=tab_line2+tab_line
		line=new_line+line
		
		lenght=len(tab_line)

		if (line.find('Media omitted')!=-1):
			line=""

		if lenght<100 and lenght>2 and line.count('\t')==3 and line.count(' am\t')+line.count(' pm\t')==1:
			if line.find("media")==-1:
				low=line.lower()
				
				f_csv.write(low)

			else:
				print(line)

		else:
			f_trash.write(line)

	f_txt.close()
	f_csv.close()
	f_trash.close()

	return HttpResponse('File Uploaded <img src="http://chittagongit.com//images/green-check-icon-transparent/green-check-icon-transparent-0.jpg" width="100px" height="60px"> <br><br> <a href="/app">Go back</a>')
	

#-----------------


def foo(request):
	if request.method != 'POST':
		return HttpResponseBadRequest('<p align="center"><h1>Unkown File Error!</h1></p>')


	file=request.FILES['file']
	
	with open('/home/sumit/locallib/files/%s.txt' %file.name[:-4],'wb') as dest:
		file_write=File(dest)
		for chunk in file:
			abcd=chunk

			file_write.write(abcd)

	dest.close()
	foo2(request,file.name)

	#return HttpResponse('File Uploaded <img src="http://chittagongit.com//images/green-check-icon-transparent/green-check-icon-transparent-0.jpg" width="100px" height="60px"> <br><br> <a href="/app">Go back</a>')
	return render(request,'musicapp/call_nn.html', {'file': file.name})




#---------------------------------------------------



def nn(request):

	file=request.POST.get('file')
	

	dataset =pd.read_csv('/home/sumit/locallib/files/csv/%s.tsv' %file[:-4],delimiter='\t',header=None)
	y=dataset.iloc[:,2].values
	x=dataset.iloc[:,3].values


   
	le=preprocessing.LabelEncoder()
	y=le.fit_transform(y)

	from sklearn.feature_extraction.text import CountVectorizer
	cv=CountVectorizer()
	cv.fit(x)
	X=cv.transform(x).toarray()

	#naive-baise
	from sklearn.cross_validation import train_test_split
	x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

	from sklearn.naive_bayes import GaussianNB
	clas=GaussianNB()
	clas.fit(x_train,y_train)

	y_pred=clas.predict(x_test)

	from sklearn.metrics import confusion_matrix
	cm=confusion_matrix(y_test,y_pred)

	ans=((cm[0][0]+cm[1][1])/y_pred.size)

	return render(request,"musicapp/ans_nn.html",{'ans':ans, 'q':cm[0][0] ,'w':cm[0][1] ,
		'a':cm[1][0] , 's':cm[1][1]})


#-------------------------ann--------------------------

def ann(request):

	file=request.POST.get('file')
	

	dataset =pd.read_csv('/home/sumit/locallib/files/csv/%s.tsv' %file[:-4],delimiter='\t',header=None)
	y=dataset.iloc[:,2].values
	x=dataset.iloc[:,3].values


	   
	le=preprocessing.LabelEncoder()
	y=le.fit_transform(y)

	from sklearn.feature_extraction.text import CountVectorizer
	cv=CountVectorizer()
	X=cv.fit_transform(x).toarray()

	#naive-baise
	from sklearn.cross_validation import train_test_split
	x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

	#nural net
	import keras
	from keras.models import Sequential
	from keras.layers import Dense
	 
	clas=Sequential()
	clas.add(Dense(output_dim=983,init='uniform' ,activation='relu',input_dim=int(x_test.size/y_test.size)))
	clas.add(Dense(output_dim=283,init='uniform',activation='relu'))
	clas.add(Dense(output_dim=483,init='uniform',activation='relu'))
	clas.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))

	clas.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

	clas.fit(x_train,y_train,batch_size=10,nb_epoch=2)
	y_pred=clas.predict(x_test)

	#probality to 0-1
	y_pred=(y_pred>0.5)

	from sklearn.metrics import confusion_matrix
	cm=confusion_matrix(y_test,y_pred)

	ans=((cm[0][0]+cm[1][1])/y_pred.size)

	return render(request,"musicapp/ans_nn.html",{'ans':ans, 'q':cm[0][0] ,'w':cm[0][1] ,
		'a':cm[1][0] , 's':cm[1][1]})
	

	