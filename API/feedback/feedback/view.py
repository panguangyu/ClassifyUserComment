from django.http import HttpResponse
from django.shortcuts import render
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from snownlp import SnowNLP
from sklearn.externals import joblib
from snownlp import sentiment
import jieba
import pdb

def index(request):
	feedback = request.GET.get('feedback')
	context = {}
	if feedback == "" or (not isinstance(feedback, str)):
		context['tag'] = 4
	else:
		context['tag'] = getModelValue(feedback)
		context['feedback'] = feedback
		#context['proba'] = round(countProba(context['score'], context['tag'])*100, 4)
		#print(context)
		#pdb.set_trace()
	return render(request, 'index.html', context)

#def countProba(score, tag):
	#if (tag == 0 or tag == 3):
	#	proba = score/1;
	#elif (tag == 1):
	#	proba = score/0.5
	#elif (tag == 2):
	#	proba = score/0.75
	#return proba

def getModelValue(text):
	mnb = joblib.load('./model/10jqka_mnb')
	transformer = joblib.load('./model/10jqka_transformer')
	vectorizer = joblib.load('./model/10jqka_vectorizer')
	
	#print(transformer)
	#pdb.set_trace() 
	
	test_comment = text.encode("utf-8")

	test_comment = jieba.cut(test_comment)
	test_comment = " ".join(test_comment)
	test_comment = [test_comment]

	test_vectorizer_wordFrequency = vectorizer.transform(test_comment)

	test_tfidf = transformer.transform(test_vectorizer_wordFrequency).toarray()

	predict_classification = mnb.predict(test_tfidf)
	predict_classification_proba = mnb.predict_proba(test_tfidf)

	#print(predict_classification_proba)

	snownlp = SnowNLP(str(test_comment))
	snow_score = snownlp.sentiments

	#print(snow_score)

	# 如果snow判断大于0.8或朴素贝叶斯预测值大于0.8，则选择作为分类
	# snow_tag_stack = {0:0.25, 1:0.5, 2:0.75, 3:1}

	naive_score = predict_classification_proba[0][predict_classification[0]]

	if (snow_score >= 0.5):
		if (snow_score >= naive_score):      # 如果情感分析得分大于朴素贝叶斯得分，则选择该标签
			final_score = snow_score
			if (snow_score > 0.25):
				if (snow_score > 0.5):
					if (snow_score > 0.75):
						tag = 3
					else:
						tag = 2
				else:
					tag = 1
			else:
				tag = 0
		else:
			final_score = naive_score
			tag = predict_classification[0]
	else:
		snow_score = 1-snow_score
		if (snow_score >= naive_score):      # 如果情感分析得分大于朴素贝叶斯得分，则选择该标签
			final_score = snow_score
			if (snow_score > 0.5):
				if (snow_score > 0.75):
					tag = 0
				else:
					tag = 1
		else:
			final_score = naive_score
			tag = predict_classification[0]
		
	return tag