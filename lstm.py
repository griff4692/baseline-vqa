import json

finq = open("Questions_Train_mscoco/OpenEnded_mscoco_train2014_questions.json", 'r')
fina = open("mscoco_train2014_annotations.json", 'r')
fout = open("open_ques_ans.json", 'w')

questions = json.load(finq)
answers = json.load(fina)

for ques in questions['questions']:
	print ques
	break
	
for ans in answers['annotations']:
	print ans
	break
