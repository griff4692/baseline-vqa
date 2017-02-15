from word_table import WordTable
import numpy as np

WT = WordTable()

question = WT.getBOW('How many times do I have to tell you, Mary?')

print WT.getIdx('mary')

print np.sum(question)

