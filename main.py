import word
import words_model
import random
import lwa
import numpy as np


model = words_model.words_6

grades = [list(model['words'].keys())[random.randrange(0, 1)] for _ in range(25)] + \
         [list(model['words'].keys())[random.randrange(5, 6)] for _ in range(25)]
W = []
for item in model['words']:
    W.append(grades.count(item))

h = min(item['lmf'][-1] for item in model['words'].values())
m = 50
intervals_umf = lwa.alpha_cuts_intervals(m)
intervals_lmf = lwa.alpha_cuts_intervals(m, h)


res_lmf = lwa.y_lmf(intervals_lmf, model, W)
res_umf = lwa.y_umf(intervals_umf, model, W)
res = lwa.construct_dit2fs(np.arange(*model['x']), intervals_lmf, res_lmf, intervals_umf, res_umf)
res.plot()

sm = []
model = words_model.words_32
for title, fou in model['words'].items():
    sm.append((title, res.similarity_measure(word.Word(None, model['x'], fou['lmf'], fou['umf']))))
res_word = max(sm, key=lambda item: item[1])
print(res_word)



