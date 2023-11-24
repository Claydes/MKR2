import word
import words_model
import random
import lwa
import numpy as np


def process(grades, grades_model, res_model):
    W = []
    for item in grades_model['words']:
        W.append(grades.count(item))

    h = min(item['lmf'][-1] for item in grades_model['words'].values())
    m = 50
    intervals_umf = lwa.alpha_cuts_intervals(m)
    intervals_lmf = lwa.alpha_cuts_intervals(m, h)

    res_lmf = lwa.y_lmf(intervals_lmf, grades_model, W)
    res_umf = lwa.y_umf(intervals_umf, grades_model, W)
    res = lwa.construct_dit2fs(np.arange(*grades_model['x']), intervals_lmf, res_lmf, intervals_umf, res_umf)

    # res.plot()

    sm = []
    for title, fou in res_model['words'].items():
        sm.append((title, res.similarity_measure(word.Word(None, res_model['x'], fou['lmf'], fou['umf']))))
    res_word = max(sm, key=lambda item: item[1])
    return res_word


def generate_results(grades_count, tests_num):
    grades_model = words_model.words_7
    res_model = words_model.words_11
    for i in range(tests_num):
        grades = [list(grades_model['words'].keys())[random.randrange(
            0, len(grades_model['words']))] for _ in range(grades_count)]
        print(f"Grades: {grades}")
        print(f"Result: {process(grades, grades_model, res_model)}\n")


if __name__ == '__main__':
    generate_results(10, 5)



