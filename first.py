import survey
import nltk
import pandas as pd
import numpy as np
table = survey.Pregnancies()
table.ReadRecords()

fields = [fieldName for fieldName, *_ in table.GetFields()]
df = pd.DataFrame([[getattr(obj, name) for name in fields] for obj in table.records], columns=fields)

birthOrds = [set([pregnancy.birthord for pregnancy in table.records])]

# 求出几孩的怀孕周期的均值
birthOrds_mean = [(name, np.mean([pregnancy.birthord for pregnancy in table.records if pregnancy.birthord == name])) for name in birthOrds]

fDist = nltk.FreqDist([pregnancy.birthord for pregnancy in table.records])
totalOutcomeCount = sum([pregnancy.outcome for pregnancy in table.records])
print(fDist.most_common(5))
print(len(table.records))
print('total outcome is:', totalOutcomeCount)
