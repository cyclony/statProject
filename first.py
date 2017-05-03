# 第一胎婴儿的数据统计规律分析
import survey
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

table = survey.Pregnancies()
table.ReadRecords()

fields = [fieldName for fieldName, *_ in table.GetFields()]
df = pd.DataFrame([[getattr(obj, name) for name in fields] for obj in table.records], columns=fields)

birthOrds = [set([pregnancy.birthord for pregnancy in table.records])]


# 求出几孩的怀孕周期的均值
birthOrds_mean = [(name, np.mean([pregnancy.birthord for pregnancy in table.records if pregnancy.birthord == name])) for name in birthOrds]

b1stPregLength_fd = nltk.FreqDist([pregnancy.prglength for pregnancy in table.records if pregnancy.birthord == 1])
n1stPregLength_fd = nltk.FreqDist([pregnancy.prglength for pregnancy in table.records if pregnancy.birthord != 1])

X = np.array([x for x in range(21, 44)])
Y1 = [b1stPregLength_fd.get(y, 0) for y in X]
Y2 = [n1stPregLength_fd.get(y, 0) for y in X]


# 生成一个2 X 2的图像区域，4个图表
fig = plt.figure(figsize=(20, 10))

ax1 = fig.add_subplot(2, 2, 1)


ax1.bar(X, Y1, width=0.5, facecolor='lightskyblue', label=u'first child ')
ax1.bar(X + 0.5, Y2, width=0.5, facecolor='yellowgreen', label=u'not first child')
legend = ax1.legend(loc='upper left', shadow=True)

for x, y in zip(X, Y1):
    if y != 0:
        ax1.text(x + 0.3, y + 0.5, '%i' % y, fontsize='8', ha='center', va='bottom')
for x, y in zip(X, Y2):
    if y != 0:
        ax1.text(x + 0.3, y + 0.5, '%i' % y, fontsize='8', ha='center', va='bottom')

# plot with frequency
X = np.array([x for x in range(21, 44)])
Y1 = [b1stPregLength_fd.freq(y) for y in X]
Y2 = [n1stPregLength_fd.freq(y) for y in X]

ax2 = fig.add_subplot(2, 2, 2)


ax2.bar(X, Y1, width=0.5, facecolor='lightskyblue', label=u'first child ')
ax2.bar(X + 0.5, Y2, width=0.5, facecolor='yellowgreen', label=u'not first child')
legend = ax2.legend(loc='upper left', shadow=True)

for x, y in zip(X, Y1):
    if y != 0:
        ax2.text(x + 0.3, y, '%.3f' % y, fontsize='8', ha='center', va='bottom')
for x, y in zip(X, Y2):
    if y!= 0:
        ax2.text(x + 0.3, y, '%.3f' % y, fontsize='8', ha='center', va='bottom')

ax3 = fig.add_subplot(2, 2, 3)
ax3.plot(np.random.randn(50).cumsum(), 'k--')

ax4 = fig.add_subplot(2, 2, 4)
ax4.scatter(np.arange(30), np.arange(30) + 3 * np.random.randn(30))

fig.subplots_adjust(wspace=0.1, hspace=0.1)
plt.show()

