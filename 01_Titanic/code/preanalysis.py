#!/usr/bin/env python
# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import pandas as pd
import Tit_lib as tl


# Bypass Auto-newline
pd.set_option('display.expand_frame_repr', False)

# Load File
data_train = tl.load_file("train.csv")

# preview the data

# Figure 1: General Distribution

fig = plt.figure(1, figsize=(16, 9), dpi = 110)
fig.set(alpha=0.3)

plt.subplot2grid((2,3),(0,0)) # subplots
data_train.Survived.value_counts().plot(kind='bar')
plt.title("Survival")
plt.ylabel("Number")

plt.subplot2grid((2,3),(0,1))
data_train.Pclass.value_counts().plot(kind="bar")
plt.ylabel("Number")
plt.title("Pclass")

plt.subplot2grid((2,3),(0,2))
data_train.Age[data_train.Survived == 1].plot(kind='kde')
data_train.Age[data_train.Survived == 0].plot(kind='kde')
plt.ylabel("Age")
plt.grid(b=True, which='major', axis='y')
plt.title("Survival")

plt.subplot2grid((2,3),(1,0), colspan=2)
data_train.Age[data_train.Pclass == 1].plot(kind='kde')
data_train.Age[data_train.Pclass == 2].plot(kind='kde')
data_train.Age[data_train.Pclass == 3].plot(kind='kde')
plt.xlabel("Age")
plt.ylabel("Density")
plt.title("Age distribution")
plt.legend(('3', '2','1'), loc='best')

plt.subplot2grid((2,3),(1,2))
data_train.Embarked.value_counts().plot(kind='bar')
plt.title("From different port")
plt.ylabel("Number")# fig = plt.figure(1, figsize=(16, 9), dpi = 100)
fig.set(alpha=0.3)

# Figure 2: Survival Analysis

# fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 9), dpi = 60)
# Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()
# Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()
# df1 = pd.DataFrame({'Survived':Survived_1, 'Dead':Survived_0})
# df1.plot(kind='bar', stacked=True, ax=axes[0])
# plt.title("By Pclass")
# plt.xlabel("Pclass")
# plt.ylabel("Number")
#
# Survived_m = data_train.Survived[data_train.Sex == 'male'].value_counts()
# Survived_f = data_train.Survived[data_train.Sex == 'female'].value_counts()
# df2 = pd.DataFrame({'Male':Survived_m, 'Female':Survived_f})
# df2.plot(kind='bar', stacked=True, ax=axes[1])
# plt.title("By sex")
# plt.xlabel("Sex")
# plt.ylabel("Number")

# Figure 3: More detailed

# fig = plt.figure(3, figsize=(16, 9), dpi = 80)
# fig.set(alpha=0.65)
# plt.title("By Pclass and sex")
#
# ax1 = fig.add_subplot(141)
# data_train.Survived[data_train.Sex == 'female'][data_train.Pclass != 3].value_counts().plot(kind='bar', label="female highclass", color='#FA2479')
# ax1.set_xticklabels(["Dead", "Survived"], rotation = 0)
# ax1.legend(["Female/1&2"], loc = 'best')
#
# ax2 = fig.add_subplot(142, sharey = ax1)
# data_train.Survived[data_train.Sex == 'female'][data_train.Pclass == 3].value_counts().plot(kind='bar', label='female, low class', color='pink')
# ax2.set_xticklabels(["Dead", "Survived"], rotation=0)
# plt.legend(["Female/3"], loc = 'best')
#
# ax3 = fig.add_subplot(143, sharey = ax1)
# data_train.Survived[data_train.Sex == 'male'][data_train.Pclass != 3].value_counts().plot(kind='bar', label='male, high class',color='lightblue')
# ax3.set_xticklabels(["Dead", "Survived"], rotation=0)
# plt.legend(["Male/1&2"], loc = 'best')
#
# ax4 = fig.add_subplot(144, sharey = ax1)
# data_train.Survived[data_train.Sex == 'male'][data_train.Pclass == 3].value_counts().plot(kind='bar', label='male low class', color='steelblue')
# ax4.set_xticklabels(["Dead", "Survived"], rotation=0)
# plt.legend(["Male/3"], loc = 'best')

# Figure 4: by port
# Survived_0 = data_train.Embarked[data_train.Survived == 0].value_counts()
# Survived_1 = data_train.Embarked[data_train.Survived == 1].value_counts()
# df=pd.DataFrame({'Survived':Survived_1, 'Dead':Survived_0})
# df.plot(kind='bar', stacked=True)
# plt.title("By Port")
# plt.xlabel("Port")
# plt.ylabel("Number")
#
plt.show()

# g = data_train.groupby(['SibSp','Survived'])
# df = pd.DataFrame(g.count()['PassengerId'])
# print df
#
# g = data_train.groupby(['Parch','Survived'])`
# df = pd.DataFrame(g.count()['PassengerId'])
# print df
