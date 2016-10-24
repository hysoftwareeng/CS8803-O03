from scipy.interpolate import spline
import matplotlib.pyplot as plt
import numpy as np

# f = open('plot-results-5-0.995-0.99-0.0-1.csv', 'r')
# lines = f.readlines()
# f.close()
# data = np.array([l.split() for l in lines]).T
#
# episodes = data[0].astype(np.int)
# steps = data[1].astype(np.int)
# total_rewards = data[2].astype(np.float)
# last_rewards = data[3].astype(np.int)
# epsilon = data[4].astype(np.float)
# average_rewards = data[5].astype(np.float)
#
# fig, ax = plt.subplots()
# plt.xlabel('episode')
# plt.ylabel('total reward')
# # ax.plot(episodes, total_rewards)
# xnew = np.linspace(episodes.min(), episodes.max(), 50)
# total_rewards_smooth = spline(episodes, total_rewards, xnew)
# ax.plot(xnew, total_rewards_smooth)


# fig, ax = plt.subplots()
# plt.xlabel(r'$\lambda$')
# plt.ylabel('ERROR')
# ax.plot(lambdas, rmse_list, '-o', label='Widrow-Hoff')
# ax.text(lambdas[-1] - 0.075, rmse_list[-1] + 0.0025, 'Widrow-Hoff', fontsize=10)
# ax.margins(0.1)
# plt.legend(loc='lower right')
# fig.savefig('fig3.png')


f = open('output-exp1-0.995-0.99.csv', 'r')
lines = f.readlines()
f.close()
data = np.array([l.split(',') for l in lines[1:]]).T

episodes = data[0].astype(np.int)
total_rewards = data[1].astype(np.float)
average_rewards = data[2].astype(np.float)

fig, ax = plt.subplots()
plt.xlabel('episode')
plt.ylabel('total reward')
ax.plot(episodes, total_rewards)
ax.text(episodes[-1] + 10, total_rewards[-1] + 0.5, r'$\gamma$' + '=0.995,\ndecay=0.99', fontsize=10)
fig.savefig('fig1a.png')

fig, ax = plt.subplots()
plt.xlabel('episode')
plt.ylabel('average reward')
ax.plot(episodes, average_rewards)
ax.text(episodes[-1] + 10, average_rewards[-1] + 0.5, r'$\gamma$' + '=0.995,\ndecay=0.99', fontsize=10)
fig.savefig('fig1b.png')


f = open('plot-results-5-0.995-0.99-0.0-1.csv', 'r')
lines = f.readlines()
f.close()
data = np.array([l.split() for l in lines]).T

episodes = data[0].astype(np.int)
steps = data[1].astype(np.int)
total_rewards = data[2].astype(np.float)
last_rewards = data[3].astype(np.int)
epsilon = data[4].astype(np.float)
average_rewards = data[5].astype(np.float)

fig, ax = plt.subplots()
plt.xlabel('episode')
plt.ylabel('total reward')
ax.plot(np.arange(len(total_rewards[-500:-400])), total_rewards[-500:-400])
fig.savefig('fig2a.png')

fig, ax = plt.subplots()
plt.xlabel('episode')
plt.ylabel('average reward')
ax.plot(np.arange(len(average_rewards[-500:-400])), average_rewards[-500:-400])
fig.savefig('fig2b.png')

# fig, ax = plt.subplots()
# plt.xlabel(r'$\lambda$')
# plt.ylabel('ERROR')
# ax.plot(lambdas, rmse_list, '-o', label='Widrow-Hoff')
# ax.text(lambdas[-1] - 0.075, rmse_list[-1] + 0.0025, 'Widrow-Hoff', fontsize=10)
# ax.margins(0.1)
# plt.legend(loc='lower right')
# fig.savefig('fig3.png')


f1 = open('output-exp1-0.995-0.99.csv', 'r')
lines1 = f1.readlines()
f1.close()
data1 = np.array([l.split(',') for l in lines1[1:]]).T
episodes1 = data1[0].astype(np.int)
total_rewards1 = data1[1].astype(np.float)
average_rewards1 = data1[2].astype(np.float)

f2 = open('output-exp1-0.995-0.8.csv', 'r')
lines2 = f2.readlines()
f2.close()
data2 = np.array([l.split(',') for l in lines2[1:]]).T
episodes2 = data2[0].astype(np.int)
total_rewards2 = data2[1].astype(np.float)
average_rewards2 = data2[2].astype(np.float)

f3 = open('output-exp1-0.995-0.7.csv', 'r')
lines3 = f3.readlines()
f3.close()
data3 = np.array([l.split(',') for l in lines3[1:]]).T
episodes3 = data3[0].astype(np.int)
total_rewards3 = data3[1].astype(np.float)
average_rewards3 = data3[2].astype(np.float)

f4 = open('output-exp1-0.8-0.99.csv', 'r')
lines4 = f4.readlines()
f4.close()
data4 = np.array([l.split(',') for l in lines4[1:]]).T
episodes4 = data4[0].astype(np.int)
total_rewards4 = data4[1].astype(np.float)
average_rewards4 = data4[2].astype(np.float)

f5 = open('output-exp1-0.7-0.99.csv', 'r')
lines5 = f5.readlines()
f5.close()
data5 = np.array([l.split(',') for l in lines5[1:]]).T
episodes5 = data5[0].astype(np.int)
total_rewards5 = data5[1].astype(np.float)
average_rewards5 = data5[2].astype(np.float)

fig, ax = plt.subplots()
plt.xlabel('episode')
plt.ylabel('total reward')
ax.plot(episodes1, total_rewards1, label=r'$\gamma$' + '=0.995, decay=0.99')
ax.text(episodes1[-1] + 10, total_rewards1[-1] + 50, r'$\gamma$' + '=0.995,\ndecay=0.99', fontsize=10)
ax.plot(episodes2, total_rewards2, label=r'$\gamma$' + '=0.995, decay=0.8')
ax.text(episodes2[-1] - 100, total_rewards2[-1] + 50, r'$\gamma$' + '=0.995,\ndecay=0.8', fontsize=10)
ax.plot(episodes3, total_rewards3, label=r'$\gamma$' + '=0.995, decay=0.7')
ax.text(episodes3[-1] + 50, total_rewards3[-1] + 0.5, r'$\gamma$' + '=0.995,\ndecay=0.7', fontsize=10)
ax.plot(episodes4, total_rewards4, label=r'$\gamma$' + '=0.8, decay=0.99')
ax.text(episodes4[-1] + 50, total_rewards4[-1] + 0.5, r'$\gamma$' + '=0.8,\ndecay=0.99', fontsize=10)
ax.plot(episodes5, total_rewards5, label=r'$\gamma$' + '=0.7, decay=0.99')
ax.text(episodes5[-1] + 50, total_rewards5[-1] + 0.5, r'$\gamma$' + '=0.7,\ndecay=0.99', fontsize=10)
plt.legend(loc='lower right')
fig.savefig('fig3a.png')

fig, ax = plt.subplots()
plt.xlabel('episode')
plt.ylabel('average reward')
ax.plot(episodes1, average_rewards1, label=r'$\gamma$' + '=0.995, decay=0.99')
ax.text(episodes1[-1] + 10, average_rewards1[-1] + 25, r'$\gamma$' + '=0.995,\ndecay=0.99', fontsize=10)
ax.plot(episodes2, average_rewards2, label=r'$\gamma$' + '=0.995, decay=0.8')
ax.text(episodes2[-1] - 100, average_rewards2[-1] + 25, r'$\gamma$' + '=0.995,\ndecay=0.8', fontsize=10)
ax.plot(episodes3, average_rewards3, label=r'$\gamma$' + '=0.995, decay=0.7')
ax.text(episodes3[-1] + 50, average_rewards3[-1] + 0.5, r'$\gamma$' + '=0.995,\ndecay=0.7', fontsize=10)
ax.plot(episodes4, average_rewards4, label=r'$\gamma$' + '=0.8, decay=0.99')
ax.text(episodes4[-1] + 50, average_rewards4[-1] + 0.5, r'$\gamma$' + '=0.8,\ndecay=0.99', fontsize=10)
ax.plot(episodes5, average_rewards5, label=r'$\gamma$' + '=0.7, decay=0.99')
ax.text(episodes5[-1] + 50, average_rewards5[-1] + 0.5, r'$\gamma$' + '=0.7,\ndecay=0.99', fontsize=10)
plt.legend(loc='lower right')
fig.savefig('fig3b.png')


f6 = open('cartpole-0.995-0.99.csv', 'r')
lines6 = f6.readlines()
f6.close()
data6 = np.array([l.split(',') for l in lines6[1:]]).T
episodes6 = data6[0].astype(np.int)
total_rewards6 = data6[1].astype(np.float)
average_rewards6 = data6[2].astype(np.float)

fig, ax = plt.subplots()
plt.xlabel('episode')
plt.ylabel('total reward')
ax.plot(episodes6, total_rewards6)
ax.text(episodes6[-1] + 10, total_rewards6[-1] + 0.5, r'$\gamma$' + '=0.995,\ndecay=0.99', fontsize=10)
ax.margins(0.1)
fig.savefig('fig4a.png')

fig, ax = plt.subplots()
plt.xlabel('episode')
plt.ylabel('average reward')
ax.plot(episodes6, average_rewards6)
ax.text(episodes6[-1] + 10, average_rewards6[-1] + 0.5, r'$\gamma$' + '=0.995,\ndecay=0.99', fontsize=10)
ax.margins(0.1)
fig.savefig('fig4b.png')
