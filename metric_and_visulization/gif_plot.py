import random
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# fig：即为我们的绘图对象.

# interval是指时间间隔，每一帧与每一帧之间的时间间隔为20毫秒，间隔越大，动画越慢
# blit:只更新当前点，不是全部，True则是更新全部画面



def read_file(path_root):                                        #读取插值文件目录
    insert_root = []
    for files, names, dirs in os.walk(path_root):
        file_dir = []
        # print(dirs)
        for dir in dirs:
            file_dir.append(dir)
        # print()
        for i in range(len(file_dir)):
            file_root1 = os.path.join(files, file_dir[i])
            insert_root.append(file_root1)
        # print(insert_root)
    return insert_root


def file_information(read_path):
    path = read_file(read_path)
    print(path)
    print(1)

    for one_path in path:
        # print(one_path.split('\\')[-1][:-4])
        xx = np.genfromtxt(one_path,
                           names=['number', 'time', 'mmsi1', 'lng1', 'lat1', 'sog1', 'cog1', 'mmsi2', 'lng2',
                                  'lat2', 'sog2', 'cog2'],
                           dtype='int,S20,int,f8,f8,f8,f8,int,f8,f8,f8,f8', delimiter=',')

        time_min = int(xx['number'][0])
        time_max = int(xx['number'][-1]+1)
        # print(time_min,time_max,len(xx))

        fig = plt.figure()
        ims = []

        for c in range(time_min, time_max, 10):
            # imm = []    #第c帧
            # im1 = plt.plot(xx['lng1'][:c],xx['lat1'][:c],c='steelblue')
            # imm +=im1
            # im2 = plt.plot(xx['lng2'][:c], xx['lat2'][:c],c='palevioletred')
            # imm +=im2
            # ims.append(imm)   #轨迹数据每隔30s加入一帧
            im1, = plt.plot(xx['lng1'][:c], xx['lat1'][:c], c='steelblue')
            im2, = plt.plot(xx['lng2'][:c], xx['lat2'][:c], c='palevioletred')
            ims.append([im1,im2])   #轨迹数据每隔30s加入一帧
        mmsi = str(one_path.split('\\')[-1][:-4])
        # save_path = 'E:\whut\决策实验\changjiang/all\新建文件夹\追越动态图/%s.gif'%(one_path.split('\\')[-1][:-4])
        # save_path = 'E:\whut\决策实验\changjiang/all\新建文件夹\追越动态图/'+mmsi+'.gif'
        ani = animation.ArtistAnimation(fig, artists=ims, interval=100, repeat_delay=1000,blit=True)
        plt.show()
        # ani.save(save_path)
        writer = animation.FFMpegWriter()
        # ani.save(save_path,writer = writer)
        print("保存了" + one_path.split('\\')[-1][:-4])


# file_information(r'E:\whut\决策实验\changjiang\all\新建文件夹\追越信息表')

import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig = plt.figure()

ims = []
for _ in range(10):
    im1, = plt.plot([random.randrange(10), random.randrange(10)], [random.randrange(10), random.randrange(10)])
    im2, = plt.plot([random.randrange(10), random.randrange(10)], [random.randrange(10), random.randrange(10)])
    ims.append([im1, im2])
ani = animation.ArtistAnimation(fig, ims)
plt.show()

ani.save('im.mp4')
