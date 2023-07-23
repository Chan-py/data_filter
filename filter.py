import numpy as np
import os
dir_list = os.listdir('episodes4')
numbers = [file.split('_')[0] for file in dir_list]
dir_list2 = os.listdir('filtered_episodes4')
numbers2 = [file.split('_')[0] for file in dir_list2]

# 등장 횟수가 5인 숫자만 필터링하고, 중복을 제거한 뒤 name 리스트에 저장
name = list(set([number for number in numbers if number not in numbers2]))

from tqdm import tqdm

def read_dataset(path):
    data = np.load(path)
    if path.split('.')[-1] == 'npz':
        data = data['arr_0']
    return data

idx, idx1 = 0, 0

num = len(name)
num_lst = list(range(0, num+1, num//5))

for i in tqdm(range(num_lst[0], num_lst[1])):
    obs = [read_dataset(f"episodes4/{name[i]}_obs0.npz"), read_dataset(f"episodes4/{name[i]}_obs1.npz")]
    act = [read_dataset(f"episodes4/{name[i]}_action0.npz"), read_dataset(f"episodes4/{name[i]}_action1.npz")]

    for j in range(len(obs)):
        for k in range(len(obs[0])-1):
            img = obs[j][k]
            white_pixels = np.sum((img[:, :, 0] > 145) & (img[:, :, 1] > 145) & (img[:, :, 2] > 145))  # 흰색 픽셀 수 계산
            black_pixels = np.sum((img[:, :, 0] < 20) & (img[:, :, 1] < 20) & (img[:, :, 2] < 20)) # 검정색 픽셀 수 계산
            # whole_pixels = np.sum(img[:, :, 0] >= 0)
            # whole_pixels are 65536
            if white_pixels > 50 and black_pixels > 20:
                # np.save('filtered_episodes/good_{}_img.npy'.format(idx), img)
                # print("White pixels:", white_pixels)
                # print("Black_pixels: ", black_pixels)
                # plt.imshow(img)
                # plt.show()
                idx += 1
            else:
                # print("White pixels:", white_pixels)
                # print("Black_pixels: ", black_pixels)
                # plt.imshow(img)
                # plt.show()
                act[1-j][k] = 7
                idx1 += 1

    # print(*act[0])
    # print(*act[1])
    np.savez_compressed(f"filtered_episodes4/{name[i]}_action0.npz", np.array(act[0]))
    np.savez_compressed(f"filtered_episodes4/{name[i]}_action1.npz", np.array(act[1]))
    # print(idx, idx1)
    idx, idx1 = 0, 0
