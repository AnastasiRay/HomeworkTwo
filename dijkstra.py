import random
import numpy as np
import matplotlib.pyplot as plt
from heapq import heappop, heappush
from PIL import Image

width = 15
height = 15
matrix = np.zeros((width, height))
obs_width = 3
obs_num = 10
obs_list = [
    (
        random.randint(0, width - obs_width - 1),
        random.randint(0, height - obs_width - 1)
    )
    for _ in range(obs_num)
]

def test_obs(x, y):
    for x_step in range(-1, 2):
        for y_step in range(-1, 2):
            if matrix[x + x_step][y + y_step] == 1:
                return False
    return True

for x_1, y_1 in obs_list:
    x_2 = x_1 + random.randint(1, obs_width)
    y_2 = y_1 + random.randint(1, obs_width)
    temp_obs = []
    for x in range(x_1, x_2):
        for y in range(y_1, y_2):
            if test_obs(x, y):
                temp_obs.append((x, y))
    for x, y in temp_obs:            
        matrix[x][y] = 1

temp_matrix = np.transpose((matrix == 0).nonzero())
start_x, start_y = temp_matrix[random.randint(0, temp_matrix.shape[0] - 1)]
matrix[start_x][start_y] = 2

temp_matrix = np.transpose((matrix == 0).nonzero())
end_x, end_y  = temp_matrix[random.randint(0, temp_matrix.shape[0] - 1)]
matrix[end_x][end_y] = 3

heap = []
distances = {}
heappush(heap, (0, start_x, start_y))
distances[(start_x, start_y)] = 0

path = []
visited = []
visited.append((start_x, start_y))

while heap:
    (distance, curr_x, curr_y) = heappop(heap)
    if curr_x == end_x and curr_y == end_y:
        while curr_x != start_x or curr_y != start_y:
            if curr_x == end_x and curr_y == end_y:
                (curr_x, curr_y) = distances[(curr_x, curr_y)][1]
                continue
            path.append((curr_x, curr_y))
            (curr_x, curr_y) = distances[(curr_x, curr_y)][1]
        break
    for x_step in range(-1, 2):
        for y_step in range(-1, 2):
            if x_step == 0 and y_step == 0:
                continue
            new_x, new_y = curr_x + x_step, curr_y + y_step
            if 0 <= new_x < matrix.shape[0] and 0 <= new_y < matrix.shape[1]:
                if (new_x, new_y) not in visited and matrix[new_x][new_y] in (0, 3):
                    new_distance = distance + (x_step**2 + y_step**2)**0.5
                    heappush(heap, (new_distance, new_x, new_y))
                    distances[(new_x, new_y)] = (new_distance, (curr_x, curr_y))
    visited.append((curr_x, curr_y))

for x, y in path:
    matrix[x][y] = 4

path_img = Image.new('RGB', (width, height), color='white')
colors = []
index_to_colors = {
    #Пустое пространство - белый
    0: (255, 255, 255),
    #Пустое пространство - чёрный
    1: (0, 0, 0),
    #Стартовая точка - синий
    2: (0, 0, 255),
    #Конечная точка - зелёный
    3: (0, 255, 0),
    #Путь - белый
    4: (255, 0, 0)
}

for row in matrix:
    for paint in row:
        colors.append(index_to_colors[paint])

path_img.putdata(colors)

plt.imshow(path_img)
plt.title('Дейкстра')
plt.show()