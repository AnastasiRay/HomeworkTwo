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

def heuristic(node, goal):
    return abs(node[0] - goal[0]) + abs(node[1] - goal[1])

def astar(matrix, start, end):
    heap = []
    heappush(heap, (0, start))
    distances = {start: 0}
    path = {start: []}
    visited = set()
    while heap:
        (distance, current) = heappop(heap)
        if current == end:
            return path[current] + [current]
        visited.add(current)
        for x_step, y_step in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            if x_step == 0 and y_step == 0:
                continue
            neighbor = current[0] + x_step, current[1] + y_step
            if 0 <= neighbor[0] < matrix.shape[0] and 0 <= neighbor[1] < matrix.shape[1]:
                if matrix[neighbor] in (0, 3) and neighbor not in visited:
                    new_distance = distances[current] + 1
                    if neighbor not in distances or new_distance < distances[neighbor]:
                        distances[neighbor] = new_distance
                        priority = new_distance + heuristic(end, neighbor)
                        heappush(heap, (priority, neighbor))
                        path[neighbor] = path[current] + [current]
    return None

start = start_x, start_y
end = end_x, end_y

path = astar(matrix, start, end)[1:-1]

if path:
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
plt.title('А-стар')
plt.show()