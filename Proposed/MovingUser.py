'''
Args:
속도 벡터를 가지는 미친 ground object

Returns:
나에게 주어지지 않는 긴 학습 시간
'''


def set_moving_ground_objects(Lx, Ly, num_objects=30):
    file_path = './Environ/ground_objects.json'
    
    if os.path.isfile(file_path):
        with open(file_path, 'r') as file:
            ground_objects = json.load(file)
    else:
        ground_objects = set_gobjects(Lx, Ly, num_objects)
    
    for i in range(ground_objects):
        ground_objects[i]
    
    return ground_objects
