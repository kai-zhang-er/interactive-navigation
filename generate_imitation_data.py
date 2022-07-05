from RRT_algo import RRT
import cv2
import random
import matplotlib.pyplot as plt

def generate_start_end_points(occupancy_map, max_iter=100):
    max_y, max_x=(occupancy_map.shape[0]-10)//10, (occupancy_map.shape[1]-10)//10

    for i in range(max_iter):
        start_x=random.randint(1, max_x)
        start_y=random.randint(1, max_y)
        if occupancy_map[start_y*10][start_x*10] > 0:
            end_x=random.randint(1, max_x)
            end_y=random.randint(1, max_y)
            if occupancy_map[end_y*10][end_x*10] > 0:
                distance=(end_x-start_x)^2+(end_y-start_y)^2
                if distance>25:
                    return start_x*10, start_y*10, end_x*10, end_y*10
    # print(start_x, start_y, end_x, end_y)
    print("not found valid starting and end points")
    return None, None, None, None


occupancy_map=cv2.imread("floor_plan_simple.png", 0)//240
num_test=500
show_animation=False
save_f=open("label.txt","w+")

test_list=[[76, 74], [93, 247], [262, 245], [383, 93],[369, 230], [233,61]]

for i in range(num_test):
    # Set Initial parameters
    if random.random()<0.1:
        start, end=random.sample(test_list, 2)
        start_x, start_y=start
        end_x, end_y=end
    else:
        start_x, start_y, end_x, end_y=generate_start_end_points(occupancy_map)
        start=[start_x, start_y]
        end=[end_x, end_y]
        if start_x is None:
            continue
    rrt = RRT(
        start=[start_x, start_y],
        goal=[end_x, end_y],
        rand_area=[10, 480],
        occupancy_map=occupancy_map,
        # play_area=[0, 10, 0, 14]
        robot_radius=8   # pixels
        )
    path = rrt.planning(animation=show_animation)

    if path is None:
        print("iter {}: Cannot find path {} -> {}".format(i,start, end))
    else:
        print("iter {}: found path {} -> {}".format(i,start, end))
        # Draw final path
        if show_animation:
            rrt.draw_graph()
            plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r')
            plt.grid(True)
            plt.pause(0.01)  # Need for Mac
            plt.show()
        
        # Draw final path
        for (x, y) in path:
            save_f.write("{},{};".format(x,y))
        save_f.write("\r")
        
        plt.imshow(occupancy_map)
        plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r')
        plt.show()

save_f.close()