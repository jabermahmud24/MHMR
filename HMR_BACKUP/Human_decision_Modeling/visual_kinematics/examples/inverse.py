# !/usr/bin/env python3

from visual_kinematics.RobotSerial import *
import numpy as np
from math import pi
import time
start_time = time.time()



def main():

    dh_params = np.array([[0.78, 0.117, 0, 0],
                          [0, 0.219, 0.5 * pi, 0],
                          [0, 0.133, -0.5 * pi, 0],
                          [0, 0.197, 0.5 * pi, 0],
                          [0, 0.1245, -0.5 * pi, 0.],
                          [0, 0.1385, 0.5 * pi,0],
                          [0,0.1665, 0, 0]]) 
    # a = np.array([0.117, 0.219, 0.133, 0.197, 0.1245, 0.1385, 0.1665])
    # d = np.array([0.78, 0, 0, 0, 0, 0, 0])
    # print(a[2])
    # alpha = np.array([0.0, pi/2, -pi/2, pi/2, -pi/2, pi/2, 0])
    # print(alpha[2])
    

    robot = RobotSerial(dh_params)


    xyz = np.array([[0.38156198], [0.33891546], [1.00623167]])
    abc = np.array([-1.3770371, 0,0])
    end = Frame.from_euler_3(abc, xyz)
    robot.inverse(end)
    cnt = 0

    print("inverse is successful: {0}".format(robot.is_reachable_inverse))
    print("axis values: \n{0}".format(robot.axis_values))
    #robot.show()
    #for i in range(len(dh_params_base)):
    # for offset in [-0.05]:
    #     cnt +=1
    #     dh_params = np.copy(dh_params_base)
    #     dh_params[2, -1] += offset    
    #     robot = RobotSerial(dh_params)
    #     joint_angle_combination = robot.inverse(end)
    #     #print("axis values: {0}".format(robot.axis_values))
    #     #robot.show()
    #     #print(cnt)
    #     print(f"{joint_angle_combination}")
        
    """
    #for offset in [-0.05, 0.05]:
     #   cnt +=1
        dh_params = np.copy(dh_params_base)
        dh_params[1, -1] += offset    
        robot = RobotSerial(dh_params)
        robot.inverse(end)

        #print("Configuration with offset {0}:".format(offset))
            #print("inverse is successful: {0}".format(robot.is_reachable_inverse))
        #print("axis values: {0}".format(robot.axis_values))
            #robot.show()
        #print(cnt)
        #print("\n")
    
        
    for offset in [-0.05, 0.05]:
        cnt +=1
        dh_params = np.copy(dh_params_base)
        dh_params[3, -1] += offset    
        robot = RobotSerial(dh_params)
        robot.inverse(end)

        #print("Configuration with offset {0}:".format(offset))
            #print("inverse is successful: {0}".format(robot.is_reachable_inverse))
        #print("axis values: {0}".format(robot.axis_values))
            #robot.show()
        #print(cnt)
        #print("\n")
        
        
    for offset in [-0.05, 0.05]:
        cnt +=1
        dh_params = np.copy(dh_params_base)
        dh_params[5, -1] += offset    
        robot = RobotSerial(dh_params)
        robot.inverse(end)

        #print("Configuration with offset {0}:".format(offset))
            #print("inverse is successful: {0}".format(robot.is_reachable_inverse))
        #print("axis values: {0}".format(robot.axis_values))
            #robot.show()
       # print(cnt)
       # print("\n")
        
    
        
        
        
        
    # example of unsuccessful inverse kinematics
    #xyz = np.array([[2.2], [0.], [1.9]])
    #end = Frame.from_euler_3(abc, xyz)
    #robot.inverse(end)

    #print("inverse is successful: {0}".format(robot.is_reachable_inverse))
"""

if __name__ == "__main__":
    main()


print("--- %s seconds ---" % (time.time() - start_time))
