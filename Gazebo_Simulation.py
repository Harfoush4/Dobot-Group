import rospy
from geometry_msgs.msg import PoseStamped
from franka_gripper.msg import MoveActionGoal,GraspActionGoal

def move_end_effector():
    rospy.init_node('end_effector_mover', anonymous=True)

    # Publisher to send commands to move the end effector
    pub = rospy.Publisher('/cartesian_impedance_example_controller/equilibrium_pose', PoseStamped, queue_size=10)
    close_gripper = rospy.Publisher("/franka_gripper/grasp/goal",GraspActionGoal, queue_size=10)

    # Ensure the publisher has time to register in ROS network
    rospy.sleep(1)
    
    #closes gripper
    close_grippermsg = GraspActionGoal()
    close_grippermsg.goal.width = 0.03
    close_grippermsg.goal.epsilon.inner = 0.003
    close_grippermsg.goal.epsilon.outer = 0.007
    close_grippermsg.goal.speed = 0.1
    close_grippermsg.goal.force = 5.0
    rospy.sleep(1)
    close_gripper.publish(close_grippermsg)
    print("Published close")

    # start from the first point
    pose = PoseStamped()
    pose.header.frame_id = "panda_link0"
    pose.pose.position.x = 0.2 
    pose.pose.position.y = -0.1
    pose.pose.position.z = 0.45
    pose.pose.orientation.x = 0
    pose.pose.orientation.y = 0
    pose.pose.orientation.z = 0
    pose.pose.orientation.w = 0.0
    # Publish the first point
    pub.publish(pose)
    rospy.sleep(1)
    print("Moved to first point.")
    
    # second point
    pose = PoseStamped()
    pose.header.frame_id = "panda_link0"
    pose.pose.position.x = 0.4
    pose.pose.position.y = -0.1
    pose.pose.position.z = 0.45
    # Publish the second point
    pub.publish(pose)
    rospy.sleep(1)
    print("Moved to the second point.")

    
    # third point
    pose = PoseStamped()
    pose.header.frame_id = "panda_link0"
    pose.pose.position.x = 0.4
    pose.pose.position.y = 0.1
    pose.pose.position.z = 0.45
    # Publish the third point
    pub.publish(pose)
    rospy.sleep(1)
    print("Moved to the third point.")
    
    
    # fourth point
    pose = PoseStamped()
    pose.header.frame_id = "panda_link0"
    pose.pose.position.x = 0.2 
    pose.pose.position.y = 0.1
    pose.pose.position.z = 0.45
    # Publish the starting position
    pub.publish(pose)
    rospy.sleep(1)
    print("Moved to the fourth point.")
    
    # back to firt point
    pose = PoseStamped()
    pose.header.frame_id = "panda_link0"
    pose.pose.position.x = 0.2
    pose.pose.position.y = -0.2
    pose.pose.position.z = 0.45
    # Publish the last point
    pub.publish(pose)
    rospy.sleep(1)
    print("Back to first point.")





    

if __name__ == '__main__':
    try:
        move_end_effector()
    except rospy.ROSInterruptException:
        pass


	

