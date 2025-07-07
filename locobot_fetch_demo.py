#!/usr/bin/env python3
"""
Control Fetch *and* LoCoBot from a single ROS node.

Fetch part  : heavily trimmed from the original demo_2.py
LoCoBot part: uses LocobotCommander from the minimal helper script
Both run concurrently in two Python threads.

Tested with ROS Noetic, Gazebo 11, moveit-python 0.5.
"""

import copy
import threading
import rospy
import actionlib
from math import sin, cos

# -------------------------------------------------------------------------
#  Fetch helpers (unchanged API, Python-3 friendly)
# -------------------------------------------------------------------------
from control_msgs.msg        import (FollowJointTrajectoryAction,
                                     FollowJointTrajectoryGoal,
                                     PointHeadAction, PointHeadGoal)
from trajectory_msgs.msg     import JointTrajectory, JointTrajectoryPoint
from move_base_msgs.msg      import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg       import PoseStamped
from moveit_msgs.msg         import PlaceLocation, MoveItErrorCodes
from moveit_python           import (MoveGroupInterface,
                                     PlanningSceneInterface,
                                     PickPlaceInterface)
from moveit_python.geometry  import rotate_pose_msg_by_euler_angles
from grasping_msgs.msg       import (FindGraspableObjectsAction,
                                     FindGraspableObjectsGoal)

# -------------------------------------------------------------------------
#  LoCoBot helpers (verbatim from previous script)
# -------------------------------------------------------------------------
from geometry_msgs.msg import Twist
from std_msgs.msg      import Float64
from rosgraph_msgs.msg import Clock


# --------------------------  FETCH SECTION --------------------------------
class MoveBaseClient:
    def __init__(self):
        self.client = actionlib.SimpleActionClient("move_base", MoveBaseAction)
        rospy.loginfo("Waiting for move_base action…")
        self.client.wait_for_server()

    def goto(self, x, y, theta, frame="map"):
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = frame
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = x
        goal.target_pose.pose.position.y = y
        goal.target_pose.pose.orientation.z = sin(theta / 2.0)
        goal.target_pose.pose.orientation.w = cos(theta / 2.0)
        self.client.send_goal(goal)
        self.client.wait_for_result()


class FollowTrajectoryClient:
    def __init__(self, name, joint_names):
        self.client = actionlib.SimpleActionClient(
            f"{name}/follow_joint_trajectory", FollowJointTrajectoryAction
        )
        rospy.loginfo(f"Waiting for {name} trajectory controller…")
        self.client.wait_for_server()
        self.joint_names = joint_names

    def move_to(self, positions, duration=5.0):
        if len(positions) != len(self.joint_names):
            rospy.logerr("Trajectory length mismatch")
            return False
        traj = JointTrajectory(joint_names=self.joint_names)
        point = JointTrajectoryPoint(
            positions=positions,
            velocities=[0.0] * len(positions),
            accelerations=[0.0] * len(positions),
            time_from_start=rospy.Duration(duration),
        )
        traj.points.append(point)
        goal = FollowJointTrajectoryGoal(trajectory=traj)
        self.client.send_goal(goal)
        self.client.wait_for_result()
        return True


class PointHeadClient:
    def __init__(self):
        self.client = actionlib.SimpleActionClient(
            "head_controller/point_head", PointHeadAction
        )
        rospy.loginfo("Waiting for head_controller…")
        self.client.wait_for_server()

    def look_at(self, x, y, z, frame="map", duration=1.0):
        goal = PointHeadGoal()
        goal.target.header.frame_id = frame
        goal.target.header.stamp = rospy.Time.now()
        goal.target.point.x, goal.target.point.y, goal.target.point.z = x, y, z
        goal.min_duration = rospy.Duration(duration)
        self.client.send_goal(goal)
        self.client.wait_for_result()


class GraspingClient:
    def __init__(self):
        self.scene      = PlanningSceneInterface("base_link")
        self.pickplace  = PickPlaceInterface("arm", "gripper", verbose=False)
        self.move_group = MoveGroupInterface("arm", "base_link")

        topic = "basic_grasping_perception/find_objects"
        rospy.loginfo(f"Waiting for {topic} …")
        self.find_client = actionlib.SimpleActionClient(topic,
                                                        FindGraspableObjectsAction)
        self.find_client.wait_for_server()

        self.objects  = []
        self.surfaces = []
        self.pick_res = None

    # ----- perception / scene helpers ----------------------------------
    def update_scene(self):
        goal = FindGraspableObjectsGoal(plan_grasps=True)
        self.find_client.send_goal(goal)
        self.find_client.wait_for_result(rospy.Duration(5.0))
        res = self.find_client.get_result()

        # clear previous
        for n in self.scene.getKnownCollisionObjects():
            self.scene.removeCollisionObject(n, use_service=False)
        for n in self.scene.getKnownAttachedObjects():
            self.scene.removeAttachedObject(n, use_service=False)
        self.scene.waitForSync()

        # add newly detected objects
        for idx, obj in enumerate(res.objects):
            obj.object.name = f"object{idx}"
            self.scene.addSolidPrimitive(obj.object.name,
                                         obj.object.primitives[0],
                                         obj.object.primitive_poses[0],
                                         use_service=False)
        # add support surfaces
        for surf in res.support_surfaces:
            h = surf.primitive_poses[0].position.z
            surf.primitives[0].dimensions[1] = 1.5          # widen
            surf.primitives[0].dimensions[2] += h           # extend
            surf.primitive_poses[0].position.z -= h / 2.0
            self.scene.addSolidPrimitive(surf.name,
                                         surf.primitives[0],
                                         surf.primitive_poses[0],
                                         use_service=False)
        self.scene.waitForSync()
        self.objects, self.surfaces = res.objects, res.support_surfaces

    def _first_cube(self):
        for o in self.objects:
            if not o.grasps:
                continue
            d = o.object.primitives[0].dimensions[0]
            if 0.05 <= d <= 0.07 and o.object.primitive_poses[0].position.z > 0.5:
                return o.object, o.grasps
        return None, None

    # ----- pick & place wrappers ---------------------------------------
    def pick(self, block, grasps):
        ok, self.pick_res = self.pickplace.pick_with_retry(
            block.name, grasps,
            support_name=block.support_surface,
            scene=self.scene)
        return ok

    def place(self, block, pose_stamped):
        locs = []
        l = PlaceLocation()
        l.place_pose.header.frame_id = pose_stamped.header.frame_id
        l.place_pose.pose            = pose_stamped.pose
        l.post_place_posture         = self.pick_res.grasp.pre_grasp_posture
        l.pre_place_approach         = self.pick_res.grasp.pre_grasp_approach
        l.post_place_retreat         = self.pick_res.grasp.post_grasp_retreat
        locs.append(copy.deepcopy(l))
        # add yaw-rotated variants
        m, pi = 16, 3.14159265359
        for i in range(m - 1):
            l.place_pose.pose = rotate_pose_msg_by_euler_angles(
                l.place_pose.pose, 0, 0, 2 * pi / m)
            locs.append(copy.deepcopy(l))
        ok, _ = self.pickplace.place_with_retry(block.name, locs, scene=self.scene)
        return ok

    def tuck(self):
        joints = ["shoulder_pan_joint", "shoulder_lift_joint", "upperarm_roll_joint",
                  "elbow_flex_joint", "forearm_roll_joint", "wrist_flex_joint",
                  "wrist_roll_joint"]
        pose   = [1.32, 1.40, -0.2, 1.72, 0.0, 1.66, 0.0]
        while not rospy.is_shutdown():
            res = self.move_group.moveToJointPosition(joints, pose, 0.02)
            if res.error_code.val == MoveItErrorCodes.SUCCESS:
                return


# --------------------------  LOCOBOT SECTION ------------------------------
class LocobotCommander:
    """Minimal helper exactly as before (Python 3)."""
    def __init__(self):
        rospy.loginfo("Waiting for /clock …")
        rospy.wait_for_message("/clock", Clock)
        rospy.loginfo("/clock publishing – LoCoBot up")

        self.base_pub = rospy.Publisher("/locobot/cmd_vel",
                                        Twist, queue_size=10)
        jt = ["waist", "shoulder", "elbow",
              "wrist_angle", "wrist_rotate",
              "left_finger", "right_finger",
              "pan", "tilt"]
        self.joint_pubs = {j: rospy.Publisher(
            f"/locobot/{j}_controller/command", Float64, queue_size=10)
            for j in jt}

    # --- base
    def move_base(self, vx=0.0, vth=0.0, dur=0.0, rate_hz=10):
        twist = Twist()
        twist.linear.x, twist.angular.z = vx, vth
        r     = rospy.Rate(rate_hz)
        end_t = rospy.Time.now() + rospy.Duration.from_sec(dur)
        while not rospy.is_shutdown() and rospy.Time.now() < end_t:
            self.base_pub.publish(twist)
            r.sleep()
        self.base_pub.publish(Twist())
        rospy.sleep(0.1)

    # --- arm
    def _set(self, j, v):
        if j in self.joint_pubs:
            self.joint_pubs[j].publish(Float64(v))

    def set_arm(self, waist=0, shoulder=0, elbow=0,
                wrist_angle=0, wrist_rotate=0, grip_open=True):
        self._set("waist", waist)
        self._set("shoulder", shoulder)
        self._set("elbow", elbow)
        self._set("wrist_angle", wrist_angle)
        self._set("wrist_rotate", wrist_rotate)
        g = 0.1 if grip_open else 0.0
        self._set("left_finger", g)
        self._set("right_finger", g)

    # --- simple demo
    def demo(self):
        rospy.loginfo("LoCoBot ⇢ forward 0.3 m/s for 3 s")
        self.move_base(vx=0.3, dur=3.0)
        rospy.loginfo("LoCoBot ⇢ spin 0.6 rad/s for 2 s")
        self.move_base(vth=0.6, dur=2.0)
        rospy.loginfo("LoCoBot ⇢ arm to pick pose")
        self.set_arm(waist=0.5, shoulder=-0.4, elbow=0.4,
                     wrist_angle=1.1, wrist_rotate=0.0, grip_open=True)
        rospy.sleep(2.0)
        rospy.loginfo("LoCoBot ⇢ close gripper")
        self.set_arm(grip_open=False)
        rospy.sleep(1.0)
        rospy.loginfo("LoCoBot demo ✔")


# --------------------------  WRAPPER DEMOS ---------------------------------
class FetchDemo:
    def __init__(self):
        self.move_base  = MoveBaseClient()
        self.torso      = FollowTrajectoryClient("torso_controller",
                                                 ["torso_lift_joint"])
        self.head       = PointHeadClient()
        self.grasp      = GraspingClient()

    def run(self):
        rospy.loginfo("[Fetch] Raise torso")
        self.torso.move_to([0.3])
        # navigate to first table
        self.move_base.goto(2.25, 3.118, 0.0)
        self.move_base.goto(2.75, 3.118, 0.0)
        # look at table centre
        self.head.look_at(3.7, 3.18, 0.5)
        # attempt pick
        while not rospy.is_shutdown():
            self.grasp.update_scene()
            cube, grasps = self.grasp._first_cube()
            if cube and self.grasp.pick(cube, grasps):
                break
            rospy.logwarn("[Fetch] pick failed – retrying")
        self.grasp.tuck()
        self.torso.move_to([0.0])

        # drive to second table
        self.move_base.goto(-3.53, 3.75, 1.57)
        self.move_base.goto(-3.53, 4.15, 1.57)
        self.torso.move_to([0.4])           # raise again
        # place
        while not rospy.is_shutdown():
            pose = PoseStamped()
            pose.header.frame_id = cube.header.frame_id
            pose.pose            = cube.primitive_poses[0]
            pose.pose.position.z += 0.05
            if self.grasp.place(cube, pose):
                break
            rospy.logwarn("[Fetch] place failed – retrying")
        self.grasp.tuck()
        self.torso.move_to([0.0])
        rospy.loginfo("Fetch demo ✔")


class LoCoDemo:
    def __init__(self):
        self.bot = LocobotCommander()

    def run(self):
        self.bot.demo()


# --------------------------  MAIN ENTRY ------------------------------------
def main():
    rospy.init_node("dual_robot_demo")
    # block until /use_sim_time clocks ticks (sim) or immediately (real)
    while rospy.Time.now().to_sec() == 0.0 and not rospy.is_shutdown():
        pass

    fetch_demo = FetchDemo()
    loco_demo  = LoCoDemo()

    # run both robots concurrently
    tfetch = threading.Thread(target=fetch_demo.run, name="FetchThread")
    tloco  = threading.Thread(target=loco_demo.run,  name="LoCoThread")
    tfetch.start()
    tloco.start()
    tfetch.join()
    tloco.join()
    rospy.loginfo("🎉 Both demos finished – exiting")


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
