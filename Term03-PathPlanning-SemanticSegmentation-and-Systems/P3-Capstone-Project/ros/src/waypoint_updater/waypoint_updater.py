#!/usr/bin/env python

import copy
import math
import time

import rospy
from geometry_msgs.msg import TwistStamped, PoseStamped
from std_msgs.msg import Int32
from styx_msgs.msg import Lane

LOOKAHEAD_WPS = 200  # Number of waypoints we will publish. You can change this number
CRUISE = 0
CROSS = 1
STOP = 2
RUN = 3
CRUISE_SPEED = 4.4
CROSS_SPEED = 3
RUN_SPEED = 6
STOP_WPS = 100  # Fixed length stop waypoints speeds


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater', log_level=rospy.DEBUG)

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/current_velocity', TwistStamped, self.current_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        self.init = False
        self.current_pose = None
        self.wps = None
        self.N = None  # number of waypoints
        self.final_wps = None
        self.speed = None

        # lights and light wps
        self.light_wps_ids = []
        self.light_pos = [(1148.56, 1184.65), (1559.2, 1158.43), (2122.14, 1526.79), (2175.237, 1795.71),
                          (1493.29, 2947.67), (821.96, 2905.8), (161.76, 2303.82), (351.84, 1574.65)]
        self.K = len(self.light_pos)  # Number of crosses
        self.start_x_light = [1130.0, 1545.0, 2119.0, 2170.0, 1470.0, 815.0, 155.0, 345.0]
        self.end_x_light = [1145.0, 1560.0, 2121.0, 2175.0, 1492.0, 821.0, 161.0, 351.0]

        # state logic
        self.state = CRUISE
        self.cross_id = None
        self.cross_wp = None
        self.car_wp = None

        # stop
        self.stop = False
        self.white_line_wp_id = [291, 753, 2040, 2573, 2600, 2600, 2600, 2600]

        # CV
        self.cam_stop = True  # start red
        self.first_green_after_red = False
        self.last_traffic_update = None
        self.last_wp_tl = None

        # for debug messages
        self.counter = 0

        self.stop_speeds = []  # list of 100 WPS speeds
        self.calculate_stop_speeds()

        self.loop()

    def pose_cb(self, msg):
        self.current_pose = msg

    def waypoints_cb(self, msg):
        if self.init:
            return
        self.wps = msg
        self.N = len(self.wps.waypoints)
        for wp in self.wps.waypoints:
            wp.twist.twist.linear.x = CRUISE_SPEED  # set to 17 MPH, no more than this
        self.set_all_cross_wps()

    def current_cb(self, msg):
        self.speed = msg.twist.linear.x

    def traffic_cb(self, msg):
        self.last_traffic_update = time.clock()
        self.last_wp_tl = self.car_wp
        if int(msg.data) == 1:
            self.cam_stop = True
            # rospy.logerr('red light')
        else:
            # rospy.logerr('green light')
            if self.cam_stop:
                self.first_green_after_red = True  # Start only after the first Green - Latency issues.
            else:
                self.first_green_after_red = False
            self.cam_stop = False
        if self.wps is not None and self.current_pose is not None:
            self.init = True

    @staticmethod
    def dist(a_x, a_y, b_x, b_y):
        return math.sqrt((a_x - b_x) ** 2 + (a_y - b_y) ** 2)

    def calculate_stop_speeds(self):
        v0 = CROSS_SPEED
        for i in range(STOP_WPS):
            speed = v0 - v0 * i / float(STOP_WPS - 1)
            self.stop_speeds.append(speed)

    def is_traffic_fresh(self):
        if time.clock() - self.last_traffic_update < 0.3:  # with 1 it was never stale
            return True
        rospy.logerr("Stale traffic light info")
        return False

    def is_traffic_info_same_wp(self):
        if self.car_wp == self.last_wp_tl:
            return True  # Only pay attentions to traffic information coming in the same wp
        rospy.logerr("Way past CVs, car wp: %s, tl cv wp: %s", self.car_wp, self.last_wp_tl)
        return False

    def close_to_white_lane(self):
        if abs(self.car_wp - self.white_line_wp_id[self.cross_id]) < 20:
            return True
        rospy.logerr("We're not near white line yet")
        return False

    def get_closest_wp_index(self, x, y):
        max_distance_so_far = 100  # 50000 meters away
        best_i = None
        for i in range(self.N):
            wp = self.wps.waypoints[i]
            distance = self.dist(wp.pose.pose.position.x, wp.pose.pose.position.y, x, y)
            if distance < max_distance_so_far:
                max_distance_so_far = distance
                best_i = i
        return best_i

    def set_all_cross_wps(self):
        for i in range(self.K):
            light_pos_x, light_pos_y = self.light_pos[i]
            self.light_wps_ids.append(self.get_closest_wp_index(
                light_pos_x,
                light_pos_y))

    def nearest_cross_id(self):
        for i in range(self.K):
            cross_wp = self.wps.waypoints[self.light_wps_ids[i]]
            distance_to_start_cross = self.dist(cross_wp.pose.pose.position.x,
                                                cross_wp.pose.pose.position.y,
                                                self.current_pose.pose.position.x,
                                                self.current_pose.pose.position.y)
            if distance_to_start_cross < 70:  # meters
                return i
        return None

    def cross_distance(self):
        return self.dist(self.cross_wp.pose.pose.position.x,
                         self.cross_wp.pose.pose.position.y,
                         self.current_pose.pose.position.x,
                         self.current_pose.pose.position.y)

    def update_state_values(self):
        if not self.init:
            return
        self.car_wp = self.get_closest_wp_index(self.current_pose.pose.position.x, self.current_pose.pose.position.y)
        self.cross_id = self.nearest_cross_id()
        if self.cross_id is not None:
            self.cross_wp = self.wps.waypoints[self.light_wps_ids[self.cross_id]]
        else:
            self.cross_wp = None

    def set_final_wps(self):
        wps = []
        if self.state == CRUISE:
            wps = self.wps.waypoints[
                  self.car_wp:self.car_wp + LOOKAHEAD_WPS]  # Don't forget modulo N
        elif self.state == CROSS:
            for i in range(LOOKAHEAD_WPS):  # cross mode ENDS at stop_wp
                wps.append(copy.deepcopy(self.wps.waypoints[self.car_wp + i]))
                wps[-1].twist.twist.linear.x = CROSS_SPEED
        elif self.state == STOP:
            m = self.white_line_wp_id[self.cross_id] - self.car_wp
            if m <= 0:
                rospy.logerr("Error, in stop mode, m<0, passed cross?")
                return
            for i in range(m):
                wps.append(copy.deepcopy(self.wps.waypoints[self.car_wp + i]))
                wps[-1].twist.twist.linear.x = self.stop_speeds[STOP_WPS - m + i]
        elif self.state == RUN:
            for i in range(LOOKAHEAD_WPS):  # cross mode ENDS at stop_wp
                wps.append(copy.deepcopy(self.wps.waypoints[self.car_wp + i]))
                wps[-1].twist.twist.linear.x = RUN_SPEED
        self.final_wps = wps

    def in_camera_interval(self):
        if self.cross_id is None:
            rospy.logerr('what? no light id!')
        x = self.current_pose.pose.position.x
        margin = 0
        if self.speed < 0.5:  # First light
            margin = 13
        if self.start_x_light[self.cross_id] + margin < x < self.end_x_light[self.cross_id]:
            rospy.logerr('In camera range for light: %s', self.cross_id)
            return True
        return False

    def is_really_green(self):
        return self.first_green_after_red and self.is_traffic_fresh() \
               and self.close_to_white_lane() and self.is_traffic_info_same_wp() and not self.cam_stop

    def update_final_wps(self):
        if not self.init:
            rospy.logerr('not init')
            return
        self.update_state_values()
        if self.counter % 10 == 0:
            rospy.logerr('wp: %s', self.car_wp)
        if self.state == CRUISE:
            if self.cross_id is not None:
                self.state = CROSS
                rospy.logerr("Switching from Cruise to Near Cross")
                return
        elif self.state == CROSS:
            if self.in_camera_interval():  # If we pass this mark either stop or run. Never go back to cross.
                if self.is_really_green():
                    self.state = RUN
                    rospy.logerr("Green light switching to RUN- take off!")
                    return
                else:
                    self.state = STOP
                    rospy.logerr("Switching to stop mode")
                    return
        elif self.state == STOP:
            if self.is_really_green():
                rospy.logerr("First Green light, takeoff!")
                self.state = RUN
                return
        elif self.state == RUN:
            self.stop = False
            if self.cross_id is None:
                rospy.logerr("We passed the light, cruise now!")
                self.state = CRUISE
                return
        self.set_final_wps()

    def loop(self):
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            self.counter += 1
            self.update_final_wps()
            if self.final_wps is not None:
                lane = Lane()
                lane.header.frame_id = '/world'
                lane.header.stamp = rospy.Time(0)
                lane.waypoints = self.final_wps
                self.final_waypoints_pub.publish(lane)
            rate.sleep()


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
