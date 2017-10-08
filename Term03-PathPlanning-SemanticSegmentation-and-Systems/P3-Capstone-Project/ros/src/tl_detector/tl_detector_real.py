#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import math

STATE_COUNT_THRESHOLD = 3

RED_THRESHOLD = 200


class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.waypoints = None
        self.camera_image = None
        self.lights = []

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        sub6 = rospy.Subscriber('/camera/image_raw', Image, self.image_cb)
        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)
        self.bridge = CvBridge()

        self.x = None
        self.y = None
        self.counter = 0

        self.start_x_light = -0.837
        self.end_x_light = 8.95
        self.start_y_light = 15.79
        self.end_y_light = 18.03
        self.pre = 0.2
        self.pos = 1  # 5 meters tolerance will serve for classificator lag/latency
        self.y_tol = 10

        self.first = True
        self.long_sleep_rate = rospy.Rate(1)

        rospy.spin()

    def pose_cb(self, msg):
        self.x = float(msg.pose.position.x)
        self.y = float(msg.pose.position.y)

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        if self.x is None:
            return
        l_id = self.near_light_id()
        if l_id is None:
            self.long_sleep_rate.sleep()
            # self.upcoming_red_light_pub.publish(Int32(0)) If there is no information
            return
        if self.first:
            self.first = False  # First image from sim is black!
            return
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            y1, y2, x1, x2 = self.crop_points(self.dist_to_ref())
            roi = img[y1:y2, x1:x2]
            self.counter += 1
            reds = self.glow_windows(roi)
            if reds >= RED_THRESHOLD:
                self.upcoming_red_light_pub.publish(Int32(1))
                # cv2.imwrite(
                #     '/home/crised/sdcnd/term3/pics/' + str(self.counter) + '_' + str(self.x) + '_' + str(
                #         self.y) + '_' + str(l_id) + '_' + str(reds) + '_RED' + '.jpeg', roi)
                self.counter += 1
            else:
                # cv2.imwrite(
                #     '/home/crised/sdcnd/term3/pics/' + str(self.counter) + '_' + str(self.x) + '_' + str(
                #         self.y) + '_' + str(l_id) + '_' + str(reds) + '_GREEN' + '.jpeg', roi)
                self.upcoming_red_light_pub.publish(Int32(-1))
        except CvBridgeError, e:
            rospy.logerr('error %s', e)

    def near_light_id(self):
        if self.start_x_light - self.pre < self.x < self.end_x_light + self.pos:
            if self.start_y_light - self.y_tol < self.y < self.end_y_light + self.y_tol:
                return 0
        return None

    def dist_to_ref(self):
        return math.sqrt((self.start_x_light - self.x) ** 2 + (self.start_y_light - self.y) ** 2)

    @staticmethod
    def crop_points(dist):
        rect = int(200 + 10 * dist)
        y1 = int(412 - 4.3 * dist)
        y2 = int(y1 + 0.2 * rect)
        x1 = int(647 - 8 * dist)
        x2 = int(x1 + 1.2 * rect)
        # roi = img[left_y:left_y+int(0.2*rec), left_x:left_x+int(1.2*rec)] # im[y1:y2, x1:x2] 1-> upper left corner 2 -> bottom right corner
        return y1, y2, x1, x2

    @staticmethod
    def glow_windows(img):
        height = img.shape[0]
        width = img.shape[1]
        windows = [0]
        for i in range(1, 5):
            windows.append(int(i * width / 4.0))
        results = []
        for z in range(4):
            glows = 0  # high intensity color
            for i in range(height):
                for j in range(windows[z], windows[z + 1]):
                    if (float(img[i][j][1]) + float(img[i][j][2])) / 2 > 230.0:
                        glows += 1
            results.append(glows)
        results.sort()
        if results[-2] > 150:  # if there are 2 glowing windows, probably we're talking about a bright image
            return 0
        return max(results)


if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
