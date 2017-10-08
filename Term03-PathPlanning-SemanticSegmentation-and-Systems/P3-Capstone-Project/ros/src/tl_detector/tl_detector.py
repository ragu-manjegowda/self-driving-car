#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2

STATE_COUNT_THRESHOLD = 3

RED_THRESHOLD = 19


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

        self.start_x_light = [1130.0, 1545.0, 2119.0, 2170.0, 1470.0, 815.0, 155.0, 345.0]
        self.end_x_light = [1145.0, 1560.0, 2121.0, 2175.0, 1492.0, 821.0, 161.0, 351.0]
        self.start_y_light = [1183.0, 1150.0, 1440.0, 1723.0, 2900.0, 2890.0, 2280.0, 1550.0]
        self.end_y_light = [1184.0, 1173.0, 1550.0, 1790.0, 3000.0, 2920.0, 2320.0, 1590.0]
        self.pre = 0.2
        self.pos = 1  # 5 meters tolerance will serve for classificator lag/latency
        self.y_tol = 10

        self.crop_1_x = [360, 300, 340, 330, 100, 000, 000, 000]
        self.crop_2_x = [470, 440, 680, 660, 800, 800, 800, 800]
        self.crop_1_y = [220, 210, 170, 180, 050, 000, 000, 000]
        self.crop_2_y = [345, 400, 380, 340, 300, 600, 600, 600]

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
            roi = img[self.crop_1_y[l_id]:self.crop_2_y[l_id], self.crop_1_x[l_id]:self.crop_2_x[l_id]]
            self.counter += 1
            reds = self.red_count(roi)
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
        for i in range(len(self.start_x_light)):
            min_x = min(self.start_x_light[i], self.end_x_light[i])
            max_x = max(self.start_x_light[i], self.end_x_light[i])
            min_y = min(self.start_y_light[i], self.end_y_light[i])
            max_y = max(self.start_y_light[i], self.end_y_light[i])
            if min_x - self.pre < self.x < max_x + self.pos:
                if min_y - self.y_tol < self.y < max_y + self.y_tol:
                    return i
        return None

    @staticmethod
    def red_count(img):
        height, width, _ = img.shape
        count = 0
        for i in range(height):
            for j in range(width):
                b = int(img[i][j][0])
                g = int(img[i][j][1])
                r = int(img[i][j][2])
                if b > 165 and g < 80 and r < 80:
                    count += 1
                    if count > RED_THRESHOLD:  # early stop
                        return count
        return count


if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
