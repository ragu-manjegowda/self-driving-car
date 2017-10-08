#!/usr/bin/env python

import rospy
from dbw_mkz_msgs.msg import ThrottleCmd, SteeringCmd, BrakeCmd
from geometry_msgs.msg import TwistStamped
from pid import PID
from std_msgs.msg import Bool
from yaw_controller import YawController


class DBWNode(object):
    def __init__(self):
        rospy.init_node('dbw_node', log_level=rospy.DEBUG)

        vehicle_mass = rospy.get_param('~vehicle_mass', 1736.35)
        fuel_capacity = rospy.get_param('~fuel_capacity', 13.5)
        brake_deadband = rospy.get_param('~brake_deadband', .1)
        decel_limit = rospy.get_param('~decel_limit', -5)
        accel_limit = rospy.get_param('~accel_limit', 1.)
        wheel_radius = rospy.get_param('~wheel_radius', 0.2413)
        wheel_base = rospy.get_param('~wheel_base', 2.8498)
        steer_ratio = rospy.get_param('~steer_ratio', 14.8)
        max_lat_accel = rospy.get_param('~max_lat_accel', 3.)
        max_steer_angle = rospy.get_param('~max_steer_angle', 8.)

        self.steer_pub = rospy.Publisher('/vehicle/steering_cmd',
                                         SteeringCmd, queue_size=1)
        self.throttle_pub = rospy.Publisher('/vehicle/throttle_cmd',
                                            ThrottleCmd, queue_size=1)
        self.brake_pub = rospy.Publisher('/vehicle/brake_cmd',
                                         BrakeCmd, queue_size=1)

        self.current_twist = None
        self.target_twist = None

        # FIXME: min speed ?
        min_speed = 1.0

        self.yaw_controller = YawController(wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle)

        self.pid = PID(2, 0.01, 0.005)

        rospy.Subscriber('/vehicle/dbw_enabled', Bool, self.dbw_cb)
        rospy.Subscriber('/current_velocity', TwistStamped, self.current_cb)
        rospy.Subscriber('/twist_cmd', TwistStamped, self.target_cb)

        self.counter = 0

        self.loop()

    def current_cb(self, msg):
        self.current_twist = msg

    def target_cb(self, msg):
        self.target_twist = msg

    def dbw_cb(self, msg):
        rospy.logerr("Resting PID, dbw not enabled!")
        enabled = bool(msg.data)
        if not enabled:
            rospy.logerr("Reseting PID, dbw not enabled!")
            self.pid.reset()

    def loop(self):
        rate = rospy.Rate(50)  # 50Hz
        while not rospy.is_shutdown():
            throttle = 0.00  # base throttle
            brake = 0.0
            steer = 0.0
            if self.current_twist is None or self.target_twist is None:
                rospy.loginfo("No incoming twist")
                rate.sleep()
                continue

            current_velocity = self.current_twist.twist.linear.x
            current_angular_velocity = self.current_twist.twist.angular.z
            target_velocity = self.target_twist.twist.linear.x
            target_angular_velocity = self.target_twist.twist.angular.z

            steer = self.yaw_controller.get_steering(target_velocity, target_angular_velocity,
                                                     current_velocity)

            if abs(target_angular_velocity - current_angular_velocity > 0.7) or abs(
                    current_angular_velocity) > 1 and self.counter % 4 == 0:
                rospy.logerr("Hard Steering, current: %s, target:%s delta: %s",
                             current_angular_velocity, target_angular_velocity,
                             abs(current_angular_velocity - target_angular_velocity))
                steer *= 5
            elif target_angular_velocity > 0.1:  # lane drift
                steer *= 1.5

            error = (target_velocity - current_velocity) / 6  # 8 m/s -> 17 mph

            if error > 1:
                error = 1
            elif error < -1:
                error = -1

            throttle += self.pid.step(error, 0.02)
            # When throttle is negative  we may apply brake
            # brake += self.pid2.step(error, 0.02)

            if current_velocity - target_velocity > 1 or throttle <= 0:  # or target_velocity <= 2
                throttle = 0
                brake = 5000
                # rospy.logerr("Current Velocity %s, Target Velocity %s, Throttle %s, Brake %s", current_velocity,
                #              target_velocity, throttle, brake)

            if target_velocity < 0.08:
                throttle = 0
                brake = 20000
                # rospy.logerr("ROUGH BRAKE. Current Velocity %s, Target Velocity %s, Throttle %s, Brake %s",
                #              current_velocity,
                #              target_velocity, throttle, brake)
            self.counter += 1
            self.publish(throttle, brake, steer)
            rate.sleep()

    def publish(self, throttle, brake, steer):
        tcmd = ThrottleCmd()
        tcmd.enable = True
        tcmd.pedal_cmd_type = ThrottleCmd.CMD_PERCENT
        tcmd.pedal_cmd = throttle
        self.throttle_pub.publish(tcmd)

        scmd = SteeringCmd()
        scmd.enable = True
        scmd.steering_wheel_angle_cmd = steer
        self.steer_pub.publish(scmd)

        bcmd = BrakeCmd()
        bcmd.enable = True
        bcmd.pedal_cmd_type = BrakeCmd.CMD_TORQUE
        bcmd.pedal_cmd = brake
        self.brake_pub.publish(bcmd)


if __name__ == '__main__':
    DBWNode()
