#!/usr/bin/env python3
import rospy
import asyncio
import websockets
import json
import threading 
from std_msgs.msg import String, Int32, Float32
from geometry_msgs.msg import Vector3, Point 

URI = "ws://192.168.1.130:8765"
pub_gesture = None
pub_vel = None
pub_targetpos = None
pub_gripper = None
pub_conf = None
pub_phase = None

async def listener():
    global pub_gesture, pub_vel, pub_targetpos, pub_gripper, pub_conf, pub_phase

    async with websockets.connect(URI) as uri:
        rospy.loginfo(f"[z1_bridge]: Connected to {URI}")

        async for msg in uri:
            try:
                data = json.loads(msg)
                gesture = data["gesture"]
                velocity = data["velocity"]
                targetpos = data["targetpos"]
                gripper = data["grippercmd"]
                conf = data["conf"]
                phase = data["phase"]

                pub_gesture.publish(gesture)
                pub_vel.publish(Vector3(*velocity))
                pub_targetpos.publish(Point(*targetpos))
                pub_gripper.publish(int(gripper))
                pub_conf.publish(float(conf))
                pub_phase.publish(float(phase))

            except Exception as e:
                rospy.logerr(f"[z1_bridge]: Error parsing {e}")

def start_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_until_complete(listener())

def main():
    global pub_gesture, pub_vel, pub_targetpos, pub_gripper
    rospy.init_node("z1_bridge")

    pub_gesture = rospy.Publisher("/z1/gesture", String, queue_size=10)
    pub_vel = rospy.Publisher("/z1/velocitycmd", Vector3, queue_size=10)
    pub_targetpos = rospy.Publisher("/z1/targetpos", Point, queue_size=10)
    pub_gripper = rospy.Publisher("/z1/grippercmd", Int32, queue_size=10)
    pub_conf = rospy.Publisher("/z1/conf", Float32, queue_size=10)
    pub_phase = rospy.Publisher("/z1/phase", Float32 , queue_size=10)
    
    rospy.loginfo("[z1_bridge]: Starting listener thread")
    loop = asyncio.new_event_loop()
    threading.Thread(target=start_loop, args=(loop,), daemon=True).start()
    rospy.spin()

if __name__ == "__main__":
    main()
