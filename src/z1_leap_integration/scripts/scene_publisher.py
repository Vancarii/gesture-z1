#!/usr/bin/python3
"""
ROS node: scene_publisher

Adds static collision objects to the MoveIt planning scene so the motion planner avoids them

Objects are defined as ROS params under 'scene_publisher/collision_objects'
(loaded from gesture_config.yaml).  The node republishes every
REPUBLISH_INTERVAL_S seconds so objects survive any planning-scene resets
(e.g. if move_group is restarted mid-session).

Each object entry supports:
    name        (str, required)  – unique ID in the planning scene
    position    ([x, y, z])      – metres in 'world' frame         [default: 0,0,0]
    orientation ([x, y, z, w])   – quaternion                      [default: 0,0,0,1]
    size        ([x, y, z])      – box dimensions in metres        [default: 0.1,0.1,0.1]

Objects that move with the arm are defined under 'scene_publisher/attached_objects'.
Each entry supports:
    name        (str, required)  – unique ID
    link        (str, required)  – robot link to attach to (e.g. 'link06')
    position    ([x, y, z])      – metres in the named link's frame [default: 0,0,0]
    orientation ([x, y, z, w])   – quaternion in link frame          [default: 0,0,0,1]
    size        ([x, y, z])      – box dimensions in metres          [default: 0.1,0.1,0.1]
    touch_links ([str, ...])     – links allowed to contact this object [default: [link]]

Allowed-collision pairs are disabled between objects listed in 'scene_publisher/allowed_collisions'
"""

import sys

import rospy
import moveit_commander
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import PlanningScene

REFERENCE_FRAME = "world"
STARTUP_DELAY_S = 4.0
REPUBLISH_INTERVAL_S = 15.0

# Helpers

def _pose(position, orientation):
    ps = PoseStamped()
    ps.header.frame_id    = REFERENCE_FRAME
    ps.pose.position.x    = float(position[0])
    ps.pose.position.y    = float(position[1])
    ps.pose.position.z    = float(position[2])
    ps.pose.orientation.x = float(orientation[0])
    ps.pose.orientation.y = float(orientation[1])
    ps.pose.orientation.z = float(orientation[2])
    ps.pose.orientation.w = float(orientation[3])
    return ps


def _add_object(scene, obj):
    """Add or refresh one box collision object in the planning scene."""
    name        = obj["name"]
    position    = obj.get("position",    [0.0, 0.0, 0.0])
    orientation = obj.get("orientation", [0.0, 0.0, 0.0, 1.0])
    size        = obj.get("size",        [0.1, 0.1, 0.1])
    pose        = _pose(position, orientation)

    scene.add_box(name, pose, size=tuple(float(v) for v in size))

    rospy.logdebug(
        f"[scene_publisher] '{name}' @ "
        f"[{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}]"
    )


def _attach_object(scene, obj):
    """Attach a box collision object to a robot link so it moves with the arm."""
    name        = obj["name"]
    link        = obj["link"]
    position    = obj.get("position",    [0.0, 0.0, 0.0])
    orientation = obj.get("orientation", [0.0, 0.0, 0.0, 1.0])
    size        = obj.get("size",        [0.1, 0.1, 0.1])
    touch_links = obj.get("touch_links", [link])

    # Pose is expressed in the attached link's own frame
    pose = _pose(position, orientation)
    pose.header.frame_id = link

    scene.attach_box(
        link, name,
        pose=pose,
        size=tuple(float(v) for v in size),
        touch_links=[str(l) for l in touch_links],
    )

    rospy.loginfo(
        f"[scene_publisher] attached '{name}' to '{link}' "
        f"@ [{position[0]:.4f}, {position[1]:.4f}, {position[2]:.4f}]"
    )


def _apply_allowed_collisions(scene, pairs):
    """
    Disable collision checking between named pairs of (robot_link, scene_object).

    Pairs should be lists of two strings, e.g.:
        [["link00", "mounting_plate"], ["link00", "frame_left_post"]]
    """
    if not pairs:
        return

    # Build a PlanningScene message that carries ACM diffs
    ps = PlanningScene()
    ps.is_diff = True

    for pair in pairs:
        if len(pair) != 2:
            rospy.logwarn(
                f"[scene_publisher] allowed_collisions entry {pair} must have exactly "
                "2 elements — skipped."
            )
            continue
        rospy.loginfo(
            f"[scene_publisher] Allowing contact: '{pair[0]}' ↔ '{pair[1]}'"
        )

    # Use the high-level ACM API exposed through the planning scene monitor
    try:
        scene.add_allowed_collision_frame(
            None,  # not a box — just reuse a planningscene diff
        )
    except AttributeError:
        pass

    # Fallback: publish a raw PlanningScene diff via the internal publisher
    allowed_pub = rospy.Publisher(
        "/planning_scene", PlanningScene, queue_size=1, latch=True
    )
    rospy.sleep(0.2)

    from moveit_msgs.msg import AllowedCollisionMatrix, AllowedCollisionEntry

    # Retrieve existing ACM from move_group
    try:
        current_ps = rospy.wait_for_message(
            "/move_group/monitored_planning_scene", PlanningScene, timeout=5.0
        )
        acm_msg = current_ps.allowed_collision_matrix
    except Exception:
        rospy.logwarn("[scene_publisher] Could not fetch current ACM — building from scratch.")
        acm_msg = AllowedCollisionMatrix()

    # Index existing entries
    entry_index = {name: i for i, name in enumerate(acm_msg.entry_names)}

    def _ensure_entry(name):
        if name not in entry_index:
            entry_index[name] = len(acm_msg.entry_names)
            acm_msg.entry_names.append(name)
            n = len(acm_msg.entry_names)
            # expand all existing rows by one column
            for row in acm_msg.entry_values:
                row.enabled.append(False)
            # add new row of length n
            new_row = AllowedCollisionEntry()
            new_row.enabled = [False] * n
            acm_msg.entry_values.append(new_row)

    for pair in pairs:
        if len(pair) != 2:
            continue
        a, b = str(pair[0]), str(pair[1])
        _ensure_entry(a)
        _ensure_entry(b)
        ia, ib = entry_index[a], entry_index[b]
        acm_msg.entry_values[ia].enabled[ib] = True
        acm_msg.entry_values[ib].enabled[ia] = True

    ps.allowed_collision_matrix = acm_msg
    allowed_pub.publish(ps)
    rospy.sleep(0.5)
    rospy.loginfo(f"[scene_publisher] ACM updated for {len(pairs)} pair(s).")


def main():
    rospy.init_node("scene_publisher", anonymous=False)
    moveit_commander.roscpp_initialize(sys.argv)

    scene = moveit_commander.PlanningSceneInterface()

    rospy.loginfo(
        f"[scene_publisher] Waiting {STARTUP_DELAY_S:.0f}s for move_group …"
    )
    rospy.sleep(STARTUP_DELAY_S)

    # Load parameters 
    objs     = rospy.get_param("scene_publisher/collision_objects", [])
    pairs    = rospy.get_param("scene_publisher/allowed_collisions", [])
    attached = rospy.get_param("scene_publisher/attached_objects",   [])

    if not objs and not attached:
        rospy.logwarn(
            "[scene_publisher] No collision objects defined under "
            "'scene_publisher/collision_objects' or 'scene_publisher/attached_objects'. "
            "Add entries to gesture_config.yaml to populate the planning scene."
        )
        return

    rospy.loginfo(
        f"[scene_publisher] Publishing {len(objs)} static object(s), "
        f"{len(attached)} attached object(s), "
        f"{len(pairs)} allowed-collision override(s)."
    )

    # Publish loop
    def publish_all():
        for obj in objs:
            try:
                _add_object(scene, obj)
            except KeyError as exc:
                rospy.logerr(
                    f"[scene_publisher] Missing required key {exc} "
                    f"in object '{obj.get('name', '?')}'"
                )
            except Exception as exc:  # noqa: BLE001
                rospy.logerr(
                    f"[scene_publisher] Error adding '{obj.get('name', '?')}': {exc}"
                )

    publish_all()

    # Attach objects to robot links (done once — MoveIt tracks them from here)
    for obj in attached:
        try:
            _attach_object(scene, obj)
        except KeyError as exc:
            rospy.logerr(
                f"[scene_publisher] Missing required key {exc} "
                f"in attached object '{obj.get('name', '?')}'"
            )
        except Exception as exc:  # noqa: BLE001
            rospy.logerr(
                f"[scene_publisher] Error attaching '{obj.get('name', '?')}': {exc}"
            )

    # Apply ACM overrides once after initial object publish
    if pairs:
        rospy.sleep(1.0)  # give scene time to register objects first
        _apply_allowed_collisions(scene, pairs)

    # Periodic republish keeps objects alive after any scene reset
    timer = rospy.Timer(
        rospy.Duration(REPUBLISH_INTERVAL_S), lambda _evt: publish_all()
    )

    rospy.spin()
    timer.shutdown()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
