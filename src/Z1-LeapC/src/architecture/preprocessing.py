import numpy as np

def vec_to_np(vec):
    return np.array([vec.x, vec.y, vec.z], dtype=float)

epsilon = 1e-8

def extract_features(hand, prev_data, prev_data_2):
    # 19 features, (5*3 tips) + (1*3 local velocity) + 1 scalar speed, trying feature extraction instead of real world data
    try:

        def bone_direction(bone):
            diff = vec_to_np(bone.next_joint) - vec_to_np(bone.prev_joint)
            norm = np.linalg.norm(diff)
            if norm < epsilon:
                return np.zeros(3)
            return diff / norm

        palm = vec_to_np(hand.palm.position)
        normal = vec_to_np(hand.palm.normal)
        direction = vec_to_np(hand.palm.direction)

        if np.linalg.norm(normal) < epsilon or np.linalg.norm(normal) < epsilon:
            print("Normal or direction is too small")
            return np.zeros(19) 

        #Local frame
        right = np.cross(direction, normal)
        right_norm = np.linalg.norm(right)
        if right_norm < epsilon:
            print("Right is too small")
            return np.zeros(19)

        right = right / right_norm
        R = np.column_stack([right, normal, direction])

        fingerfeatures = []
        for digit in hand.digits:
            globaltip = vec_to_np(digit.bones[3].next_joint)
            reltip = globaltip - palm
            localtip = R.T @ reltip
            fingerfeatures.extend(localtip)
        
        palmvellocal = np.zeros(3)
        speed =0.0

        if prev_data is not None:
            prev_palm = vec_to_np(prev_data.palm.position)
            palm_vel = (palm - prev_palm)
            palmvellocal = R.T @ palm_vel
            speed = np.linalg.norm(palm_vel)

        feature = np.concatenate([np.array(fingerfeatures), palmvellocal, [speed]])

        if len(feature) != 19:
            print("Feature length is not 19")
            return np.zeros(19)
        
        return feature
    
    except Exception as e:
        print(f"Error extracting features: {e}")
        return np.zeros(19)

def twohand_extract_feature(hand1, hand2):
    
    palm1 = vec_to_np(hand1.palm.position)
    palm2 = vec_to_np(hand2.palm.position)
    palm1vel = vec_to_np(hand1.palm.velocity)
    palm2vel = vec_to_np(hand2.palm.velocity)

    palmdist = np.linalg.norm(palm1 - palm2) / 100.0
    relvelvec = (palm1vel - palm2vel) / 100.0
    relvelnorm = np.linalg.norm(relvelvec)

    relpos = palm1 - palm2
    approachspeed = np.dot(relvelvec, relpos / (np.linalg.norm(relpos) + epsilon))
    feature = np.array([palmdist, relvelnorm, approachspeed]) #pad to 4
    return feature

