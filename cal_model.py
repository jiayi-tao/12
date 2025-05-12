import numpy as np

def distance(A, B):
    A = np.array(A, dtype=np.float64)
    B = np.array(B, dtype=np.float64)
    return np.linalg.norm(A - B)

def compute_angle(A, B, C):
    # Convert points to NumPy arrays for vector operations
    A = np.array(A, dtype=np.float64)
    B = np.array(B, dtype=np.float64)
    C = np.array(C, dtype=np.float64)
    
    # Compute vectors BA and BC
    BA = A - B
    BC = C - B
    
    # Compute dot product and magnitudes
    dot_product = np.dot(BA, BC)
    magnitude_BA = np.linalg.norm(BA)
    magnitude_BC = np.linalg.norm(BC)
    
    # Avoid division by zero (if points overlap)
    if magnitude_BA == 0 or magnitude_BC == 0:
        return 0.0  # Undefined angle (points coincide)
    
    # Compute cosine of the angle and clip to avoid numerical errors
    cos_theta = dot_product / (magnitude_BA * magnitude_BC)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # Ensure value is within valid range
    
    # Compute angle in radians and convert to degrees
    angle_rad = np.arccos(cos_theta)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg




def compute_2pos_angle(A, B):
    # Convert points to NumPy arrays for vector operations
    A = np.array(A, dtype=np.float64)
    B = np.array(B, dtype=np.float64)
    C = np.array([B[0], B[1]+5], dtype=np.float64)
    
    # Compute vectors BA and BC
    BA = A - B
    BC = C - B
    
    # Compute dot product and magnitudes
    dot_product = np.dot(BA, BC)
    magnitude_BA = np.linalg.norm(BA)
    magnitude_BC = np.linalg.norm(BC)
    
    # Avoid division by zero (if points overlap)
    if magnitude_BA == 0 or magnitude_BC == 0:
        return 0.0  # Undefined angle (points coincide)
    
    # Compute cosine of the angle and clip to avoid numerical errors
    cos_theta = dot_product / (magnitude_BA * magnitude_BC)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # Ensure value is within valid range
    
    # Compute angle in radians and convert to degrees
    angle_rad = np.arccos(cos_theta)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg


def compute_full_angle(A, B, C):
    # 将点转换为NumPy数组
    A = np.array(A, dtype=np.float64)
    B = np.array(B, dtype=np.float64)
    C = np.array(C, dtype=np.float64)
    print(A,B,C)
    # 计算向量BA和BC
    BA = A - B
    BC = C - B
    
    # 计算基本角度(0-180度)
    dot_product = np.dot(BA, BC)
    magnitude_BA = np.linalg.norm(BA)
    magnitude_BC = np.linalg.norm(BC)
    
    if magnitude_BA == 0 or magnitude_BC == 0:
        return 0.0
    
    cos_theta = dot_product / (magnitude_BA * magnitude_BC)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(cos_theta))
    
    # 计算叉积的Z分量来确定方向
    # 二维向量叉积的Z分量: BA_x * BC_y - BA_y * BC_x
    cross_z = BA[0] * BC[1] - BA[1] * BC[0]
    
    # 如果叉积为负，说明C在B的右侧，角度应大于180度
    if cross_z < 0:
        angle_deg = 360 - angle_deg
    
    return angle_deg


def compute_vertical_angle(A, B):
    # 转换为NumPy数组
    A = np.array(A, dtype=np.float64)
    B = np.array(B, dtype=np.float64)
    
    # 计算AB向量
    AB = B - A
    
    # 垂直向量（指向正上方）
    vertical = np.array([0, -1], dtype=np.float64)  # Y轴方向向上为负
    
    # 计算点积和模长
    dot_product = np.dot(AB, vertical)
    magnitude_AB = np.linalg.norm(AB)
    magnitude_vertical = np.linalg.norm(vertical)
    
    # 避免除以零
    if magnitude_AB == 0:
        return 0.0
    
    # 计算夹角的余弦值并限制在[-1, 1]范围内
    cos_theta = dot_product / (magnitude_AB * magnitude_vertical)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    # 计算夹角（弧度）并转换为度数
    angle_rad = np.arccos(cos_theta)
    angle_deg = np.degrees(angle_rad)
    
    # 确保返回的是最小角度（0-90度）
    if angle_deg > 90:
        angle_deg = 180 - angle_deg
    
    return angle_deg


def compute_knee_angle(hip, knee, ankle, is_side_view=True):
    # 转换为NumPy数组
    hip = np.array(hip, dtype=np.float64)
    knee = np.array(knee, dtype=np.float64)
    ankle = np.array(ankle, dtype=np.float64)
    
    # 计算向量
    thigh = hip - knee  # 大腿向量（膝盖到腰部）
    calf = ankle - knee  # 小腿向量（膝盖到脚踝）
    
    # 计算夹角的余弦值
    dot_product = np.dot(thigh, calf)
    magnitude_thigh = np.linalg.norm(thigh)
    magnitude_calf = np.linalg.norm(calf)
    
    if magnitude_thigh == 0 or magnitude_calf == 0:
        return 0.0
    
    cos_theta = dot_product / (magnitude_thigh * magnitude_calf)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    # 计算角度（0-180度）
    angle_deg = np.degrees(np.arccos(cos_theta))
    
    # 判断是否为膝关节过伸
    # 计算叉积的z分量来判断向量的相对位置
    cross_z = thigh[0] * calf[1] - thigh[1] * calf[0]
    
    # 如果是侧面视角且需要判断过伸
    if is_side_view:
        # 根据叉积判断过伸：如果叉积为负，表示大腿向量在小腿向量的顺时针方向，可能是过伸
        if cross_z < 0:
            # 过伸情况，角度大于180度
            angle_deg = 360 - angle_deg
    
    return angle_deg


# Example usage:
A = [np.float64(-45), np.float64(1.0)]
B = [np.float64(0.0), np.float64(0.0)]
C = [np.float64(384.0), np.float64(207.0)]

angle = compute_vertical_angle(A, B)
print(angle)

