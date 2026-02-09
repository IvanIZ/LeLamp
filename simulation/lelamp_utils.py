"""
LeLamp Robot Utilities - Forward Kinematics, Inverse Kinematics, and Motion Control
Handles the LeLamp robot with 'diffuser' body as end-effector.

Robot Specifications:
- Type: 5-DOF Serial Robot Arm (desk lamp)
- End-effector: 'diffuser' body (ID 6)
- Actuated Joints: "1", "2", "3", "4", "5"
- Base: scs215_v5 (fixed base body)

- Kinematic Chain (from base to end-effector, derived from robot.xml):
  * Base (scs215_v5): pos=[0, 0, 0.0418], quat=[0.15371, 0, 0, 0.988116]
  * Joint "1" (lamparm__wrist_head): axis=[-1,0,0], joint_pos=[0.04111, 0, -0.0192]
  * Joint "2" (lamparm__base_elbow): axis=[0.3038, 0.9527, 0], joint_pos=[0.00583, 0.01829, 0.08291]
  * Joint "3" (lamparm__elbow_wrist): axis=[0,0,1], joint_pos=[0,0,0]
  * Joint "4" (lamparm__wrist_head_2): axis=[0,0,1], joint_pos=[0,0,0]
  * Joint "5" (diffuser): axis=[0,0,1], joint_pos=[0,0,0]

- Joint Angle Ranges (from robot.xml, in radians):
  * Joint "1": [-5.02, 1.26] rad
  * Joint "2": [-1.08, 2.06] rad
  * Joint "3": [-2.82, 0.32] rad
  * Joint "4": [-3.79, 2.50] rad
  * Joint "5": [-0.85, 2.29] rad

Forward kinematics are derived analytically from the robot.xml body/joint definitions.
Ground truth (_gt) versions use MuJoCo's mj_forward for validation.
"""

import time
import numpy as np
import mujoco
from typing import Tuple, Optional, List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LeLampKinematics:
    """
    Kinematics solver for the LeLamp robot.
    Handles forward kinematics, inverse kinematics, and trajectory control.
    
    The analytical FK/IK functions do NOT rely on MuJoCo's physics engine
    (mj_forward) for computing body transforms. Instead, they derive the
    end-effector pose from the kinematic chain defined in robot.xml.
    
    Ground truth versions (suffixed with _gt) use MuJoCo's mj_forward
    for validation and testing purposes only.
    """
    
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        """
        Initialize the kinematics solver.
        
        Args:
            model: MuJoCo model object
            data: MuJoCo data object
        """
        self.model = model
        self.data = data
        
        # Robot configuration
        self.end_effector_name = "diffuser"
        self.base_body_name = "scs215_v5"
        
        # Joint names in MuJoCo model (these are the actuated joints)
        self.joint_names = ["1", "2", "3", "4", "5"]
        
        # Get body IDs
        self.ee_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 
                                            self.end_effector_name)
        self.base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 
                                              self.base_body_name)
        
        # Get joint IDs and their qpos addresses
        self.joint_ids = []
        self.joint_qpos_addrs = []
        for joint_name in self.joint_names:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id == -1:
                raise ValueError(f"Joint '{joint_name}' not found in model")
            self.joint_ids.append(joint_id)
            # Get qpos address for this joint
            qpos_addr = model.jnt_qposadr[joint_id]
            self.joint_qpos_addrs.append(qpos_addr)
        
        # Get actuator IDs
        self.actuator_ids = []
        for joint_name in self.joint_names:
            actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name)
            if actuator_id == -1:
                raise ValueError(f"Actuator '{joint_name}' not found in model")
            self.actuator_ids.append(actuator_id)
        
        # Joint limits (from robot.xml, in radians)
        self.joint_limits = np.array([
            [-5.021033992714066, 1.2621513144655205],   # Joint "1"
            [-1.0842575396713956, 2.0573351139183975],  # Joint "2" 
            [-2.820024081304335, 0.3215685722854582],    # Joint "3"
            [-3.7865399953934267, 2.4966453117861596],   # Joint "4"
            [-0.8533371981959901, 2.288255455393803],    # Joint "5"
        ])
        
        # Initialize kinematic chain parameters from robot.xml
        self._init_kinematic_chain()
        
        logger.info(f"LeLamp Kinematics initialized")
        logger.info(f"  End-effector: {self.end_effector_name} (body ID: {self.ee_body_id})")
        logger.info(f"  Base: {self.base_body_name} (body ID: {self.base_body_id})")
        logger.info(f"  Joints: {self.joint_names}")
        logger.info(f"  Joint qpos addresses: {self.joint_qpos_addrs}")
    
    # =========================================================================
    # Kinematic Chain Definition (from robot.xml)
    # =========================================================================
    
    def _init_kinematic_chain(self):
        """
        Initialize kinematic chain parameters extracted from robot.xml.
        
        The chain is: world -> scs215_v5 (base) -> lamparm__wrist_head (J1)
        -> lamparm__base_elbow (J2) -> lamparm__elbow_wrist (J3)
        -> lamparm__wrist_head_2 (J4) -> diffuser (J5)
        
        For each body, we store:
          - body_pos: position relative to parent body (from XML pos attribute)
          - body_quat: orientation relative to parent body (from XML quat, w,x,y,z)
          - joint_axis: hinge joint axis in body frame
          - joint_pos: hinge joint anchor position in body frame
        """
        # Base body: scs215_v5 (fixed in world, no joint)
        self._base_pos = np.array([0.0, 0.0, 0.0418])
        self._base_quat = np.array([0.15371, 0.0, 0.0, 0.988116])
        
        # Kinematic chain links (one per joint, from base to end-effector)
        self._chain = [
            {  # Joint "1": scs215_v5 -> lamparm__wrist_head
                'body_pos': np.array([0.0, -0.0192, 0.04111]),
                'body_quat': np.array([0.5, 0.5, 0.5, -0.5]),
                'joint_axis': np.array([-1.0, 0.0, 0.0]),
                'joint_pos': np.array([0.04111, 0.0, -0.0192]),
            },
            {  # Joint "2": lamparm__wrist_head -> lamparm__base_elbow
                'body_pos': np.array([0.08291, 2.965691e-08, -0.019199976]),
                'body_quat': np.array([0.570913, 0.417203, -0.570913, -0.417203]),
                'joint_axis': np.array([0.30376662, 0.95274647, 0.0]),
                'joint_pos': np.array([0.00583234, 0.0182927, 0.08291]),
            },
            {  # Joint "3": lamparm__base_elbow -> lamparm__elbow_wrist
                'body_pos': np.array([0.105806, -0.0129789, 0.301237]),
                'body_quat': np.array([0.57561409, 0.42365807, -0.56613909, 0.41069306]),
                'joint_axis': np.array([0.0, 0.0, 1.0]),
                'joint_pos': np.array([0.0, 0.0, 0.0]),
            },
            {  # Joint "4": lamparm__elbow_wrist -> lamparm__wrist_head_2
                'body_pos': np.array([0.0, 0.185185, 0.006125]),
                'body_quat': np.array([0.67065852, 0.67065852, -0.22409184, 0.22409184]),
                'joint_axis': np.array([0.0, 0.0, 1.0]),
                'joint_pos': np.array([0.0, 0.0, 0.0]),
            },
            {  # Joint "5": lamparm__wrist_head_2 -> diffuser
                'body_pos': np.array([0.0, 0.0192, -0.04111]),
                'body_quat': np.array([0.5, 0.5, -0.5, 0.5]),
                'joint_axis': np.array([0.0, 0.0, 1.0]),
                'joint_pos': np.array([0.0, 0.0, 0.0]),
            },
        ]
        
        # Precompute: normalize quaternions and joint axes
        self._base_quat = self._base_quat / np.linalg.norm(self._base_quat)
        for link in self._chain:
            link['body_quat'] = link['body_quat'] / np.linalg.norm(link['body_quat'])
            axis_norm = np.linalg.norm(link['joint_axis'])
            if axis_norm > 0:
                link['joint_axis'] = link['joint_axis'] / axis_norm
        
        # Precompute base rotation matrix
        self._base_rot = self._quat_to_rot(self._base_quat)
    
    # =========================================================================
    # Math Helpers
    # =========================================================================
    
    @staticmethod
    def _quat_to_rot(q: np.ndarray) -> np.ndarray:
        """
        Convert MuJoCo quaternion (w, x, y, z) to 3x3 rotation matrix.
        
        Args:
            q: Quaternion [w, x, y, z]
        
        Returns:
            np.ndarray: 3x3 rotation matrix
        """
        w, x, y, z = q
        return np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
            [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
        ])
    
    @staticmethod
    def _axis_angle_to_rot(axis: np.ndarray, angle: float) -> np.ndarray:
        """
        Convert axis-angle to 3x3 rotation matrix using Rodrigues' formula.
        
        Args:
            axis: Unit rotation axis [x, y, z]
            angle: Rotation angle [rad]
        
        Returns:
            np.ndarray: 3x3 rotation matrix
        """
        # Skew-symmetric matrix of axis
        K = np.array([
            [0,       -axis[2],  axis[1]],
            [axis[2],  0,       -axis[0]],
            [-axis[1], axis[0],  0],
        ])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        return R
    
    @staticmethod
    def _make_transform(R: np.ndarray, p: np.ndarray) -> np.ndarray:
        """
        Create 4x4 homogeneous transformation matrix.
        
        Args:
            R: 3x3 rotation matrix
            p: 3D translation vector
        
        Returns:
            np.ndarray: 4x4 transformation matrix
        """
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = p
        return T
    
    def _compute_link_transform(self, link_idx: int, joint_angle: float) -> np.ndarray:
        """
        Compute 4x4 transform from parent body frame to child body frame
        for a single link at a given joint angle.
        
        The transform is: T = T_static * T_joint
        where:
          T_static = [R_body_quat, body_pos; 0, 1]
          T_joint  = [R(axis,θ), (I-R(axis,θ))*joint_pos; 0, 1]
        
        Args:
            link_idx: Index into self._chain (0-4)
            joint_angle: Joint angle in radians
        
        Returns:
            np.ndarray: 4x4 homogeneous transform from parent to child body
        """
        link = self._chain[link_idx]
        
        # Static body transform
        R_body = self._quat_to_rot(link['body_quat'])
        p_body = link['body_pos']
        
        # Joint transform (rotation about axis at joint_pos)
        R_joint = self._axis_angle_to_rot(link['joint_axis'], joint_angle)
        jp = link['joint_pos']
        p_joint = (np.eye(3) - R_joint) @ jp
        
        # Combined: T_static * T_joint
        # [R_body, p_body] * [R_joint, p_joint] = [R_body*R_joint, p_body + R_body*p_joint]
        R_combined = R_body @ R_joint
        p_combined = p_body + R_body @ p_joint
        
        return self._make_transform(R_combined, p_combined)
    
    # =========================================================================
    # Joint State Access (reads from MuJoCo data - equivalent to reading encoders)
    # =========================================================================
    
    def get_joint_angles(self) -> np.ndarray:
        """
        Get current joint angles of the 5 actuated joints.
        (On a real robot, this reads joint encoders.)
        
        Returns:
            np.ndarray: Joint angles [rad] of shape (5,)
        """
        joint_angles = np.array([self.data.qpos[addr] for addr in self.joint_qpos_addrs])
        return joint_angles
    
    def set_joint_angles(self, joint_angles: np.ndarray):
        """
        Set joint angles of the 5 actuated joints in the simulator.
        
        Args:
            joint_angles: Joint angles [rad] of shape (5,)
        """
        for i, addr in enumerate(self.joint_qpos_addrs):
            self.data.qpos[addr] = joint_angles[i]
    
    # =========================================================================
    # Analytical Kinematics (no MuJoCo mj_forward dependency)
    # =========================================================================
    
    def forward_kinematics(self, joint_angles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute forward kinematics given joint angles using the analytical
        kinematic chain derived from robot.xml.
        
        Does NOT call mj_forward or read any ground-truth body transforms.
        
        Args:
            joint_angles: Array of 5 joint angles [rad]
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: 
                - position: End-effector position [x, y, z] in world frame
                - rotation: End-effector rotation matrix (3x3) in world frame
        """
        # Start with base body transform (world -> scs215_v5)
        T = self._make_transform(self._base_rot, self._base_pos)
        
        # Chain through each link/joint
        for i in range(5):
            T_link = self._compute_link_transform(i, joint_angles[i])
            T = T @ T_link
        
        position = T[:3, 3].copy()
        rotation = T[:3, :3].copy()
        return position, rotation
    
    def forward_kinematics_transform(self, joint_angles: np.ndarray) -> np.ndarray:
        """
        Compute forward kinematics and return 4x4 homogeneous transformation matrix.
        Analytical version (no simulator dependency).
        
        Args:
            joint_angles: Array of 5 joint angles [rad]
        
        Returns:
            np.ndarray: 4x4 transformation matrix
        """
        position, rotation = self.forward_kinematics(joint_angles)
        return self._make_transform(rotation, position)
    
    def get_ee_position(self) -> np.ndarray:
        """
        Get current end-effector position in world frame using analytical FK.
        Reads current joint angles and computes FK (no mj_forward).
        
        Returns:
            np.ndarray: Position [x, y, z] of the diffuser body
        """
        joint_angles = self.get_joint_angles()
        pos, _ = self.forward_kinematics(joint_angles)
        return pos
    
    def get_ee_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get current end-effector pose (position and orientation) using analytical FK.
        Reads current joint angles and computes FK (no mj_forward).
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Position [x,y,z] and rotation matrix (3x3)
        """
        joint_angles = self.get_joint_angles()
        return self.forward_kinematics(joint_angles)
    
    def get_ee_transform(self) -> np.ndarray:
        """
        Get current end-effector transformation matrix (4x4) using analytical FK.
        
        Returns:
            np.ndarray: 4x4 transformation matrix
        """
        position, rotation = self.get_ee_pose()
        return self._make_transform(rotation, position)
    
    def check_reachability(self, target_position: np.ndarray) -> bool:
        """
        Rough check if target position is reachable by the robot.
        Uses the base position from robot.xml (no simulator dependency).
        
        Args:
            target_position: [x, y, z] position
        
        Returns:
            bool: True if likely reachable
        """
        # Base position in world frame (from robot.xml constants)
        base_pos = self._base_pos
        
        # Distance from base
        distance = np.linalg.norm(target_position - base_pos)
        
        # Approximate workspace limits (in meters)
        min_reach = 0.10
        max_reach = 0.55
        
        is_reachable = min_reach <= distance <= max_reach
        
        if not is_reachable:
            logger.warning(f"Target at distance {distance:.3f}m from base")
            logger.warning(f"  Expected workspace: [{min_reach}, {max_reach}]m")
        
        return is_reachable
    
    # =========================================================================
    # Ground Truth Functions (using MuJoCo simulator - for testing/validation)
    # =========================================================================
    
    def forward_kinematics_gt(self, joint_angles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Ground truth forward kinematics using MuJoCo's mj_forward.
        Used for validating the analytical forward_kinematics function.
        
        Args:
            joint_angles: Array of 5 joint angles [rad]
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: 
                - position: End-effector position [x, y, z]
                - rotation: End-effector rotation matrix (3x3)
        """
        # Save original state
        original_qpos = self.data.qpos.copy()
        
        # Set joint angles
        self.set_joint_angles(joint_angles)
        
        # Compute forward kinematics via MuJoCo
        mujoco.mj_forward(self.model, self.data)
        
        # Get position and orientation
        position = self.data.xpos[self.ee_body_id].copy()
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, self.data.xquat[self.ee_body_id])
        rotation = rotation.reshape(3, 3)
        
        # Restore original state
        self.data.qpos[:] = original_qpos
        mujoco.mj_forward(self.model, self.data)
        
        return position, rotation
    
    def forward_kinematics_transform_gt(self, joint_angles: np.ndarray) -> np.ndarray:
        """
        Ground truth FK returning 4x4 transform using MuJoCo.
        
        Args:
            joint_angles: Array of 5 joint angles [rad]
        
        Returns:
            np.ndarray: 4x4 transformation matrix
        """
        position, rotation = self.forward_kinematics_gt(joint_angles)
        return self._make_transform(rotation, position)
    
    def get_ee_position_gt(self) -> np.ndarray:
        """
        Ground truth end-effector position using MuJoCo's mj_forward.
        
        Returns:
            np.ndarray: Position [x, y, z] of the diffuser body
        """
        mujoco.mj_forward(self.model, self.data)
        return self.data.xpos[self.ee_body_id].copy()
    
    def get_ee_pose_gt(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Ground truth end-effector pose using MuJoCo's mj_forward.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Position [x,y,z] and rotation matrix (3x3)
        """
        mujoco.mj_forward(self.model, self.data)
        position = self.data.xpos[self.ee_body_id].copy()
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, self.data.xquat[self.ee_body_id])
        rotation = rotation.reshape(3, 3)
        return position, rotation
    
    def get_ee_transform_gt(self) -> np.ndarray:
        """
        Ground truth end-effector 4x4 transform using MuJoCo.
        
        Returns:
            np.ndarray: 4x4 transformation matrix
        """
        position, rotation = self.get_ee_pose_gt()
        return self._make_transform(rotation, position)
    
    def check_reachability_gt(self, target_position: np.ndarray) -> bool:
        """
        Ground truth reachability check using MuJoCo base position.
        
        Args:
            target_position: [x, y, z] position
        
        Returns:
            bool: True if likely reachable
        """
        mujoco.mj_forward(self.model, self.data)
        base_pos = self.data.xpos[self.base_body_id]
        distance = np.linalg.norm(target_position - base_pos)
        
        min_reach = 0.10
        max_reach = 0.55
        
        is_reachable = min_reach <= distance <= max_reach
        
        if not is_reachable:
            logger.warning(f"[GT] Target at distance {distance:.3f}m from base")
            logger.warning(f"  Expected workspace: [{min_reach}, {max_reach}]m")
        
        return is_reachable
    
    # =========================================================================
    # Inverse Kinematics (uses analytical FK internally - no simulator dependency)
    # =========================================================================
    
    def inverse_kinematics(self,
                           target_position: np.ndarray,
                           target_rotation: Optional[np.ndarray] = None,
                           initial_guess: Optional[np.ndarray] = None,
                           max_iterations: int = 500,
                           position_tolerance: float = 1e-4,
                           orientation_tolerance: float = 1e-2,
                           step_size: float = 0.5) -> Tuple[np.ndarray, bool]:
        """
        Solve inverse kinematics using Damped Least Squares (numerical).
        Uses analytical forward_kinematics internally (no simulator dependency).
        
        Args:
            target_position: Desired [x, y, z] position
            target_rotation: Desired rotation matrix (3x3), optional
            initial_guess: Starting joint angles (if None, uses current configuration)
            max_iterations: Maximum optimization iterations
            position_tolerance: Position error tolerance [meters]
            orientation_tolerance: Orientation error tolerance [rad]
            step_size: Step size multiplier for updates (0 < step_size <= 1)
        
        Returns:
            Tuple[np.ndarray, bool]: 
                - joint_angles: Solution joint angles
                - success: True if converged within tolerance
        """
        # Initial guess
        if initial_guess is None:
            q = self.get_joint_angles()
        else:
            q = initial_guess.copy()
        
        # Apply joint limits
        q = self._apply_joint_limits(q)
        
        # Determine if we're doing position-only or full pose IK
        position_only = (target_rotation is None)
        
        position_error_mag = float('inf')
        
        # Optimization loop using Damped Least Squares
        for iteration in range(max_iterations):
            # Current pose from analytical FK
            current_pos, current_rot = self.forward_kinematics(q)
            
            # Position error
            position_error = target_position - current_pos
            position_error_mag = np.linalg.norm(position_error)
            
            # Orientation error (if target rotation specified)
            if not position_only:
                rotation_error_matrix = target_rotation @ current_rot.T
                orientation_error = self._rotation_matrix_to_axis_angle(rotation_error_matrix)
                orientation_error_mag = np.linalg.norm(orientation_error)
                
                error = np.concatenate([position_error, orientation_error])
                
                if position_error_mag < position_tolerance and orientation_error_mag < orientation_tolerance:
                    logger.info(f"IK converged in {iteration} iterations (full pose)")
                    logger.info(f"  Position error: {position_error_mag:.6f} m")
                    logger.info(f"  Orientation error: {orientation_error_mag:.6f} rad")
                    return q, True
            else:
                error = position_error
                
                if position_error_mag < position_tolerance:
                    logger.info(f"IK converged in {iteration} iterations (position only)")
                    logger.info(f"  Position error: {position_error_mag:.6f} m")
                    return q, True
            
            # Compute Jacobian (uses analytical FK internally)
            if position_only:
                jacobian = self._compute_jacobian_position(q)
            else:
                jacobian = self._compute_jacobian_full(q)
            
            # Damped Least Squares (Levenberg-Marquardt)
            damping = 0.01
            J_damped = jacobian.T @ jacobian + damping * np.eye(5)
            
            try:
                delta_q = np.linalg.solve(J_damped, jacobian.T @ error)
            except np.linalg.LinAlgError:
                logger.warning("Singular Jacobian in IK")
                return q, False
            
            # Update joint angles with step size
            q += step_size * delta_q
            
            # Apply joint limits
            q = self._apply_joint_limits(q)
        
        # Did not converge
        logger.warning(f"IK did not converge after {max_iterations} iterations")
        logger.warning(f"  Final position error: {position_error_mag:.6f} m")
        return q, False
    
    # =========================================================================
    # Jacobian Computation (uses analytical FK - no simulator dependency)
    # =========================================================================
    
    def _compute_jacobian_position(self, q: np.ndarray, delta: float = 1e-6) -> np.ndarray:
        """
        Compute position Jacobian matrix using finite differences
        with the analytical FK.
        
        Args:
            q: Joint angles [rad]
            delta: Finite difference step size
        
        Returns:
            np.ndarray: 3x5 Jacobian matrix (position only)
        """
        jacobian = np.zeros((3, 5))
        q0_pos, _ = self.forward_kinematics(q)
        
        for i in range(5):
            q_plus = q.copy()
            q_plus[i] += delta
            q_plus_pos, _ = self.forward_kinematics(q_plus)
            
            jacobian[:, i] = (q_plus_pos - q0_pos) / delta
        
        return jacobian
    
    def _compute_jacobian_full(self, q: np.ndarray, delta: float = 1e-6) -> np.ndarray:
        """
        Compute full Jacobian matrix (position + orientation) using finite
        differences with the analytical FK.
        
        Args:
            q: Joint angles [rad]
            delta: Finite difference step size
        
        Returns:
            np.ndarray: 6x5 Jacobian matrix (3 position + 3 orientation)
        """
        jacobian = np.zeros((6, 5))
        q0_pos, q0_rot = self.forward_kinematics(q)
        
        for i in range(5):
            q_plus = q.copy()
            q_plus[i] += delta
            q_plus_pos, q_plus_rot = self.forward_kinematics(q_plus)
            
            # Position component
            jacobian[0:3, i] = (q_plus_pos - q0_pos) / delta
            
            # Orientation component (axis-angle representation)
            rot_diff = q_plus_rot @ q0_rot.T
            axis_angle = self._rotation_matrix_to_axis_angle(rot_diff)
            jacobian[3:6, i] = axis_angle / delta
        
        return jacobian
    
    # =========================================================================
    # Utility Functions
    # =========================================================================
    
    def _rotation_matrix_to_axis_angle(self, R: np.ndarray) -> np.ndarray:
        """
        Convert rotation matrix to axis-angle representation.
        
        Args:
            R: 3x3 rotation matrix
        
        Returns:
            np.ndarray: Axis-angle vector [rx, ry, rz]
        """
        trace = np.trace(R)
        angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))
        
        if angle < 1e-6:
            return np.zeros(3)
        
        axis = np.array([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1]
        ])
        axis = axis / (2 * np.sin(angle))
        
        return angle * axis
    
    def _apply_joint_limits(self, q: np.ndarray) -> np.ndarray:
        """
        Clamp joint angles to their limits.
        
        Args:
            q: Joint angles [rad]
        
        Returns:
            np.ndarray: Clamped joint angles
        """
        q_limited = q.copy()
        for i in range(5):
            q_limited[i] = np.clip(q[i], self.joint_limits[i, 0], self.joint_limits[i, 1])
        return q_limited
    
    def check_joint_limits(self, q: np.ndarray) -> bool:
        """
        Check if joint angles are within limits.
        
        Args:
            q: Joint angles [rad]
        
        Returns:
            bool: True if all joints within limits
        """
        for i in range(5):
            if q[i] < self.joint_limits[i, 0] or q[i] > self.joint_limits[i, 1]:
                return False
        return True


# =========================================================================
# Standalone Utility Functions
# =========================================================================

def convert_to_dictionary(qpos: np.ndarray) -> Dict[str, float]:
    """
    Convert joint position array to dictionary (angles in degrees).
    
    Args:
        qpos: Joint positions [rad] of shape (5,)
    
    Returns:
        Dict mapping joint names to angles in degrees
    """
    return {
        'joint_1': np.degrees(qpos[0]),
        'joint_2': np.degrees(qpos[1]),
        'joint_3': np.degrees(qpos[2]),
        'joint_4': np.degrees(qpos[3]),
        'joint_5': np.degrees(qpos[4]),
    }


def convert_to_array(position_dict: Dict[str, float]) -> np.ndarray:
    """
    Convert joint position dictionary (angles in degrees) to array.
    
    Args:
        position_dict: Dict mapping joint names to angles in degrees
    
    Returns:
        np.ndarray: Joint positions [rad] of shape (5,)
    """
    return np.array([
        np.radians(position_dict['joint_1']),
        np.radians(position_dict['joint_2']),
        np.radians(position_dict['joint_3']),
        np.radians(position_dict['joint_4']),
        np.radians(position_dict['joint_5']),
    ])


def move_to_pose(m, d, viewer, desired_position, duration):
    start_time = time.time()

    # Get current joint angles
    kinematics = LeLampKinematics(m, d)
    current_q = kinematics.get_joint_angles()
    current_dict = convert_to_dictionary(current_q)
    
    # Convert desired position to array
    target_q = convert_to_array(desired_position)
    
    while True:
        t = time.time() - start_time
        if t > duration:
            break

        # Interpolation factor [0,1] (make sure it doesn't exceed 1)
        alpha = min(t / duration, 1)

        # Interpolate each joint
        intermediate_q = np.zeros(5)
        for i in range(5):
            p0 = current_q[i]
            pf = target_q[i]
            intermediate_q[i] = (1 - alpha) * p0 + alpha * pf

        # Send position commands to actuators
        for i, actuator_id in enumerate(kinematics.actuator_ids):
            d.ctrl[actuator_id] = intermediate_q[i]

        mujoco.mj_step(m, d)
        
        # Pick up changes to the physics state, apply perturbations, update options from GUI.
        viewer.sync()


def move_to_ee_position(model: mujoco.MjModel,
                        data: mujoco.MjData,
                        viewer,
                        target_position: np.ndarray,
                        duration: float = 2.0,
                        use_current_as_initial: bool = True) -> bool:
    """
    Move robot end-effector to desired Cartesian position using IK.
    Uses analytical FK/IK (no simulator ground truth for kinematics computation).
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        viewer: MuJoCo viewer for visualization
        target_position: Desired [x, y, z] position for end-effector
        duration: Time to complete motion [seconds]
        use_current_as_initial: Use current joint config as IK initial guess
    
    Returns:
        bool: Success flag
    """
    kinematics = LeLampKinematics(model, data)
    
    # Check reachability (analytical, no simulator)
    if not kinematics.check_reachability(target_position):
        logger.error(f"Target position {target_position} may be unreachable")
    
    # Solve IK (analytical, no simulator)
    if use_current_as_initial:
        initial_guess = kinematics.get_joint_angles()
    else:
        initial_guess = None
    
    logger.info(f"Solving IK for target position: {target_position}")
    target_q, ik_success = kinematics.inverse_kinematics(
        target_position, 
        initial_guess=initial_guess,
        max_iterations=500,
        position_tolerance=1e-4
    )
    
    if not ik_success:
        logger.warning("IK did not fully converge, but attempting motion anyway")
    
    # Verify IK solution (analytical FK)
    fk_pos, fk_rot = kinematics.forward_kinematics(target_q)
    ik_error = np.linalg.norm(fk_pos - target_position)
    logger.info(f"IK solution error: {ik_error:.6f} m")
    logger.info(f"IK joint solution: {convert_to_dictionary(target_q)}")
    
    # Convert to dictionary and call move_to_pose
    target_dict = convert_to_dictionary(target_q)
    return move_to_pose(model, data, viewer, target_dict, duration)


def hold_position(model: mujoco.MjModel,
                  data: mujoco.MjData,
                  viewer,
                  duration: float) -> None:
    """
    Hold current joint position for a specified duration.
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        viewer: MuJoCo viewer
        duration: Time to hold [seconds]
    """
    kinematics = LeLampKinematics(model, data)
    current_q = kinematics.get_joint_angles()
    
    start_time = time.time()
    while time.time() - start_time < duration:
        # Send current position as control target
        for i, actuator_id in enumerate(kinematics.actuator_ids):
            data.ctrl[actuator_id] = current_q[i]
        
        mujoco.mj_step(model, data)
        
        if viewer is not None:
            viewer.sync()


def track_trajectory(model: mujoco.MjModel,
                     data: mujoco.MjData,
                     viewer,
                     trajectory: List[Tuple[np.ndarray, Optional[np.ndarray]]],
                     duration_per_segment: float = 0.5,
                     position_only: bool = True,
                     ik_position_tolerance: float = 1e-4,
                     ik_orientation_tolerance: float = 1e-2,
                     max_ik_iterations: int = 500,
                     approach_duration: float = 1.0,
                     settle_steps: int = 0) -> Dict:
    """
    Track a trajectory of end-effector poses using analytical IK and
    joint-space interpolation. No simulator ground truth is used for
    kinematics computation.
    
    The function operates in three phases:
      1. **IK Pre-solve**: Solve IK for every waypoint up front, using
         warm-starting (previous solution seeds the next) for continuity
         and fast convergence.
      2. **Approach**: Smoothly move from the current robot configuration
         to the first waypoint (cubic-smoothed to start from rest).
      3. **Execute**: Linearly interpolate joint angles between consecutive
         waypoints at the simulation timestep rate. Linear interpolation is
         used (rather than cubic smoothing) so the end-effector flows
         continuously through waypoints without stopping, which is
         appropriate for dense trajectories.
    
    After execution, actual end-effector positions are measured at each
    waypoint using analytical FK (reading joint encoders), and tracking
    errors are reported.
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        viewer: MuJoCo viewer, or None for headless execution
        trajectory: List of (position, rotation) tuples where:
                    - position: np.ndarray shape (3,) — [x, y, z] in meters
                    - rotation: np.ndarray shape (3,3) rotation matrix, or None
                    If position_only=True the rotation element is ignored (can
                    be None).
        duration_per_segment: Time for each segment between consecutive
                              waypoints [seconds]. Controls execution speed.
        position_only: If True, IK tracks position only (3-DOF, ignoring
                       orientation). Recommended for the 5-DOF lamp arm.
        ik_position_tolerance: IK position convergence tolerance [meters]
        ik_orientation_tolerance: IK orientation convergence tolerance [rad]
        max_ik_iterations: Maximum IK iterations per waypoint
        approach_duration: Time to move from current config to the first
                          waypoint [seconds]. Set to 0 to skip approach.
        settle_steps: Number of extra simulation steps to hold the target
                      position at the end of each segment, allowing the
                      position controller to converge before measurement.
                      0 = no settling (best for continuous/smooth motion).
                      Values of 10-50 help reduce tracking error at the
                      cost of slightly choppier motion.
    
    Returns:
        dict: {
            'success': bool — True if all IK solutions converged,
            'n_waypoints': int,
            'joint_solutions': list of np.ndarray (5,) — IK solutions,
            'ik_converged': list of bool — per-waypoint IK convergence,
            'ik_position_errors': list of float — IK position error [m],
            'ik_max_error': float,
            'ik_mean_error': float,
            'actual_positions': list of np.ndarray (3,) — measured EE
                                positions after reaching each waypoint,
            'target_positions': list of np.ndarray (3,),
            'tracking_errors': list of float — ||actual - target|| [m],
            'tracking_max_error': float,
            'tracking_mean_error': float,
        }
    """
    kinematics = LeLampKinematics(model, data)
    dt = model.opt.timestep
    n_waypoints = len(trajectory)
    
    if n_waypoints < 2:
        logger.error("Trajectory must have at least 2 waypoints")
        return {'success': False, 'n_waypoints': n_waypoints}
    
    # ------------------------------------------------------------------
    # Parse trajectory into separate position / rotation arrays
    # ------------------------------------------------------------------
    target_positions: List[np.ndarray] = []
    target_rotations: List[Optional[np.ndarray]] = []
    
    for wp in trajectory:
        if isinstance(wp, np.ndarray) and wp.ndim == 1:
            # Bare position vector
            target_positions.append(wp.copy())
            target_rotations.append(None)
        else:
            pos, rot = wp
            target_positions.append(np.asarray(pos).copy())
            if position_only or rot is None:
                target_rotations.append(None)
            else:
                target_rotations.append(np.asarray(rot).copy())
    
    # ------------------------------------------------------------------
    # Phase 1: Pre-solve IK for all waypoints (warm-started)
    # ------------------------------------------------------------------
    logger.info(f"Phase 1: Pre-solving IK for {n_waypoints} waypoints "
                f"(position_only={position_only})...")
    
    joint_solutions: List[np.ndarray] = []
    ik_converged_list: List[bool] = []
    ik_position_errors: List[float] = []
    
    q_guess = kinematics.get_joint_angles()
    
    for i in range(n_waypoints):
        q_sol, converged = kinematics.inverse_kinematics(
            target_positions[i],
            target_rotation=target_rotations[i],
            initial_guess=q_guess,
            max_iterations=max_ik_iterations,
            position_tolerance=ik_position_tolerance,
            orientation_tolerance=ik_orientation_tolerance,
            step_size=0.5,
        )
        joint_solutions.append(q_sol)
        ik_converged_list.append(converged)
        
        fk_pos, _ = kinematics.forward_kinematics(q_sol)
        err = np.linalg.norm(fk_pos - target_positions[i])
        ik_position_errors.append(err)
        
        if not converged:
            logger.warning(f"  Waypoint {i}/{n_waypoints}: IK did not converge "
                           f"(error={err:.6f} m)")
        
        # Warm-start: seed next solve with current solution
        q_guess = q_sol.copy()
    
    ik_max_err = max(ik_position_errors)
    ik_mean_err = float(np.mean(ik_position_errors))
    n_converged = sum(ik_converged_list)
    
    logger.info(f"  IK pre-solve done: {n_converged}/{n_waypoints} converged")
    logger.info(f"  IK max position error:  {ik_max_err:.6f} m")
    logger.info(f"  IK mean position error: {ik_mean_err:.6f} m")
    
    # ------------------------------------------------------------------
    # Phase 2: Approach first waypoint (cubic-smoothed from rest)
    # ------------------------------------------------------------------
    if approach_duration > 0:
        logger.info(f"Phase 2: Approaching first waypoint "
                    f"({approach_duration:.2f}s)...")
        steps_approach = max(int(approach_duration / dt), 1)
        q_current = kinematics.get_joint_angles()
        q_first = joint_solutions[0]
        
        for step in range(steps_approach):
            alpha = (step + 1) / steps_approach
            # Cubic smoothing: zero velocity at both endpoints
            alpha_s = alpha * alpha * (3.0 - 2.0 * alpha)
            q_cmd = (1.0 - alpha_s) * q_current + alpha_s * q_first
            
            for j, aid in enumerate(kinematics.actuator_ids):
                data.ctrl[aid] = q_cmd[j]
            mujoco.mj_step(model, data)
            if viewer is not None:
                viewer.sync()
    
    # ------------------------------------------------------------------
    # Phase 3: Execute trajectory segments
    # ------------------------------------------------------------------
    steps_per_seg = max(int(duration_per_segment / dt), 1)
    total_segments = n_waypoints - 1
    
    logger.info(f"Phase 3: Executing {total_segments} segments "
                f"({steps_per_seg} steps each, "
                f"{duration_per_segment:.3f}s/seg)...")
    
    # Record actual EE position at each waypoint boundary
    actual_positions: List[np.ndarray] = []
    
    # Record at waypoint 0 (after approach)
    actual_q = kinematics.get_joint_angles()
    actual_pos, _ = kinematics.forward_kinematics(actual_q)
    actual_positions.append(actual_pos.copy())
    
    for seg in range(total_segments):
        q_start = joint_solutions[seg]
        q_end = joint_solutions[seg + 1]
        
        for step in range(steps_per_seg):
            alpha = (step + 1) / steps_per_seg
            # Linear interpolation — gives continuous flow through waypoints
            # (no stop-and-go), appropriate for dense trajectories
            q_cmd = (1.0 - alpha) * q_start + alpha * q_end
            
            for j, aid in enumerate(kinematics.actuator_ids):
                data.ctrl[aid] = q_cmd[j]
            mujoco.mj_step(model, data)
            if viewer is not None:
                viewer.sync()
        
        # Optional settling: hold target for extra steps
        if settle_steps > 0:
            for _ in range(settle_steps):
                for j, aid in enumerate(kinematics.actuator_ids):
                    data.ctrl[aid] = q_end[j]
                mujoco.mj_step(model, data)
                if viewer is not None:
                    viewer.sync()
        
        # Record actual position after reaching this waypoint
        actual_q = kinematics.get_joint_angles()
        actual_pos, _ = kinematics.forward_kinematics(actual_q)
        actual_positions.append(actual_pos.copy())
    
    # ------------------------------------------------------------------
    # Compute tracking errors
    # ------------------------------------------------------------------
    tracking_errors: List[float] = []
    for i in range(n_waypoints):
        err = np.linalg.norm(actual_positions[i] - target_positions[i])
        tracking_errors.append(err)
    
    tracking_max = max(tracking_errors)
    tracking_mean = float(np.mean(tracking_errors))
    
    logger.info(f"Trajectory tracking complete.")
    logger.info(f"  Tracking max error:  {tracking_max:.6f} m")
    logger.info(f"  Tracking mean error: {tracking_mean:.6f} m")
    
    return {
        'success': all(ik_converged_list),
        'n_waypoints': n_waypoints,
        'joint_solutions': joint_solutions,
        'ik_converged': ik_converged_list,
        'ik_position_errors': ik_position_errors,
        'ik_max_error': ik_max_err,
        'ik_mean_error': ik_mean_err,
        'actual_positions': actual_positions,
        'target_positions': target_positions,
        'tracking_errors': tracking_errors,
        'tracking_max_error': tracking_max,
        'tracking_mean_error': tracking_mean,
    }


if __name__ == "__main__":
    print("LeLamp Utilities Module")
    print("Import this module to use FK, IK, and motion control functions")
    print("Use _gt suffix functions for ground truth validation")
