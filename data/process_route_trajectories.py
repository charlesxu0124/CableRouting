import numpy as np
import glob
from scipy.spatial.transform import Rotation as R
import os
from tqdm import tqdm
import argparse

def rotate_action_frame(data):
    states = data['robot_state']
    actions = data['action']

    for i in tqdm(range(states.shape[0])):
        state = states[i]
        action = actions[i]

        #Rotation matrix
        r = R.from_quat(state[3:7])
        r = r.as_matrix()

        #Translation vector
        p = state[:3]

        #Translation hat
        p_hat = np.array([[0, -p[2], p[1]],
        [p[2], 0, -p[0]],
        [-p[1], p[0], 0]])

        #Adjoint
        adjoint = np.zeros((6,6))
        adjoint[:3, :3] = r
        adjoint[3:, :3] = p_hat@r
        adjoint[3:, 3:] = r

        #Velocity in base frame
        V_s = np.zeros((6))
        V_s[:3] = action[:3]
        V_s[5] = action[3]

        #Velocity in tool frame
        V_b = adjoint @ V_s

        #Update action
        action[:3] = V_b[:3]
        action[3] = V_b[5]

        actions[i] = action

    actions = np.clip(actions, -1, 1)
    data['action'] = actions
    return data


def process_trajectories_into_transitions(trajectories):
    actions = []
    robot_state = []
    side_image = []
    top_image = []
    wrist45_image = []
    wrist225_image = []


    output = dict()

    for traj in trajectories:
        traj_npy = np.load(os.path.join(traj, 'traj.npy'), allow_pickle=True).item()

        

        actions += traj_npy['actions']
        robot_state += traj_npy['observations/tcp_pose']
        side_image += traj_npy['observations/side']
        top_image += traj_npy['observations/top']
        wrist45_image += traj_npy['observations/wrist45']
        wrist225_image += traj_npy['observations/wrist225']

    output['action'] = np.array(actions)
    output['robot_state'] = np.array(robot_state)
    output['side_image'] = np.array(side_image)
    output['top_image'] = np.array(top_image)
    output['wrist45_image'] = np.array(wrist45_image)
    output['wrist225_image'] = np.array(wrist225_image)

    output = rotate_action_frame(output)

    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trajectory_path', type=str, help='Path to the route trajectories directory')
    parser.add_argument('--output_path', type=str, help='Path to the output directory')
    args = parser.parse_args()


    PATH = args.trajectory_path
    OUTPUT_PATH = args.output_path
    trajectories = glob.glob(os.path.join(PATH, "*"))
    processed = process_trajectories_into_transitions(trajectories)

    np.save(os.path.join(OUTPUT_PATH, 'route_transitions.npy'), processed)

