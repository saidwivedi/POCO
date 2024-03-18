# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import os
import json
import shutil
import subprocess
import numpy as np
import os.path as osp


def run_openpose(
        video_file,
        output_folder,
        staf_folder,
        vis=False,
):
    pwd = os.getcwd()

    os.chdir(staf_folder)

    render = 1 if vis else 0
    display = 2 if vis else 0
    cmd = [
        'build/examples/openpose/openpose.bin',
        '--model_pose', 'BODY_25',
        # '--tracking', '1',
        '--render_pose', str(render),
        '--video', video_file,
        '--write_json', output_folder,
        '--display', str(display)
    ]

    print('Executing', ' '.join(cmd))
    subprocess.call(cmd)
    os.chdir(pwd)

def run_singularity_openpose(
        staf_path,
        video_file,
        input_dir,
        output_dir,
        vis=False,
):
    render = 1 if vis else 0

    storage = staf_path.split('/')[1]

    cmd = [
        'singularity', 'exec', '--nv', '-B', f'/{storage}:/{storage}',
        f'{staf_path}/openpose.simg',
        'python3',
        f'{staf_path}/scripts/openpose_script.py',
        '--video_file', video_file,
        '--input_dir', input_dir,
        '--output_dir', output_dir,
        '--render_pose', str(render),
        '--hand_keypoints', '0',
        '--face_keypoints', '0',
    ]

    print('Executing', ' '.join(cmd))
    subprocess.call(cmd)



def read_posetrack_keypoints(output_folder):

    people = dict()

    for idx, result_file in enumerate(sorted(os.listdir(output_folder))):
        json_file = osp.join(output_folder, result_file)
        data = json.load(open(json_file))
        # print(idx, data)
        for person in data['people']:
            person_id = person['person_id'][0]
            joints2d  = person['pose_keypoints_2d']
            if person_id in people.keys():
                people[person_id]['joints2d'].append(joints2d)
                people[person_id]['frames'].append(idx)
            else:
                people[person_id] = {
                    'joints2d': [],
                    'frames': [],
                }
                people[person_id]['joints2d'].append(joints2d)
                people[person_id]['frames'].append(idx)

    for k in people.keys():
        people[k]['joints2d'] = np.array(people[k]['joints2d']).reshape((len(people[k]['joints2d']), -1, 3))
        people[k]['frames'] = np.array(people[k]['frames'])

    return people

def read_singularity_keypoints(output_folder):

    frame_kps = []

    for idx, result_file in enumerate(sorted(os.listdir(output_folder))):
        json_file = osp.join(output_folder, result_file)
        data = json.load(open(json_file))
        fr_kps = dict()
        for person_id, person in enumerate(data['people']):
            joints2d  = person['pose_keypoints_2d']
            fr_kps.update({
                str(person_id): np.array(joints2d).astype(np.float32).reshape(-1, 3)
            })
        frame_kps.append(fr_kps)

    return frame_kps

def read_vitpose_keypoints(root_dir, fname):
    import joblib as jl
    frame_kps = []
    fname = '/'.join(fname[:-4].split('-'))
    datafile = f'{root_dir}/{fname}.data.pyd'
    try:
        data = jl.load(datafile)
    except:
        print(f'No file found - {datafile}')
        return None
    for i in range(len(data)):
        frame_kps.append(data[i]['keypoints_2d'][:25,:])
    return frame_kps

def run_posetracker(video_file, staf_folder, posetrack_output_folder='/tmp', display=False):
    posetrack_output_folder = os.path.join(
        posetrack_output_folder,
        f'{os.path.basename(video_file)}_posetrack'
    )

    # run posetrack on video
    run_openpose(
        video_file,
        posetrack_output_folder,
        vis=display,
        staf_folder=staf_folder
    )

    people_dict = read_posetrack_keypoints(posetrack_output_folder)

    shutil.rmtree(posetrack_output_folder)

    return people_dict

def run_keypoint_detection(video_file, tracking_results, output_dir='/tmp', display=False):

    output_dir = os.path.join(output_dir, f'{os.path.basename(video_file)[:-4]}_openpose_simg_dump')

    run_singularity_openpose(
        video_file,
        '',
        output_dir,
        display
    )

    frame_kps = read_singularity_keypoints(output_dir)

    for person_id in list(tracking_results.keys()):
        tracking_results[person_id]['frame_kps'] = frame_kps

    shutil.rmtree(output_dir)

    return tracking_results


