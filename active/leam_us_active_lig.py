from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import yaml
from lib import utils
from lib.utils import load_graph_data
from model.pytorch.dcrnn_supervisor import DCRNNSupervisor
import random
import numpy as np
import os
import json
import requests
import zipfile
import time

def main(args):
    # Get API key
    with open("secrets/secrets.json", "r") as s:
         api_token = json.load(s)["token"]
    
    # Stress test and get timestamp
    st_url = "https://gleam-seir-api-883627921778.us-west1.run.app/create_dummy_compute"
    st_headers = {
        "X-API-Key": api_token,
        "Content-Type": "application/json"
    }
    st_payload = {
        "cpu": 1,
        "io": 1,
        "vm": 1,
        "vm_bytes": "1G",
        "timeout": "1M"
    }
    st_response = requests.post(st_url, json=st_payload, headers=st_headers)
    if st_response.status_code != 200:
         print(f"Error with stress test API: {st_response.status_code}, {st_response.text}")
         return
    
    current_timestamp = st_response.json()

    # Get data
    data_url = "https://gleam-seir-api-883627921778.us-west1.run.app/download-folder"
    cloud_folder = f"outputdata/data{current_timestamp}"
    local_zip_path = "downloaded_folder.zip"

    data_headers = {
        "X-API-Key": api_token
    }
    data_url = f"{data_url}?folder_name={cloud_folder}"
    data_response = requests.get(data_url, headers=data_headers, stream=True)
    if data_response.status_code != 200:
         print(f"Error with data API: {data_response.status_code}, {data_response.text}")
         return
    
    # Get zipfile
    download_url = data_response.json()["download_url"]
    zip_response = None
    retries = 3
    for _ in range(retries):
         time.sleep(60)
         zip_response = requests.get(download_url, stream=True)
         if zip_response.status_code != 200:
              print("Retrying Zip Pull")
              continue
         
         break
    
    if zip_response.status_code != 200:
         print(f"Error pulling zip file: {zip_response.status_code}, {zip_response.text}")
         return
    
    with open(local_zip_path, "wb") as f:
         for chunk in zip_response.iter_content(chunk_size=8192):
              f.write(chunk)

    # Extract zipfile
    extract_to_path = "data/data"
    os.makedirs(extract_to_path, exist_ok=True)

    with zipfile.ZipFile(local_zip_path, "r") as zip_ref:
        for file in zip_ref.namelist():
            if file.endswith(".npy") or file.endswith(".npz"):
                file_name = file.split("/")[-1]
                dest_path = os.path.join(extract_to_path, file_name)

                with zip_ref.open(file) as source, open(dest_path, "wb") as dest:
                    dest.write(source.read())

    print("Successfully extracted the data")

    with open(args.config_filename) as f:
        supervisor_config = yaml.safe_load(f)

        graph_pkl_filename = supervisor_config['data'].get('graph_pkl_filename')
        sensor_ids, sensor_id_to_ind, adj_mx = load_graph_data(graph_pkl_filename)
        
        i=2
        np.random.seed(i)
        random.seed(i)
        max_itr = 12 #12
        data, search_data_x, search_data_y = utils.load_dataset(**supervisor_config.get('data'))
        supervisor = DCRNNSupervisor(random_seed=i, iteration=0, max_itr = max_itr, 
                adj_mx=adj_mx, **supervisor_config)

        if not os.path.exists('seed%d/reward_list' % (i)):
                os.makedirs('seed%d/reward_list' % (i))
        if not os.path.exists('seed%d/index_list' % (i)):
                os.makedirs('seed%d/index_list' % (i))

        for itr in range(max_itr):
            supervisor.iteration = itr
            supervisor._data = data
            supervisor.train()

            reward_list = []
            index_list = []
            for k in range(int(len(search_data_x))):
                index = np.random.choice(len(search_data_x), 8, replace=False).tolist()
                index_list.append(index)
                search_data_x_all = np.concatenate([search_data_x[i] for i in index],0)
                search_data_y_all = np.concatenate([search_data_y[i] for i in index],0)
                reward = supervisor.acquisition(search_data_x_all, search_data_y_all)
                reward_list.append(reward.item())

            np.save('seed%d/reward_list/itr%d.npy' % (i, itr+1),np.array(reward_list))
            np.save('seed%d/index_list/itr%d.npy' % (i, itr+1),np.stack(index_list))

            # print('reward_list:',reward_list)
            selected_ind = np.argmax(np.array(reward_list))
            selected_data_x = [search_data_x[i] for i in index_list[selected_ind]]
            selected_data_y = [search_data_y[i] for i in index_list[selected_ind]]

            selected_data = {}
            selected_data['x'] = selected_data_x
            selected_data['y'] = selected_data_y
            search_config = supervisor_config.get('data').copy()
            search_config['selected_data'] = selected_data
            search_config['previous_data'] = data

            data = utils.generate_new_trainset(**search_config)

            # search_data_x = [search_data_x[i] for i in sort_ind[:-16]]
            # search_data_y = [search_data_y[i] for i in sort_ind[:-16]]
            search_data_x = [e for i, e in enumerate(search_data_x) if i not in index_list[selected_ind]]
            search_data_y = [e for i, e in enumerate(search_data_y) if i not in index_list[selected_ind]]
            print('remained scenarios:', len(search_data_x))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', default='data/model/dcrnn_cov.yaml', type=str,
                        help='Configuration filename for restoring the model.')
    parser.add_argument('--use_cpu_only', default=False, type=bool, help='Set to true to only use cpu.')
    args = parser.parse_args()
    main(args)

