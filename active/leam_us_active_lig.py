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

def main(args):
    # Stress Test and Get Time Stamp
    with open("secrets/secrets.json", "r") as s:
         api_token = json.load(s)["api_token"]
    
    API_URL = "https://gleam-seir-api-883627921778.us-west1.run.app/download-folder"
    FOLDER_NAME = "outputdata/data1739842798/data"  # The folder you want to download based on timestamp outputted from compute run
    OUTPUT_ZIP_PATH = "downloaded_folder.zip"  # Where to save the ZIP file
    API_KEY = api_token


    headers = {
        "X-API-Key": API_KEY
    }

    response = requests.get(f"{API_URL}?folder_name={FOLDER_NAME}", headers=headers, stream=True)

    if response.status_code == 200:
        download_url = response.json()["download_url"]
        print(f"Download URL: {download_url}")
        zip_response = requests.get(download_url, stream=True)
        if zip_response.status_code == 200:
            with open("downloaded_folder.zip", "wb") as f:
                for chunk in zip_response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Downloaded successfully: downloaded_folder.zip")
        else:
            print(f"Error downloading ZIP: {zip_response.status_code}, {zip_response.text}")
            return

    else:
        print(f"Error: {response.status_code}, {response.text}")
        return

    ZIP_FILE_PATH = "downloaded_folder.zip"  # Path to the downloaded ZIP file
    EXTRACT_TO = "data/data"  # Directory where files will be extracted

    os.makedirs(EXTRACT_TO, exist_ok=True)


    with zipfile.ZipFile(ZIP_FILE_PATH, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_TO)

    print(f"Files extracted to: {EXTRACT_TO}")

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

