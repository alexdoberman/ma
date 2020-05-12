# -*- coding: utf-8 -*-

import xmlrpc.client
import os
import time
import copy

from dhps.client.client_config import LST_SERVER_ADR
from dhps.client.tasks_config import LST_TASKS, TASK_CONFIG_PATH



def get_avaliable_servers():
    """

    :return:
    """
    # Get avail servers
    avaliable_servers = []
    for server_adr in LST_SERVER_ADR:
        try:
            uri = "http://" + server_adr['host'] + ":" + (str)(server_adr['port']) + "/"
            proxy = xmlrpc.client.ServerProxy(uri)
            busy = proxy.is_busy()
            if not busy:
                avaliable_servers.append(server_adr)
        except Exception:
            print("Can`t ping addr: {}".format(uri))
    return avaliable_servers

def get_str_task_descr(task_id):
    task = LST_TASKS[task_id]
    return '< script_name = {}  config_path = {} dataset_name = {} >'.format(task['script_name'], task['config_path'], task['dataset_name'])

def get_str_server_descr(server):
    return "http://" + server['host'] + ":" + (str)(server['port']) + "/"


def main():

    lst_task = copy.deepcopy(LST_TASKS)

    map_task_on_server = {}
    map_task_result    = {}
    while True:

        avaliable_servers = get_avaliable_servers()

        done_tasks_id = []
        # Check task is done
        print("Check running task status ...")
        for task_id, server_adr in map_task_on_server.items():
            uri = "http://" + server_adr['host'] + ":" + (str)(server_adr['port']) + "/"
            proxy = xmlrpc.client.ServerProxy(uri)
            if (not proxy.is_busy()):
                map_task_result[task_id] =  proxy.get_result()
                done_tasks_id.append(task_id)
                print('    task = {} | server = {} | status = {}'.format(get_str_task_descr(task_id),
                                                                         get_str_server_descr(server_adr), 'done'))

                with open("all_results.txt", "a") as res_file:
                    res_file.write('task = {} | result = {} \n'.format(get_str_task_descr(task_id), map_task_result[task_id]))

            else:
                print('    task = {} | server = {} | status = {}'.format(get_str_task_descr(task_id),
                                                                         get_str_server_descr(server_adr), 'training'))
        # Remove done tasks
        for k in done_tasks_id:
            del map_task_on_server[k]

        if (avaliable_servers):
            print ('Start new tasks ...')

        # Try start new tasks
        for server_adr in avaliable_servers:

            if not lst_task:
                break

            uri = "http://" + server_adr['host'] + ":" + (str)(server_adr['port']) + "/"
            proxy = xmlrpc.client.ServerProxy(uri)
            if proxy.is_busy():
                continue

            current_task = lst_task.pop()
            current_task_id = LST_TASKS.index(current_task)

            config_path = os.path.join(TASK_CONFIG_PATH, current_task['config_path'])
            with open(config_path, "r") as myfile:
                config_string = myfile.read()

            proxy.start_train(current_task['script_name'], config_string, current_task['config_path'], current_task['dataset_name'])
            map_task_on_server[current_task_id] = server_adr

            print('    start task: {}  on  server: {}'.format(get_str_task_descr(current_task_id),
                                                                     get_str_server_descr(server_adr)))

        if (not lst_task) and len(map_task_on_server) == 0:
            # Done
            print('-------------------------------------------------------')
            print('-------------------------------------------------------')

            for task_id, result in map_task_result.items():
                print('Task: {}  Result: {}'.format(get_str_task_descr(task_id), result))
            return 0

        time.sleep(5)


if __name__ == "__main__":
    main()






