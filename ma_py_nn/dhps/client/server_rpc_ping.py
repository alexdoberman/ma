# -*- coding: utf-8 -*-

import xmlrpc.client

#from dhps.client.client_config import LST_SERVER_ADR
from client_config import LST_SERVER_ADR

def main():

    for server_adr in LST_SERVER_ADR:
        try:
            uri = "http://" + server_adr['host'] + ":" +(str)(server_adr['port']) + "/"
            proxy = xmlrpc.client.ServerProxy(uri)
            is_busy = proxy.is_busy()
            print ("Ping server {}, is busy = {}".format(uri, is_busy))
        except Exception as e:
            print ("Can`t ping addr: {}, e = {}".format(uri, e))
    print ("Exit...")


if __name__ == "__main__":
    main()





