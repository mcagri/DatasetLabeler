from prefect import flow, task, serve
from pymongo import MongoClient
import configparser
import pandas as pd
import requests
import json

config = configparser.ConfigParser()
config.read("../Config/db.config")
username = config['MONGODB']['MONGO_USER']
pwd = config['MONGODB']['MONGO_PASSWORD']
dbname = config['MONGODB']['MONGO_DB']
url = config['MONGODB']['MONGO_URL']
port = config['MONGODB']['MONGO_PORT']
client = MongoClient(url,
                         username=username,
                         password=pwd,
                         authSource="admin",
                         authMechanism='SCRAM-SHA-256')
db = client[dbname]
collection = db['Audio']