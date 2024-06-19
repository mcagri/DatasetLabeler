import configparser

config = configparser.ConfigParser()
config.read("../Config/db.config")
username = config['MONGODB']['MONGO_USER']
pwd = config['MONGODB']['MONGO_PASSWORD']
dbname = config['MONGODB']['MONGO_DB']
dbcollection = config['MONGODB']['MONGO_COLLECTION']
url = config['MONGODB']['MONGO_URL']
port = int(config['MONGODB']['MONGO_PORT'])

path = config['PREFECT']['PREFECT_PATH']
mono_path = config['PREFECT']['PREFECT_MONO_PATH']
resample_path = config['PREFECT']['PREFECT_RESAMPLE_PATH']
chunk_path = config['PREFECT']['PREFECT_CHUNK_PATH']
sample_rate = int(config['PREFECT']['PREFECT_SAMPLE_RATE'])
buffer_size = int(config['PREFECT']['PREFECT_BUFFER_SIZE'])

dataset_path = config['WHISPER']['WHISPER_DATASET_PATH']