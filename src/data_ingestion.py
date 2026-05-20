import pandas as pd
import os
from sklearn.model_selection import train_test_split

import logging

log_dir = "logs"

os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger("data_ingestion")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")


file_path = os.path.join(log_dir, "data_ingestion.log")
file_handler = logging.FileHandler(file_path)
file_handler.setLevel("DEBUG")


formatter = logging.Formatter(" %(asctime)s --- %(name)s --- %(levelname)s --- %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)



def load_data(data_url : str) -> pd.DataFrame:
    
    try:
        df = pd.read_csv(data_url)
        logger.debug(f"Data Loaded From %s", data_url)
        return df
    except pd.errors.ParserError as e:
        logger.error("Failed to Parse the CSV file %s", e)
    except Exception as e:
        logger.error("Unexpected Error While Loading the File : %s ", e)
        raise


def preProcess_data(df : pd.DataFrame) -> pd.DataFrame:

    try:
        df.drop(columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], inplace=True)
        df.rename(columns={"v1": "target","v2" : "text"}, inplace=True)
        logger.debug("Data PreProcessed Completed")
        return df
    
    except pd.errors.ParserError as e:
        logger.error("Failed to PreProcess Data %s", e)
    except Exception as e:
        logger.error("Unexpected Error While PreProcess Data : %s ", e)
        raise




def main():
    try:
        data_path = 'https://raw.githubusercontent.com/vikashishere/Datasets/main/spam.csv'
        df = load_data(data_url=data_path)
        print(df.head())

        df = preProcess_data(df)
        print(df)
      
    except Exception as e:
        logger.error('Failed to complete the data ingestion process: %s', e)
        print(f"Error: {e}")
        
if __name__ == '__main__':
    main()


    


