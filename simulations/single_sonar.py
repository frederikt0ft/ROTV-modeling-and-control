import pandas as pd
import multiprocessing as mp
import time

# define a function to continuously modify the dataframe
def modify_df(df_dict):
    while True:
        # retrieve the dataframe from the shared dictionary
        df = df_dict['df']

        # modify the dataframe
        df['column2'] = [100, 200, 300, 400, 500]

        # store the modified dataframe back in the shared dictionary
        df_dict['df'] = df

        # wait for a short time
        time.sleep(1)

# define a function to continuously read the dataframe
def read_df(df_dict):
    while True:
        # retrieve the dataframe from the shared dictionary
        df = df_dict['df']

        # print the dataframe
        print(df)

        # wait for a short time
        time.sleep(1)

if __name__ == '__main__':
    # create a Pandas dataframe
    df = pd.DataFrame({'column1': [1, 2, 3, 4, 5], 'column2': [10, 20, 30, 40, 50]})

    # create a shared dictionary to hold the dataframe reference
    manager = mp.Manager()
    df_dict = manager.dict({'df': df})

    # create two processes
    p1 = mp.Process(target=modify_df, args=(df_dict,))
    p2 = mp.Process(target=read_df, args=(df_dict,))

    # start the processes
    p1.start()
    p2.start()

    # wait for the processes to finish
    p1.join()
    p2.join()