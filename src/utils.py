import pandas as pd


def combine_dataframes(df1, df2, ratio, total_num_data_points):
    # Calculate the number of data points for each DataFrame based on the ratio
    num_data_points_df1 = int(total_num_data_points * ratio )
    num_data_points_df2 = total_num_data_points - num_data_points_df1

    if num_data_points_df1 > len(df1):
        raise ValueError("Requested more data points from df1 than available rows.")
    if num_data_points_df2 > len(df2):
        raise ValueError("Requested more data points from df2 than available rows.")

    sampled_df1 = df1.sample(n=num_data_points_df1, random_state=1)
    sampled_df2 = df2.sample(n=num_data_points_df2, random_state=1)

    combined_df = pd.concat([sampled_df1, sampled_df2]).reset_index(drop=True)

    # Return the combined DataFrame
    return combined_df
