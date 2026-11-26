def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        required_columns = ['PT08.S1(CO)_mean', 'PT08.S2(NMHC)_mean', 'PT08.S3(NOx)_mean', 'PT08.S4(NO2)_mean', 'PT08.S5(O3)_mean', 'NOx(GT)_mean', 'C6H6(GT)_mean', 'T_mean', 'RH_mean', 'AH_mean']
        if all(col in data.columns for col in required_columns):
            return data
        else:
            print(f"Error: Required columns not found in data.")
            return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
