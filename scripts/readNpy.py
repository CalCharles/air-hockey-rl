import numpy as np

def read_npy_file(file_path):
    try:
        # Load the .npy file
        data = np.load(file_path)
        
        # Display the contents of the file
        print("Data loaded from the .npy file:")
        print(data)
        
        # Optionally, you can return the data if you want to use it elsewhere in your program
        return data

    except Exception as e:
        print(f"An error occurred while reading the file: {e}")

# Example usage
file_path = "gat_log/AirHockey/monitor/training_losses.npy"  # Replace with your .npy file path
read_npy_file(file_path)
