import os
import requests

def download_medquad_xml_files():
    # Define the base URL for the GitHub API to list files in the folder
    api_url = "https://api.github.com/repos/abachaa/MedQuAD/contents/5_NIDDK_QA"

    # Create the directory to save the XML files if it doesn't exist
    directory = "data/5_NIDDK_QA"
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Get the list of files in the folder using GitHub API
    response = requests.get(api_url)
    
    # Check if the request was successful
    if response.status_code == 200:
        files = response.json()  # This will be a list of files in the folder
        for file in files:
            file_name = file['name']
            file_url = file['download_url']  # Get the raw file URL
            
            # Send a GET request to download the file
            file_response = requests.get(file_url)
            
            # Check if the file was downloaded successfully
            if file_response.status_code == 200:
                # Save the content to a local file in the specified directory
                with open(os.path.join(directory, file_name), "wb") as f:
                    f.write(file_response.content)
                print(f"Downloaded: {file_name}")
            else:
                print(f"Failed to download: {file_name}")
    else:
        print(f"Failed to get file list. Status code: {response.status_code}")

# Call the function to download all the XML files
download_medquad_xml_files()
