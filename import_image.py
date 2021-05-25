import argparse
import os
import pickle
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import requests
import re

# Remaining issues: Filename display output incorrect
# Remaining issues: File size display output has no unit and may be unnecessary
# Delete token.pickle to use with your google account - request access from joeyhark@gmail.com

ap = argparse.ArgumentParser()
ap.add_argument('-f', '--filename', required=False,
                help='[Str] File name of image to import',
                default='nicolascage.jpg')
ap.add_argument('-u', '--unlock', required=False, action='store_true', default=False,
                help='Passing this argument will unlock restricted files in Google Drive')
args = vars(ap.parse_args())


SCOPES = ['https://www.googleapis.com/auth/drive.metadata',
          'https://www.googleapis.com/auth/drive',
          'https://www.googleapis.com/auth/drive.file'
          ]


def declare_service():
    creds = None

    # token.pickle stores access and refresh tokens; created automatically on first iteration
    # of authorization flow
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)

    # Prompt user login if no valid credentials exist
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)

        # Save credentials
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    # Return Google Drive API resource
    return build('drive', 'v3', credentials=creds)


def search(service, query):
    result = []
    page_token = None

    # File search
    while True:
        response = service.files().list(q=query,
                                        spaces='drive',
                                        fields='nextPageToken, files(id, name, mimeType)',
                                        pageToken=page_token).execute()

        # Iterate over files
        for file in response.get('files', []):
            result.append((file['id'], file['name'], file['mimeType']))

        page_token = response.get('nextPageToken', None)
        if not page_token:
            # No more files
            break

    return result


def download(Filename):
    service = declare_service()
    filename = Filename

    # Search by name
    search_result = search(service, query=f"name='{filename}'")

    # Extract Google Drive file ID
    file_id = search_result[0][0]

    # Change file permissions to shareable
    if args['unlock']:
        service.permissions().create(body={'role': 'reader', 'type': 'anyone'}, fileId=file_id).execute()

    pull_file_GDrive(file_id, filename)


def pull_file_GDrive(id, destination):

    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    def save_response_content(response, destination):

        # Extract file size
        file_size = int(response.headers.get('Content-Length', 0))

        # Extract content disposition
        content_disposition = response.headers.get('content-disposition')

        # Parse file name
        filename = re.findall('filename=\"(.+)\"', content_disposition)[0]

        # Print helpful info
        print('[INFO] File size: ', file_size)
        print('[INFO] File Name:  ', filename)

        with open(destination, 'wb') as f:
            f.write(response.content)

    # Base URL
    URL = 'https://docs.google.com/uc?export=download'

    # Initialize HTTP session
    session = requests.Session()

    # Make request
    response = session.get(URL, params={'id': id}, stream=True)
    print('[INFO] Downloading ', response.url)

    # Extract confirmation token
    token = get_confirm_token(response)
    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    # Download to destination
    save_response_content(response, destination)


if __name__ == '__main__':
    download(args['filename'])
