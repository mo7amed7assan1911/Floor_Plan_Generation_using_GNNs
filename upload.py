import pyrebase
import os

config = {
  "apiKey": "AIzaSyB2C4jSNmWRzTUsI5GoG4kwPN9OyZ4hWXs",
  "authDomain": "planify-images.firebaseapp.com",
  "projectId": "planify-images",
  "databaseURL": "https://planify-images.firebaseapp.com",
  "storageBucket": "planify-images.appspot.com",
  "messagingSenderId": "894888223160",
  "appId": "1:894888223160:web:d9d2e9b24f105d029d6f4d",
  "measurementId": "G-F1N3RZ3KT4"
};

firebase = pyrebase.initialize_app(config)
storage = firebase.storage()



def upload_to_firebase(unique_name):
    loacal_path = os.path.abspath("outputs/" + unique_name)
    remote_path = "images/"+unique_name
    storage.child(remote_path).put(loacal_path)
    
    image_url = storage.child(remote_path).get_url(None)
    
    return image_url




