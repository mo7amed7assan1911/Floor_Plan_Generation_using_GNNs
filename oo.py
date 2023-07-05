import imgurpython

# Create an Imgur client instance with your API credentials
client = imgurpython.ImgurClient("YOUR_CLIENT_ID", "YOUR_CLIENT_SECRET")

# Upload the image and get the link
image_path = "path/to/your/image.png"
response = client.upload_from_path(image_path, anon=True)
image_link = response["link"]

# Print the image link
print("Image link:", image_link)

