import requests

# Send request to your deployed API
response = requests.post(
    "https://realism-voice.onrender.com/tts",
    data="Hey dawg, this is AB Uncle checking in with you today!",
)

# Check if we got audio back
if response.status_code == 200:
    # Save the audio to a file
    with open("test_response.mp3", "wb") as f:
        f.write(response.content)
    print("Successfully saved audio to test_response.mp3")
else:
    print(f"Error: {response.status_code}")
    print(response.text) 