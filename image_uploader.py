import requests
import base64

file_path = "/Users/maao/Desktop/地市上报/东营公司实体卡照片/898602D5151870186598.jpg"
with open(file_path, mode="rb") as image_file:
    encoded_string = base64.b64encode(image_file.read())

r = requests.post("http://localhost:5000/recognize",
                  data={'image_b64': encoded_string})
print(r.status_code, r.reason)
print("Get serial from remote server: " + r.text)

r = requests.post("http://localhost:5000/collect",
                  data={'image_b64': encoded_string,
                        'ICCID': file_path[file_path.rfind('/') + 1: file_path.rfind('.')]
                        })
