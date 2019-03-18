from facial_recognition_helper import captureImages
from facial_recognition_helper import extract_align_faces
from facial_recognition_helper import initialize
from facial_recognition_helper import add_user
from facial_recognition_helper import extract_features
from facial_recognition_helper import train_knn
from facial_recognition_helper import recognize_face

#initialize()
#add_user("praveen_seeniraj")
#add_user("anand_palanisamy")
#add_user("koustav_samaddar")
#add_user("hridesh_sainani")
#train_knn()
result = recognize_face()
print(result)