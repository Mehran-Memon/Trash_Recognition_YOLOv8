from roboflow import Roboflow
rf = Roboflow(api_key="T0wFToAaOYWZDBERldKM")
project = rf.workspace("federico-andrea-rizzi-g1fjs").project("dataset-itis-cardano-trash-detection")
version = project.version(1)
dataset = version.download("yolov8")
                