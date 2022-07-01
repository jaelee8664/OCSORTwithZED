import yolox
import os
yolox_path = os.path.dirname(os.path.dirname(yolox.__file__))
print(yolox.__file__)
print(os.path.dirname(yolox.__file__))
print(yolox_path)
print(os.path.split(os.path.realpath(__file__))[1].split(".")[0])
print(os.path.split(os.path.realpath(__file__)))

