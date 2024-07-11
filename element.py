import cv2
import utils 

class Element:
    def __init__(self, new_img_file):
        self.img_file = new_img_file
        self.img = self._load_img()
        self.contours = self._find_contours()

    def _find_contours(self):
        self.contours = utils.get_contours(self.img, mode=cv2.RETR_TREE)

    def _load_img(self):
        self.img = cv2.imread(self.img_file)


class Template(Element):
    def __init__(self, new_img_file):
        Element.__init__(self, new_img_file)
        self.traits = []
    #Need template trait
    def add_trait(self, new_trait):
        self.traits.append(new_trait)

class TemplateTrait:
    def __init__(self, new_position, new_text=None, new_element=None):
        self.position = new_position
        self.text = new_text
        self.element = new_element