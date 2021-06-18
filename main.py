import cv2
import numpy as np

from PIL import Image
from facenet_pytorch import MTCNN
from matplotlib import pyplot as plt
import matplotlib.patches as patches


# loads PIL image into memory
def load_image(image_path):
    img = cv2.imread(image_path)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# draws the boxes around the faces and saves the figure
def save_image(img, boxes, probs, output_file, add_probs=False):
    px = 1/plt.rcParams['figure.dpi']  # pixel in inches
    fig, ax = plt.subplots(figsize=(img.size[0]*px, img.size[1]*px))
    ax.imshow(img)
    ax.axis("off")

    for box, prob in zip(boxes, probs):
        ax.add_patch(patches.Rectangle(
                     (box[0], box[1]),
                     box[2]-box[0],
                     box[3]-box[1],
                     color="red",
                     linewidth=2,
                     fill=False)) 
        if add_probs:
            ax.text(box[0], box[1], f"{100*prob:.2f}%", color="red", fontsize=10)
    plt.show()
    fig.savefig(output_file, bbox_inches='tight', pad_inches=0)


def main():

    # load img into memory
    input_img_path = "renoir"
    input_img = load_image(input_img_path + ".jpg")

    # initialize MTCNN and detect faces
    mtcnn = MTCNN(keep_all=True, device="cpu", thresholds=[0.55, 0.6, 0.7])
    boxes, probs = mtcnn.detect(input_img)
    # save detection result    
    save_image(input_img, boxes, probs, input_img_path+"_faces.jpg")



if __name__ == "__main__":
    main()