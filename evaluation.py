import PIL.Image
import PIL.ImageChops

MAX = 255 * 200 * 200

def evaluate(solution, target_image):
    image = draw(solution)
    diff = PIL.ImageChops.difference(image, target_image)
    hist = diff.convert("L").histogram()
    count = sum(i * n for i, n in enumerate(hist))
    return (MAX - count) / MAX
