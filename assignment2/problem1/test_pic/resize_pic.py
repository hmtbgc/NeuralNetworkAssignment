from PIL import Image

name = ["bird.jpg", "cat.jpg", "dog.jpg"]
resize_name = ["bird_re.jpg", "cat_re.jpg", "dog_re.jpg"]

for n, re_n in zip(name, resize_name):
    img = Image.open(n)
    img = img.resize((128, 128))
    img.save(re_n)

    