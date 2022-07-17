from PIL import Image
import os

base = "./trainingset"
for folder in os.listdir(base):
    if os.path.isdir(base+ "/" + folder):
        for filename in os.listdir(base +"/" + folder):
            path = base+"/" + folder + "/" + filename
            print(path)
            if os.path.isfile(path):
                try:
                    im = Image.open(path, 'r')
                    width, height = im.size
                    pixels = [[0 for _ in range(height)] for _ in range(width)]
                    
                    for i in range(width):
                        for j in range(height):
                            pixels[i][j] = im.getpixel((i,j))
                            
                    low = 123
                    top = height+10
                    bottom = -1
                    left = width+10
                    right = -1

                    for i in range(width):
                        for j in range(height):
                            if pixels[i][j] >= low:
                                top = min(top, j)
                                bottom = max(bottom, j)
                                left = min(left, i)
                                right = max(right, i)

                    image_out = im.crop((left, top, right, bottom)).resize((15,15))
                    sample = []
                    for i in range(10):
                        for j in range(10):
                            sample.append(image_out.getpixel((i,j)))
                    image_out.save(path)
                except:
                    print("Something went wrong")
                    os.remove(path)