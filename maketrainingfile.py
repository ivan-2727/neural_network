from PIL import Image
import os

base = "./trainingset"
train = open("train.txt", "w", newline='')
train.write("10\n\n")
cnt = 0
for folder in os.listdir(base):
    if os.path.isdir(base + "/" + folder):
        ld = os.listdir(base + "/" + folder)
        mx = (len(ld)*3)//4
        #train.write(str(len(ld)-mx) + "\n\n")
        train.write(str(mx) + "\n\n")
        #for k in range(mx,len(ld)):
        for k in range(mx):
            filename = ld[k]
            path = base + "/" + folder + "/" + filename
            try:
                im = Image.open(path, 'r')
                width, height = im.size
                sample = []
                for i in range(width):
                    for j in range(height):
                        sample.append(im.getpixel((i,j)))
                for x in sample:
                    train.write(str(x) + " ")
                train.write("\n")
                train.write(folder + "\n\n")
            except:
                print("something went wrong")

print("all ", cnt)            