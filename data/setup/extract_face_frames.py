import os, glob, cv2

# get cwd: current working directory
try:
    cwd = os.path.dirname(os.path.abspath(__file__)) + '/'
except:
    cwd = ''

expand_factor = 1.5  # expand the detected face region
image_size = 256     # resize the detected face region
frameRoot = cwd + '../face/expand%.1f_size%d/'%(expand_factor, image_size)

def mkdir_p(path):
    try:
        os.makedirs(path)
    except Exception as e:
        pass

def expand_region(region, factor):
    x1, y1, x2, y2 = region
    h, w = y2 - y1 + 1, x2 - x1 + 1
    xc, yc = (x2 + x1)/2.0, (y2 + y1)/2.0
    y1, y2 = yc - h*factor/2.0, yc + h*factor/2.0
    x1, x2 = xc - w*factor/2.0, xc + w*factor/2.0
    region = list(map(lambda x: max(0, int(x)), [x1, y1, x2, y2]))
    return region

def square_region(region):
    x1, y1, x2, y2 = region
    h, w = y2 - y1 + 1, x2 - x1 + 1
    hw = max(h, w)
    x2, y2 = x1 + hw - 1, y1 + hw - 1
    return [x1, y1, x2, y2]

def extract_face_frames(videoPath, region, dest, expand, resize, pattern='frame_%06d.jpg'):
    videoID = os.path.basename(videoPath)
    mkdir_p(os.path.join(dest, videoID))
    x1, y1, x2, y2 = expand_region(region, expand)

    destPattern = os.path.join(dest, videoID, pattern)
    cap = cv2.VideoCapture(videoPath)
    count = 0
    while cap.isOpened():
        count += 1
        ret, frame = cap.read()
        if ret:
            frame = frame[y1:y2+1, x1:x2+1, :]
            frame = cv2.resize(frame, (image_size, image_size)) 
            cv2.imwrite(destPattern%(count), frame)
        else:
            break
    cap.release()

# get a list of videos with detected face regions
videos = []
regions = []
videoRoots = [cwd + '../FaceForensics_AVIs/train',
              cwd + '../FaceForensics_AVIs/valid']
for videoRoot in videoRoots:
    with open(os.path.join(videoRoot, 'rect.txt'), 'r') as f:
        for line in f.readlines():
            tmp = line.rstrip().split()
            videos.append(os.path.join(videoRoot, tmp[0]))
            region = list(map(int, tmp[1:]))
            region = square_region(expand_region(region, expand_factor))
            regions.append(region)

for i, (video, region) in enumerate(zip(videos, regions)):
    print('%d/%d'%(i, len(videos)))
    extract_face_frames(video, region, frameRoot, expand=expand_factor, resize=image_size)

print('Done!')
