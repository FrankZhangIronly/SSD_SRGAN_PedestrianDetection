import cv2
from lxml import etree
colors_tableau = [(255, 255, 255), (31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229),(158, 218, 229),(158, 218, 229)]
img = cv2.imread("./data/VOCdevkit/VOC2007/JPEGImages/BAHNHOF/image_00000629_0.png")
anno = etree.parse("./data/VOCdevkit/VOC2007/Annotations/BAHNHOF/image_00000629_0.xml").getroot()

def getbnd(target, width, height):
    res = []
    for obj in target.iter('object'):
        # difficult = int(obj.find('difficult').text) == 1
        # if not self.keep_difficult and difficult:
        # continue
        name = obj.find('name').text.lower().strip()
        bbox = obj.find('bndbox')

        pts = ['xmin', 'ymin', 'xmax', 'ymax']
        bndbox = []
        for i, pt in enumerate(pts):
            cur_pt = int(bbox.find(pt).text) - 1
            # scale height or width
            cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
            bndbox.append(cur_pt)
        res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
        # img_id = target.find('filename').text[:-4]
    return res
res = getbnd(anno, 1, 1)
for pt in res:
    coords = (pt[0], pt[1], pt[2], pt[3])
    # add show
    # display_txt = '%s: %.2f' % (label_name, score)
    color = colors_tableau[4]
    print(coords)
    cv2.rectangle(img, (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), color, 2)
    #cv2.putText(img, display_txt, (int(pt[0]), int(pt[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, 10)
cv2.imshow('test',img)
cv2.waitKey(0)
cv2.imwrite("./test.png",img)