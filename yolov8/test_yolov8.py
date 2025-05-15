from ultralytics import YOLO


if __name__ == '__main__':

    model = YOLO('/civi/data/hit_children/runs/detect/train_imgs280-bg286_50eps_v8x/weights/best.pt')
    model.predict(source='/civi/data/hit_children/testdemo6_328/', conf = 0.3, save = True)
