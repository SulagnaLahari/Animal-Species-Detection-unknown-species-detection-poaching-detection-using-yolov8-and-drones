import ultralytics
from ultralytics import YOLO
import cv2

# Load Model (replace with your trained model path)
model = YOLO('best.pt')

# Define a dictionary mapping unknown class to similar classes (modify as needed)
unknown_similarity = {
    0:  ['insect', 'Spider'],
    1:  ['bird', 'Parrot'],
    2:  ['insect', 'Scorpion'],  # Technically an arachnid, but might be confused with insect
    3:  ['Chelonioidea', 'Sea turtle or turtle'],
    4:  ['ungulate', 'Cow or Buffalo'],
    5:  ['mammal', 'Canine or Jackals'],
    6:  ['mammal', 'Moles or Shrews'],
    7:  ['reptile','Tortoise or Terrapin'],
    8:  ['mammal', 'Domestic cat or Puma'],
    9:  ['reptile','Snake'],
    10: ['fish','Sawfish'],
    11: ['ungulate', 'Donkey or Asses'],
    12: ['bird','Pied currawong'],
    13: ['rodent','American pika'],
    14: ['bird','Sapsucker'],
    15: ['bird','Turkey'],
    16: ['bird','Penguin'],
    17: ['insect','Butterfly'],
    18: ['mammal', 'Lion'],  # Large cat
    19: ['mammal', 'Weasel or Badgers'],
    20: ['human', 'Person'],
    21: ['mammal','Coati'],
    22: ['mammal', 'Hippopotamus'],  # Large herbivore
    23: ['mammal','Seals'],
    24: ['bird', 'Grouse or Turkey'],
    25: ['mammal','Hog or Peccaries'],
    26: ['bird','Owl'],
    27: ['insect', 'Larva'],
    28: ['mammal', 'Marsupial'],
    29: ['mammal', 'Bear'],  # Subtype of bear
    30: ['invertebrate', 'Squid'],
    31: ['mammal', 'Dolphins or Porpoise'],
    32: ['mammal', 'Seal'],
    33: ['bird','crow'],
    34: ['mammal','Rodent'],
    35: ['mammal', 'Tiger'],  # Large cat
    36: ['reptile','Lizard or Crocodile'],
    37: ['insect', 'Beetle'],
    38: ['mammal','Panda'],
    39: ['mammal', 'Kangaroo'],
    40: ['echinoderm','Fish'],
    41: ['invertebrate','Millipede'],
    42: ['reptile', 'Turtle'],
    43: ['bird', 'Rheas'],
    44: ['fish','Fish'],
    45: ['amphibian','Frog'],
    46: ['bird','Goose or Pelican'],
    47: ['mammal','Elephant'],
    48: ['mammal','Alpaca'],
    49: ['invertebrate', 'Slug'],
    50: ['mammal', 'Zebra or Okapi or Horse'],  # Similar to horse
    51: ['insect','Moth or Butterflies'],
    52: ['invertebrate', 'Prawn'],
    53: ['vertebrate', 'Fish'],
    54: ['mammal', 'Bear'],  # Subtype of bear
    55: ['mammal', 'Bob cat'],  # Similar to cat
    56: ['bird', 'Goose'],
    57: ['mammal', 'Ocelot'],  # Large cat
    58: ['bird', 'Duck'],
    59: ['mammal','Yalk or Bison'],
    60: ['mammal', 'Rodent'],
    61: ['mammal','Giraffe'],
    62: ['invertebrate', 'Crab'],
    63: ['invertebrate', 'Scorpions'],
    64: ['mammal', 'Lemurs or Apes'],
    65: ['mammal', 'Cattle'],
    66: ['fish','Seahorse'],
    67: ['invertebrate', 'Millipede'],
    68: ['mammal', 'Donkey'],  # Similar to horse
    69: ['mammal','Rhinoceros'],
    70: ['bird','Wild Canary'],
    71: ['mammal','Camel'],
    72: ['mammal', 'Bear'],  # Subtype of bear
    73: ['bird','Sparrow'],
    74: ['mammal', 'Rodent or Squirrel'],
    75: ['mammal', 'Leopard'],  # Large cat
    76: ['cnidarian','Fish'],
    77: ['reptile','Crocodiles'],
    78: ['mammal','Sambar'],
    79: ['bird', 'Turkey'],
    80: ['mammal', 'Seal'] 
}

# Class Names (replace with your actual class names)
class_names = ['Spider', 'Parrot', 'Scorpion', 'Sea turtle', 'Cattle', 'Fox', 'Hedgehog',
              'Turtle', 'Cheetah', 'Snake', 'Shark', 'Horse', 'Magpie', 'Hamster',
              'Woodpecker', 'Eagle', 'Penguin', 'Butterfly', 'Lion', 'Otter','Poacher', 'Raccoon',
              'Hippopotamus', 'Bear', 'Chicken', 'Pig', 'Owl', 'Caterpillar', 'Koala',
              'Polar bear', 'Squid', 'Whale', 'Harbor seal', 'Raven', 'Mouse', 'Tiger',
              'Lizard', 'Ladybug', 'Red panda', 'Kangaroo', 'Starfish', 'Worm', 'Tortoise',
              'Ostrich', 'Goldfish', 'Frog', 'Swan', 'Elephant', 'Sheep', 'Snail', 'Zebra', 'Moth and butterflies', 
              'Shrimp', 'Fish', 'Panda', 'Lynx', 'Duck', 'Jaguar', 'Goose', 'Goat', 'Rabbit', 'Giraffe', 'Crab',
              'Tick', 'Monkey', 'Bull', 'Seahorse', 'Centipide', 'Mule', 'Rhinoceros', 'Canary', 'Camel', 'Brown Bera',
              'Sparrow', 'Squirrel', 'Leopard', 'Jellyfish', 'Crocodile', 'Deer','Turkey', 'Sea Lion']

########################
#wCam, hCam = 640, 480
########################

# Open Video Stream (or use '0' for webcam)
cap = cv2.VideoCapture("poaching_detection1.mp4")
#cap.set(3,wCam)
#cap.set(4,hCam)
while cap.isOpened():
    success, frame = cap.read()

    if success:
        # Run Inference on Frame
        results = model.predict(frame, show=True)  # Assuming model takes OpenCV image directly

        for result in results:
            boxes = result.boxes

            for box in boxes:
                cls = int(box.cls)  # Get predicted class
                conf = box.conf  # Get confidence score

                if class_names[cls] == 'Poacher':  # Check if class is Poacher
                    print("Poaching Activity and Poacher Detected!!")
                else:
                    if class_names[cls] not in class_names:  # Check if class is unknown
                        if conf < 0.5:  # Check for high confidence
                            # Suggest similar class based on the dictionary
                            if cls in unknown_similarity:
                                similar_class = unknown_similarity[cls][0]
                                print(f"Potential unknown species detected! Confidence: {conf.item():.2f}, Most similar species like: {similar_class}")
                                similar_type = unknown_similarity[cls][1]
                                print(f"{conf.item():.2f}, Most similar animal like: {similar_type}")
                            else:
                                print(f"Potential unknown class detected! Confidence: {conf.item():.2f}, Class: {cls}")

                            # Draw Bounding Box and Label on Frame
                            x1, y1, x2, y2 = box.xyxy[0]  # Get coordinates from box object
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # Green rectangle
                            
                            # Label text
                            if cls.item() in unknown_similarity:
                              similar_class = unknown_similarity[cls.item()][0]
                              label_text = f"Unknown ({conf.item():.2f}) - Most similar to: {similar_class}"
                              similar_type = unknown_similarity[cls.item()][1]
                              label_text=f"Similar Species like ({conf.item():.2f}) - Most similar to: {similar_type}"
                            else:
                              label_text = f"{cls.item()} ({conf.item():.2f}) - {cls}"
                            cv2.putText(frame, label_text, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    
                    # Display Frame
                    cv2.imshow('Unknown Species Detection:', frame)

                    # Exit on 'q' Key Press
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                      break

# Release Video Capture and Close Windows
cap.release()
cv2.destroyAllWindows()