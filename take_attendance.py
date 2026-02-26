# import library
import dlib
import numpy as np
import cv2
import os
import pandas as pd
import time
import logging
import sqlite3
import datetime

# loading the models of Dlib
# Dlib  / Use frontal face detector of Dlib
detector = dlib.get_frontal_face_detector()
# Dlib landmark / Get face landmarks
predictor = dlib.shape_predictor('data_dlib/shape_predictor_68_face_landmarks.dat')
# Dlib Resnet Use Dlib resnet50 model to get 128D face descriptor
face_reco_model = dlib.face_recognition_model_v1("data_dlib/dlib_face_recognition_resnet_model_v1.dat")
# Create a connection to the database
conn = sqlite3.connect("attendance.db")
cursor = conn.cursor()
# Create a table for the current date
current_date = datetime.datetime.now().strftime("%Y_%m_%d")  # Replace hyphens with underscores
table_name = "attendance" 
create_table_sql = f"CREATE TABLE IF NOT EXISTS {table_name} (name TEXT, time TEXT, date DATE, UNIQUE(name, date))"
cursor.execute(create_table_sql)
# Commit changes and close the connection
conn.commit()
conn.close()

class Face_Recognizer:
    def __init__(self):
        self.font = cv2.FONT_ITALIC
        # FPS
        self.frame_time = 0
        self.frame_start_time = 0
        self.fps = 0
        self.fps_show = 0
        self.start_time = time.time()

        # cnt for frame
        self.frame_cnt = 0

        #  Save the features of faces in the database
        self.face_features_known_list = []
        # / Save the name of faces in the sqlite database
        self.face_name_known_list = []

        #  List to save centroid positions of ROI in frame N-1 and N
        self.last_frame_face_centroid_list = []
        self.current_frame_face_centroid_list = []

        # List to save names of objects in frame N-1 and N
        self.last_frame_face_name_list = []
        self.current_frame_face_name_list = []

        #  cnt for faces in frame N-1 and N
        self.last_frame_face_cnt = 0
        self.current_frame_face_cnt = 0

        # Save the e-distance for faceX when recognizing
        self.current_frame_face_X_e_distance_list = []

        # Save the positions and names of current faces captured
        self.current_frame_face_position_list = []
        #  Save the features of people in current frame
        self.current_frame_face_feature_list = []

        # e distance between centroid of ROI in last and current frame
        self.last_current_frame_centroid_e_distance = 0

        #  Reclassify after 'reclassify_interval' frames
        self.reclassify_interval_cnt = 0
        self.reclassify_interval = 10

        # Liveness detection: track face landmarks to detect real motion
        self.last_frame_landmarks_list = []
        self.landmarks_motion_threshold = 5           # Min pixel movement (increase for stricter)
        self.face_motion_detected = {}  # Track motion for each face
        self.consistent_match_count = {}  # Track consecutive frames with same person
        self.required_consistent_frames = 5           # Frames needed before attendance

        # Attendance cooldown to prevent duplicate marking
        self.last_attendance_time = {}  # {person_name: timestamp}
        self.attendance_cooldown_seconds = 60         # Seconds between duplicate marks

    #  "features_all.csv"  / Get known faces from "features_all.csv"
    def get_face_database(self):
        if os.path.exists("data/features_all.csv"):
            path_features_known_csv = "data/features_all.csv"
            csv_rd = pd.read_csv(path_features_known_csv, header=None)
            for i in range(csv_rd.shape[0]):
                features_someone_arr = []
                self.face_name_known_list.append(csv_rd.iloc[i][0])
                for j in range(1, 129):
                    if csv_rd.iloc[i][j] == '':
                        features_someone_arr.append('0')
                    else:
                        features_someone_arr.append(csv_rd.iloc[i][j])
                self.face_features_known_list.append(features_someone_arr)
            logging.info("Faces in Database： %d", len(self.face_features_known_list))
            return 1
        else:
            logging.warning("'features_all.csv' not found!")
            logging.warning("Please run take_attandance.py and features_extraction.py before 'faces_register.py'")
            return 0

    def update_fps(self):
        now = time.time()
        # Refresh fps per second
        if str(self.start_time).split(".")[0] != str(now).split(".")[0]:
            self.fps_show = self.fps
        self.start_time = now
        self.frame_time = now - self.frame_start_time
        self.fps = 1.0 / self.frame_time
        self.frame_start_time = now

    @staticmethod
    # / Compute the e-distance between two 128D features
    def return_euclidean_distance(feature_1, feature_2):
        feature_1 = np.array(feature_1)
        feature_2 = np.array(feature_2)
        dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
        return dist

    # Detect facial landmarks motion (liveness detection)
    def detect_landmark_motion(self, img_rd, faces):
        motion_detected_list = []
        current_landmarks_list = []

        for face_idx, face in enumerate(faces):
            try:
                shape = predictor(img_rd, face)
                landmarks = np.array([(p.x, p.y) for p in shape.parts()])
                current_landmarks_list.append(landmarks)

                # Check motion only if we have previous landmarks
                if face_idx < len(self.last_frame_landmarks_list):
                    last_landmarks = self.last_frame_landmarks_list[face_idx]
                    # Calculate average movement of all landmarks
                    motion = np.mean(np.sqrt(np.sum((landmarks - last_landmarks) ** 2, axis=1)))
                    is_moving = motion > self.landmarks_motion_threshold
                    motion_detected_list.append(is_moving)
                    logging.debug(f"Face {face_idx} landmark motion: {motion:.2f} pixels")
                else:
                    motion_detected_list.append(False)
            except:
                motion_detected_list.append(False)

        self.last_frame_landmarks_list = current_landmarks_list
        return motion_detected_list

    # Check if enough time has passed for another attendance record
    def can_mark_attendance(self, person_name):
        current_time = time.time()     
        if person_name not in self.last_attendance_time:
            return True      
        time_elapsed = current_time - self.last_attendance_time[person_name]
        return time_elapsed >= self.attendance_cooldown_seconds

    # / Use centroid tracker to link face_x in current frame with person_x in last frame
    def centroid_tracker(self):
        for i in range(len(self.current_frame_face_centroid_list)):
            e_distance_current_frame_person_x_list = []
            #  For object 1 in current_frame, compute e-distance with object 1/2/3/4/... in last frame
            for j in range(len(self.last_frame_face_centroid_list)):
                self.last_current_frame_centroid_e_distance = self.return_euclidean_distance(
                    self.current_frame_face_centroid_list[i], self.last_frame_face_centroid_list[j])

                e_distance_current_frame_person_x_list.append(
                    self.last_current_frame_centroid_e_distance)

            last_frame_num = e_distance_current_frame_person_x_list.index(
                min(e_distance_current_frame_person_x_list))
            self.current_frame_face_name_list[i] = self.last_frame_face_name_list[last_frame_num]

    #  cv2 window / putText on cv2 window
    def draw_note(self, img_rd):
        cv2.putText(img_rd, "Take Attendance (Smart-Attandance System)", (20, 40), self.font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "Frame:  " + str(self.frame_cnt), (20, 100), self.font, 0.8, (0, 255, 0), 1,cv2.LINE_AA)
        cv2.putText(img_rd, "FPS:    " + str(self.fps.__round__(2)), (20, 130), self.font, 0.8, (0, 255, 0), 1,cv2.LINE_AA)
        cv2.putText(img_rd, "Faces:  " + str(self.current_frame_face_cnt), (20, 160), self.font, 0.8, (0, 255, 0), 1,cv2.LINE_AA)
        cv2.putText(img_rd, "q: Quit", (20, 450), self.font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

        for i in range(len(self.current_frame_face_name_list)):
            img_rd = cv2.putText(img_rd, "Face_" + str(i + 1), tuple(
                [int(self.current_frame_face_centroid_list[i][0]), int(self.current_frame_face_centroid_list[i][1])]),
                                 self.font,
                                 0.8, (255, 190, 0),
                                 1,
                                 cv2.LINE_AA)
    # insert data in database
    def attendance(self, name):
        """
        Mark attendance only if liveness check passed and cooldown period has elapsed.
        """
        # Check cooldown period
        if not self.can_mark_attendance(name):
            logging.info(f"{name} skipped: Within cooldown period (< {self.attendance_cooldown_seconds}s)")
            return False
        current_date = datetime.datetime.now().strftime('%Y-%m-%d')
        current_time = datetime.datetime.now().strftime('%H:%M:%S')

        try:
            with sqlite3.connect("attendance.db") as conn:
                cursor = conn.cursor()
                # Use INSERT OR IGNORE to handle uniqueness atomically (prevents race conditions)
                cursor.execute("INSERT OR IGNORE INTO attendance (name, time, date) VALUES (?, ?, ?)",(name, current_time, current_date),)

                # If the insert was ignored (already present), rowcount will be 0
                if cursor.rowcount == 0:
                    logging.info(f"{name} is already marked as present for {current_date}")
                    return False

                conn.commit()
                self.last_attendance_time[name] = time.time()
                logging.info(f"{name} marked as present for {current_date} at {current_time}")
                print(f"{name} marked as present for {current_date} at {current_time}")
                return True

        except sqlite3.IntegrityError:
            logging.warning(f"IntegrityError: {name} likely already present for {current_date}")
            return False
        except Exception as e:
            logging.exception("Error marking attendance: %s", e)
            return False

    #  Face detection and recognition wit OT from input video stream
    def process(self, stream):
        # 1.  Get faces known from "features.all.csv"
        if self.get_face_database():
            while stream.isOpened():
                self.frame_cnt += 1
                logging.debug("Frame " + str(self.frame_cnt) + " starts")
                flag, img_rd = stream.read()
                kk = cv2.waitKey(1)

                # 2.  Detect faces for frame X
                faces = detector(img_rd, 0)

                # 3.  Update cnt for faces in frames
                self.last_frame_face_cnt = self.current_frame_face_cnt
                self.current_frame_face_cnt = len(faces)

                # 4.  Update the face name list in last frame
                self.last_frame_face_name_list = self.current_frame_face_name_list[:]

                # 5.  update frame centroid list
                self.last_frame_face_centroid_list = self.current_frame_face_centroid_list
                self.current_frame_face_centroid_list = []

                # 6.1  if cnt not changes
                if (self.current_frame_face_cnt == self.last_frame_face_cnt) and (self.reclassify_interval_cnt != self.reclassify_interval):
                    logging.debug("scene 1:   No face cnt changes in this frame!!!")
                    self.current_frame_face_position_list = []
                    if "unknown" in self.current_frame_face_name_list:
                        self.reclassify_interval_cnt += 1
                    if self.current_frame_face_cnt != 0:
                        for k, d in enumerate(faces):
                            self.current_frame_face_position_list.append(tuple(
                                [faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4)]))
                            self.current_frame_face_centroid_list.append(
                                [int(faces[k].left() + faces[k].right()) / 2,
                                 int(faces[k].top() + faces[k].bottom()) / 2])

                            img_rd = cv2.rectangle(img_rd,
                                                   tuple([d.left(), d.top()]),
                                                   tuple([d.right(), d.bottom()]),
                                                   (255, 255, 255), 2)

                    #  Multi-faces in current frame, use centroid-tracker to track
                    if self.current_frame_face_cnt != 1:
                        self.centroid_tracker()

                    for i in range(self.current_frame_face_cnt):
                        # 6.2 Write names under ROI
                        img_rd = cv2.putText(img_rd, self.current_frame_face_name_list[i],
                                             self.current_frame_face_position_list[i], self.font, 0.8, (0, 255, 255), 1,
                                             cv2.LINE_AA)
                    self.draw_note(img_rd)

                # 6.2  If cnt of faces changes, 0->1 or 1->0 or ...
                else:
                    logging.debug("scene 2: / Faces cnt changes in this frame")
                    self.current_frame_face_position_list = []
                    self.current_frame_face_X_e_distance_list = []
                    self.current_frame_face_feature_list = []
                    self.reclassify_interval_cnt = 0

                    # 6.2.1  Face cnt decreases: 1->0, 2->1, ...
                    if self.current_frame_face_cnt == 0:
                        logging.debug("  / No faces in this frame!!!")
                        # clear list of names and features
                        self.current_frame_face_name_list = []
                        self.last_frame_landmarks_list = []
                    # 6.2.2 / Face cnt increase: 0->1, 0->2, ..., 1->2, ...
                    else:
                        logging.debug("  scene 2.2  Get faces in this frame and do face recognition")
                        self.current_frame_face_name_list = []
                        
                        # Detect facial landmarks motion (liveness detection)
                        motion_detected_list = self.detect_landmark_motion(img_rd, faces)

                        for i in range(len(faces)):
                            shape = predictor(img_rd, faces[i])
                            self.current_frame_face_feature_list.append(
                                face_reco_model.compute_face_descriptor(img_rd, shape))
                            self.current_frame_face_name_list.append("unknown")

                        # 6.2.2.1 Traversal all the faces in the database
                        recognized_in_frame = set()
                        for k in range(len(faces)):
                            logging.debug("  For face %d in current frame:", k + 1)
                            self.current_frame_face_centroid_list.append(
                                [int(faces[k].left() + faces[k].right()) / 2,
                                 int(faces[k].top() + faces[k].bottom()) / 2])

                            self.current_frame_face_X_e_distance_list = []

                            # 6.2.2.2  Positions of faces captured
                            self.current_frame_face_position_list.append(tuple(
                                [faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4)]))

                            # 6.2.2.3 
                            # For every faces detected, compare the faces in the database
                            for i in range(len(self.face_features_known_list)):
                                # 
                                if str(self.face_features_known_list[i][0]) != '0.0':
                                    e_distance_tmp = self.return_euclidean_distance(
                                        self.current_frame_face_feature_list[k],
                                        self.face_features_known_list[i])
                                    logging.debug("      with student %d, the e-distance: %f", i + 1, e_distance_tmp)
                                    self.current_frame_face_X_e_distance_list.append(e_distance_tmp)
                                else:
                                    #  student_X
                                    self.current_frame_face_X_e_distance_list.append(999999999)

                            # 6.2.2.4 / Find the one with minimum e distance
                            similar_person_num = self.current_frame_face_X_e_distance_list.index(min(self.current_frame_face_X_e_distance_list))

                            if min(self.current_frame_face_X_e_distance_list) < 0.4:
                                recognized_name = self.face_name_known_list[similar_person_num]
                                
                                # LIVENESS CHECK: Only mark attendance if motion is detected
                                if k < len(motion_detected_list) and motion_detected_list[k]:
                                    logging.info("✓ LIVE MOTION DETECTED - Face recognition result: %s", recognized_name)
                                    
                                    # Update consistent match counter for this person
                                    if recognized_name not in self.consistent_match_count:
                                        self.consistent_match_count[recognized_name] = 0
                                    self.consistent_match_count[recognized_name] += 1
                                    
                                    # Mark attendance only after consistent matches with motion
                                    if self.consistent_match_count[recognized_name] >= self.required_consistent_frames:
                                        self.current_frame_face_name_list[k] = recognized_name
                                        self.attendance(recognized_name)
                                        self.consistent_match_count[recognized_name] = 0  # Reset counter
                                    else:
                                        self.current_frame_face_name_list[k] = recognized_name + "*"
                                        logging.debug(f"Confirming motion: {self.consistent_match_count[recognized_name]}/{self.required_consistent_frames}")
                                        self.attendance(recognized_name)
                                        self.consistent_match_count[recognized_name] = 0  # Reset counter
                                    # record that this person was seen this frame
                                    recognized_in_frame.add(recognized_name)
                                else:
                                    logging.warning("✗ NO MOTION - Likely a photo/static image: %s", recognized_name)
                                    self.current_frame_face_name_list[k] = recognized_name + " (NO MOTION) "
                                    self.consistent_match_count[recognized_name] = 0  # Reset counter
                            else:
                                logging.debug("  Face recognition result: Unknown student")
                                # Do not clear all counters on unknowns; only prune after the frame
                                pass

                        # Prune consistent_match_count keys for people not seen in this frame
                        for name in list(self.consistent_match_count.keys()):
                            if name not in recognized_in_frame:
                                del self.consistent_match_count[name]

                        # 7.  / Add note on cv2 window
                        self.draw_note(img_rd)

                # 8.  'q'  / Press 'q' to exit
                if kk == ord('q'):
                    break

                self.update_fps()
                cv2.namedWindow("camera", 1)
                cv2.imshow("camera", img_rd)

                logging.debug("Frame ends\n\n")

    


    def run(self):
        # cap = cv2.VideoCapture("video.mp4")  # Get video stream from video file
        cap = cv2.VideoCapture(0)              # Get video stream from camera
        self.process(cap)

        cap.release()
        cv2.destroyAllWindows()

def main():
    # logging.basicConfig(level=logging.DEBUG) # Set log level to 'logging.DEBUG' to print debug info of every frame
    logging.basicConfig(level=logging.INFO)
    Face_Recognizer_con = Face_Recognizer()
    Face_Recognizer_con.run()


if __name__ == '__main__':
    main()
