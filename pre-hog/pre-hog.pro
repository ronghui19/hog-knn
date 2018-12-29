TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

#QMAKE_LFLAGS += -L/usr/share/ewaybot/thirdpartylib
#QMAKE_LFLAGS += -Wl,-rpath=/usr/share/ewaybot/thirdpartylib

#INCLUDEPATH += /usr/share/ewaybot/include
INCLUDEPATH += /home/moro/trunk/ThirdPartyLib/caffe/include
INCLUDEPATH += /home/moro/trunk/ThirdPartyLib/glog/include
INCLUDEPATH += /home/moro/trunk/ThirdPartyLib/gflags/include
INCLUDEPATH += /home/moro/trunk/ThirdPartyLib/protobuf/include
INCLUDEPATH += /home/moro/trunk/ThirdPartyLib/opencv-3.1.0/include

QMAKE_LFLAGS += -L/home/moro/trunk/ThirdPartyLib/caffe/build/lib
QMAKE_LFLAGS += -Wl,-rpath=/home/moro/trunk/ThirdPartyLib/caffe/build/lib

QMAKE_LFLAGS += -L/home/moro/trunk/ThirdPartyLib/glog/lib
QMAKE_LFLAGS += -Wl,-rpath=/home/moro/trunk/ThirdPartyLib/glog/lib

QMAKE_LFLAGS += -L/home/moro/trunk/ThirdPartyLib/gflags/lib
QMAKE_LFLAGS += -Wl,-rpath=/home/moro/trunk/ThirdPartyLib/gflags/lib

#QMAKE_LFLAGS += -L/home/moro/trunk/ThirdPartyLib/protobuf
#QMAKE_LFLAGS += -Wl,-rpath=/home/moro/trunk/ThirdPartyLib/protobuf


QMAKE_LFLAGS += -L/home/moro/trunk/ThirdPartyLib/boost/lib
QMAKE_LFLAGS += -Wl,-rpath=/home/moro/trunk/ThirdPartyLib/boost/lib

QMAKE_LFLAGS += -L/home/moro/trunk/ThirdPartyLib/opencv-3.1.0/build/lib
QMAKE_LFLAGS += -Wl,-rpath=/home/moro/trunk/ThirdPartyLib/opencv-3.1.0/build/lib


LIBS += -lopencv_core
LIBS += -lopencv_highgui
LIBS += -lopencv_imgcodecs
LIBS += -lopencv_features2d
LIBS += -lopencv_calib3d
LIBS += -lopencv_imgproc
LIBS += -lopencv_objdetect
LIBS += -lopencv_ml

LIBS += -lcaffe
LIBS += -lglog
LIBS += -lgflags
LIBS += -lprotobuf
LIBS += -lboost_system

SOURCES += main.cpp
