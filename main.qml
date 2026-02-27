import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtMultimedia

ApplicationWindow {
    id: window
    visible: true
    width: 480
    height: 800
    title: "Camera App"
    
    leftPadding: 12
    rightPadding: 12

    Camera {
        id: camera
        active: true
    }

    ImageCapture {
        id: imageCapture
        onImageSaved: (requestId, fileName) => {
            console.log("Snapshot saved. requestId:", requestId, "file:", fileName)
        }
        onErrorOccurred: (error, errorString) => {
            console.error("Image capture error:", error, errorString)
        }
    }
        
    CaptureSession {
        id: captureSession
        camera: camera
        imageCapture: imageCapture
        videoOutput: viewfinder
    }
    //
    Timer {
        id: timer
        interval: 5000
        repeat: true
        running: false
    }
    
    
    

    Item {
        anchors.fill: parent

        VideoOutput {
            id: viewfinder
            width: 300
            height: 300
            anchors.left: parent.left
            anchors.verticalCenter: parent.verticalCenter
            fillMode: VideoOutput.PreserveAspectFit
        }
        
        ColumnLayout {
            anchors.fill: parent
            spacing: 12

            // Identify button (toggle loop)
            Button {
                id: buttonIdentify
                anchors.verticalCenter: parent.verticalCenter
                Layout.alignment: Qt.AlignRight | Qt.AlignVCenter
                text: timer.running ? "Stop Identify" : "Start Identify"
                onClicked: {
                    timer.running = !timer.running
                }
            }
            

        // Controls overlay
        Item {
            id: controls
            anchors.left: parent.left
            anchors.right: parent.right
            anchors.bottom: parent.bottom
            anchors.margins: 16
            height: 100
            
            
            // Item { Layout.fillWidth: true }
            Button {
                id: buttonShutter
                anchors.horizontalCenter: parent.horizontalCenter
                anchors.bottom: parent.bottom
                anchors.bottomMargin: 8
                Layout.alignment: Qt.AlignHCenter | Qt.AlignVCenter
                width: 50
                height: 50
                text: ""
                Accessible.name: "Shutter"
                contentItem: Item {}
                background: Rectangle {
                    radius: width / 2
                    color: "transparent"
                    border.width: 10
                    border.color: "grey"
                }
                onClicked: {
                    imageCapture.captureToFile("/Users/krist/Documents/project/iris_on_rpi/captures")
                    background: Rectangle {
                        radius: width / 2
                        color: "transparent"
                        border.width: 20
                        border.color: "grey"
                    }
                }
            }
    
                // Item { Layout.fillWidth: true }


            }
        }
    }
}
