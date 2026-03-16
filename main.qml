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
    
    
    Timer {
        id: timer
        interval: 5000
        repeat: true
        running: false
    }

    Item {
        anchors.fill: parent
        anchors.margins: 12

        VideoOutput {
            id: viewfinder
            width: 300
            height: 300
            anchors.left: parent.left
            anchors.verticalCenter: parent.verticalCenter
            fillMode: VideoOutput.PreserveAspectFit
        }
        
        // Actions panel to the right of the camera preview
        Column {
            id: actionsPanel
            anchors.left: viewfinder.right
            anchors.leftMargin: 32
            anchors.verticalCenter: viewfinder.verticalCenter
            spacing: 64

            // Identify button (toggle loop)
            Button {
                id: buttonIdentify
                text: timer.running ? "Stop Identify" : "Start Identify"
                onClicked: { timer.running = !timer.running }
            }

            // Enroll button
            Button {
                id: buttonEnroll
                text: "Enroll"
                onClicked: { console.log("Enroll tapped") }
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
                }
            }
    
            // Item { Layout.fillWidth: true }
        }
    }
}
