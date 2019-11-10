#pragma once

#include "ofMain.h"

#include <Eigen/Geometry>
#include <Eigen/StdVector>

class ofApp : public ofBaseApp {

public:
	void setup();
	void update();
	void draw();

	void updateBalls(Eigen::MatrixXd& bc, Eigen::VectorXd& Beq);
	void updateShape();

	void keyPressed(int key);
	void keyReleased(int key);
	void mouseMoved(int x, int y);
	void mouseDragged(int x, int y, int button);
	void mousePressed(int x, int y, int button);
	void mouseReleased(int x, int y, int button);
	void mouseEntered(int x, int y);
	void mouseExited(int x, int y);
	void windowResized(int w, int h);
	void dragEvent(ofDragInfo dragInfo);
	void gotMessage(ofMessage msg);

	ofMesh mMesh;
	ofEasyCam mCam;
	ofLight mLight;
	int m_closest_ball_index;
	bool m_enable_cameraInput;

};
