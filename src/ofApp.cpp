#include "ofApp.h"
#include <igl/colon.h>
#include <igl/directed_edge_orientations.h>
#include <igl/directed_edge_parents.h>
#include <igl/per_vertex_normals.h>
#include <igl/forward_kinematics.h>
#include <igl/PI.h>
#include <igl/partition.h>
#include <igl/mat_max.h>
#include <igl/lbs_matrix.h>
#include <igl/slice.h>
#include <igl/deform_skeleton.h>
#include <igl/dqs.h>
#include <igl/lbs_matrix.h>
#include <igl/columnize.h>
#include <igl/readDMAT.h>
#include <igl/readOBJ.h>
#include <igl/arap.h>
#include <igl/arap_dof.h>

#include <algorithm>

typedef
std::vector<Eigen::Quaterniond, Eigen::aligned_allocator<Eigen::Quaterniond> >
RotationList;

const Eigen::RowVector3d sea_green(70. / 255., 252. / 255., 167. / 255.);
Eigen::MatrixXd V, U, M;
Eigen::MatrixXi F;
Eigen::VectorXi S, b;
Eigen::MatrixXd L;
Eigen::RowVector3d mid;
double anim_t = 0.0;
double anim_t_dir = 0.03;
double bbd = 1.0;
bool resolve = true;
igl::ARAPData arap_data, arap_grouped_data;
igl::ArapDOFData<Eigen::MatrixXd, double> arap_dof_data;
Eigen::SparseMatrix<double> Aeq;
Eigen::MatrixXd BallPos;

enum ModeType
{
	MODE_TYPE_ARAP = 0,
	MODE_TYPE_ARAP_GROUPED = 1,
	MODE_TYPE_ARAP_DOF = 2,
	NUM_MODE_TYPES = 4
} mode;

void convertMesh_IGL2OF(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, ofMesh& mesh) {

	Eigen::MatrixXd N;
	igl::per_vertex_normals(V, F, N);

	mesh.clear();

	for (int i = 0; i < V.rows(); i++) {
		mesh.addVertex(glm::vec3(V(i, 0), V(i, 1), V(i, 2)));
	}

	for (int i = 0; i < N.rows(); i++) {
		mesh.addNormal(glm::vec3(N(i, 0), N(i, 1), N(i, 2)));
	}

	for (int i = 0; i < F.rows(); i++) {
		mesh.addTriangle(F(i, 2), F(i, 1), F(i, 0));
	}

	//cout << "MESH SYNTHESIS" << endl;
	//cout << "NUM OF V: " << mesh.getNumVertices() << endl;
	//cout << "NUM OF N: " << mesh.getNumNormals() << endl;
	//cout << "NUM OF FACES: " << mesh.getIndices().size() / 3 << endl;
}

//--------------------------------------------------------------
void ofApp::setup() {

	mode = MODE_TYPE_ARAP_DOF;

	m_enable_cameraInput = true;
	mCam.enableMouseInput();

	using namespace Eigen;
	using namespace std;
    
    string filepath = ofToDataPath("");

    igl::readOBJ(filepath + "armadillo.obj", V, F);
	convertMesh_IGL2OF(V, F, mMesh);

	U = V;
	MatrixXd W;
	igl::readDMAT(filepath + "armadillo-weights.dmat", W);
	igl::lbs_matrix_column(V, W, M);

	// Cluster according to weights
	VectorXi G;
	{
		VectorXi S;
		VectorXd D;
		igl::partition(W, 50, G, S, D);
	}

	// vertices corresponding to handles (those with maximum weight)
	{
		VectorXd maxW;
		igl::mat_max(W, 1, maxW, b);
	}

	// Precomputation for FAST
	cout << "Initializing Fast Automatic Skinning Transformations..." << endl;
	// number of weights
	const int m = W.cols();
	Aeq.resize(m * 3, m * 3 * (3 + 1));
	vector<Triplet<double> > ijv;
	for (int i = 0; i < m; i++)
	{
		RowVector4d homo;
		homo << V.row(b(i)), 1.;
		for (int d = 0; d < 3; d++)
		{
			for (int c = 0; c < (3 + 1); c++)
			{
				ijv.push_back(Triplet<double>(3 * i + d, i + c * m * 3 + d * m, homo(c)));
			}
		}
	}
	Aeq.setFromTriplets(ijv.begin(), ijv.end());
	igl::arap_dof_precomputation(V, F, M, G, arap_dof_data);
	igl::arap_dof_recomputation(VectorXi(), Aeq, arap_dof_data);
	// Initialize
	MatrixXd Istack = MatrixXd::Identity(3, 3 + 1).replicate(1, m);
	igl::columnize(Istack, m, 2, L);

	// Precomputation for ARAP
	cout << "Initializing ARAP..." << endl;
	arap_data.max_iter = 1;
	igl::arap_precomputation(V, F, V.cols(), b, arap_data);
	// Grouped arap
	cout << "Initializing ARAP with grouped edge-sets..." << endl;
	arap_grouped_data.max_iter = 2;
	arap_grouped_data.G = G;
	igl::arap_precomputation(V, F, V.cols(), b, arap_grouped_data);


	// bounding box diagonal
	bbd = (V.colwise().maxCoeff() - V.colwise().minCoeff()).norm();

	MatrixXd bc(b.size(), V.cols());
	VectorXd Beq(3 * b.size());
	updateBalls(bc, Beq);
}

void ofApp::updateBalls(Eigen::MatrixXd& bc, Eigen::VectorXd& Beq) {
	using namespace Eigen;
	using namespace std;

	for (int i = 0; i < b.size(); i++)
	{
		bc.row(i) = V.row(b(i));
		switch (i % 4)
		{
		case 2:
			bc(i, 0) += 0.15*bbd*sin(0.5*anim_t);
			bc(i, 1) += 0.15*bbd*(1. - cos(0.5*anim_t));
			break;
		case 1:
			bc(i, 1) += 0.10*bbd*sin(1.*anim_t*(i + 1));
			bc(i, 2) += 0.10*bbd*(1. - cos(1.*anim_t*(i + 1)));
			break;
		case 0:
			bc(i, 0) += 0.20*bbd*sin(2.*anim_t*(i + 1));
			break;
		}
		Beq(3 * i + 0) = bc(i, 0);
		Beq(3 * i + 1) = bc(i, 1);
		Beq(3 * i + 2) = bc(i, 2);
	}

	BallPos = bc;
}

//--------------------------------------------------------------
void ofApp::update() {

}

//--------------------------------------------------------------
void ofApp::draw() {

	ofEnableDepthTest();
	ofEnableLighting();
	mLight.enable();
	mLight.setGlobalPosition(mCam.getGlobalPosition());

	mCam.begin();

	ofSetColor(255);
	mMesh.draw();

	// draw sphere
	ofSetColor(70, 252, 167);
	for (int i = 0; i < BallPos.rows(); i++) {
		glm::vec3 pos(BallPos(i, 0), BallPos(i, 1), BallPos(i, 2));
		ofDrawSphere(pos, 5.0);
	}
	mCam.end();

	mLight.disable();
	ofDisableDepthTest();

	ofSetColor(0);
	if (m_enable_cameraInput) {
		ofDrawBitmapString("SPACE: camera enable, drag disable", glm::vec2(20, 20));
	}
	else {
		ofDrawBitmapString("SPACE: camera disable, drag enable", glm::vec2(20, 20));
	}
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key) {
	switch (key)
	{
	case ' ':
		if (m_enable_cameraInput) {
			mCam.disableMouseInput();
			resolve = true;
			cout << "disableMouseInput" << endl;
		}
		else {
			mCam.enableMouseInput();
			resolve = false;
			cout << "enableMouseInput" << endl;
		}
		m_enable_cameraInput = !m_enable_cameraInput;
		break;
	case OF_KEY_RIGHT:
		mode = (ModeType)(((int)mode + 1) % ((int)NUM_MODE_TYPES + 1));
		cout << "mode: " << mode << endl;
		break;
	default:
		break;
	}
}

//--------------------------------------------------------------
void ofApp::keyReleased(int key) {

}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y) {

}

void ofApp::updateShape() {

	using namespace Eigen;
	using namespace std;
	if (resolve)
	{
		MatrixXd bc(b.size(), V.cols());
		VectorXd Beq(3 * b.size());

		bc = BallPos;

		for (int i = 0; i < b.size(); i++)
		{

			Beq(3 * i + 0) = bc(i, 0);
			Beq(3 * i + 1) = bc(i, 1);
			Beq(3 * i + 2) = bc(i, 2);
		}

		//updateBalls(bc, Beq);

		switch (mode)
		{
		default:
			assert("unknown mode");
		case MODE_TYPE_ARAP:
			igl::arap_solve(bc, arap_data, U);
			cout << "MODE_TYPE_ARAP" << endl;
			break;
		case MODE_TYPE_ARAP_GROUPED:
			igl::arap_solve(bc, arap_grouped_data, U);
			cout << "MODE_TYPE_ARAP_GROUPED" << endl;
			break;
		case MODE_TYPE_ARAP_DOF:
		{
			VectorXd L0 = L;
			arap_dof_update(arap_dof_data, Beq, L0, 30, 0, L);
			const auto & Ucol = M * L;
			U.col(0) = Ucol.block(0 * U.rows(), 0, U.rows(), 1);
			U.col(1) = Ucol.block(1 * U.rows(), 0, U.rows(), 1);
			U.col(2) = Ucol.block(2 * U.rows(), 0, U.rows(), 1);
			cout << "MODE_TYPE_ARAP_DOF" << endl;
			break;
		}
		}

		// update vertices
		for (int i = 0; i < U.rows(); i++) {
			mMesh.setVertex(i, glm::vec3(U(i, 0), U(i, 1), U(i, 2)));
		}

		anim_t += anim_t_dir;
	}
}
//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button) {

	if (resolve) {
		float dx = x - ofGetPreviousMouseX();
		float dy = y - ofGetPreviousMouseY();

		glm::vec3 ball(BallPos(m_closest_ball_index, 0), BallPos(m_closest_ball_index, 1), BallPos(m_closest_ball_index, 2));
		glm::vec3 screen_pos = mCam.worldToScreen(ball);
		glm::vec3 world_pos = mCam.screenToWorld(glm::vec3(screen_pos.x + dx, screen_pos.y + dy, screen_pos.z));

		BallPos(m_closest_ball_index, 0) = world_pos.x;
		BallPos(m_closest_ball_index, 1) = world_pos.y;
		BallPos(m_closest_ball_index, 2) = world_pos.z;
	}

	updateShape();
}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button) {

	if (resolve) {
		// compute closest ball
		int closest_ball_index = -1;
		float min_distance = std::numeric_limits<float>::max();

		for (int i = 0; i < BallPos.rows(); i++) {
			glm::vec3 pos(BallPos(i, 0), BallPos(i, 1), BallPos(i, 2));
			glm::vec3 screen_pos = mCam.worldToScreen(pos);

			float length = glm::distance(screen_pos, glm::vec3(x, y, 0));
			if (length < min_distance) {
				min_distance = length;
				closest_ball_index = i;
			}
		}

		m_closest_ball_index = closest_ball_index;
	}
}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button) {

}

//--------------------------------------------------------------
void ofApp::mouseEntered(int x, int y) {

}

//--------------------------------------------------------------
void ofApp::mouseExited(int x, int y) {

}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h) {

}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg) {

}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo) {

}
