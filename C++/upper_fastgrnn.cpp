#ifdef MOTE_PROFILE
#define MOTE
#include <tinyhal.h>
#endif

#ifdef MOTE
#undef DBG
#endif

#include <iostream>
#include <string>
#include <sstream>
#include <algorithm>
#ifndef MOTE
#include <fstream>
#endif

#include "model_params.h"
#include "test_data.h"

using namespace std;

#define min(a,b) ((b)>(a))?a:b
#define max(a,b) ((a)>(b))?a:b

// initialize outfile if running on PC
#ifndef MOTE
	// Initialize output file
	ofstream outfile;
#endif

// Copy utils
inline void copyUIntVecToLL(uint* invec, ll* outvec, int vec_len)
{
	copy(invec,invec+vec_len, outvec);
}

inline void copyUShortVecToUInt(ushort* invec, uint* outvec, int vec_len)
{
	copy(invec,invec+vec_len, outvec);
}

//Vector-scalar multiplication
void mulVecScal(ll* vec, ll scal, int vec_len, ll* out){
	for(int i=0; i < vec_len; i++)
		out[i] = *(vec+i)*scal;
} 

//Vector-scalar division
void divVecScal(ll* vec, ll scal, int vec_len, ll* out){
	for(int i=0; i < vec_len; i++)
		out[i] = *(vec+i)/scal;
} 

//Matrix-vector multiplication
void mulMatVec(ll* mat, ll* vec, int mat_rows, int vec_len, ll* out){
	for(int i=0; i < mat_rows; i++){
		out[i] = 0;
		for(int j=0; j < vec_len; j++)
			out[i] += *((mat+i*vec_len)+j)*(*(vec+j));
	}
}

//Vector-vector addition
void addVecs(ll* vec1, ll* vec2, int vec_len, ll* out){
	for(int i=0; i < vec_len; i++)
		out[i] = *(vec1+i)+*(vec2+i);
}

//Vector-vector subtraction
void subVecs(ll* vec1, ll* vec2, int vec_len, ll* out){
	for(int i=0; i < vec_len; i++)
		out[i] = *(vec1+i)-*(vec2+i);
}

//Vector-vector multiplication (Hadamard)
void mulVecs(ll* vec1, ll* vec2, int vec_len, ll* out){
	for(int i=0; i < vec_len; i++)
		out[i] = *(vec1+i)*(*(vec2+i));
}

// Standardization with scaling
void stdScaleInput(ll* in_vec, int vec_len, ll* out_vec){
	for(int i=0; i<vec_len; i++)
		*(out_vec+i) = I_u*(*(in_vec+i)-mean_u[i])/stdev_u[i];
}

// quantTanh
inline void quantTanh(ll* vec, int vec_len, ll scale, ll* out_vec){
	for(int i=0; i<vec_len; i++)
		*(out_vec+i) = max(-scale, min(scale, *(vec+i)));
}

// quantSigm
inline void quantSigm(ll* vec, int vec_len, ll scale, ll* out_vec){
	for(int i=0; i<vec_len; i++)
		*(out_vec+i) = max(min((*(vec+i)+scale)/2, scale),0);
}

// Print utils
#ifdef DBG
void util_printVec(ll* vec, int vec_len){
	for(int i=0; i < vec_len; i++)
		cout << vec[i] << '\t';
	cout << "\n\n";
}

void util_printMatrix(uint* mat, int row_len, int col_len){
	for(int i=0; i < row_len; i++){
		for(int j=0; j < col_len; j++)
			cout << *(mat+i*col_len+j) << '\t';
		cout << endl;
	}
	cout << "\n\n";
}
#endif

// Extract instance
void extract_instance(uint* src, uint*dst, int row_start, int row_end, int vec_len){
	for(int t=0, k=row_start * vec_len; k < row_end * vec_len; t++, k++)
		*(dst+t) = *(src + k);
}

// Matrix slice utils
void util_slice2D(uint* src, uint*dst, int row_index, int vec_len){
	for(int j=0; j < vec_len; j++)
		*(dst+j) = *((src+row_index*vec_len)+j);
}

void util_slice3D(uint* src, uint*dst, int row_index, int col_len, int vec_len){
	uint *slice_beg = (src + row_index * col_len * vec_len);
	for(int k=0; k < col_len * vec_len; k++)
			*(dst+k) = *(slice_beg + k);
}
#ifndef MOTE
// String builder util
string strBuild(ll i, char delim)
{
    string s;
    stringstream out;
    out << i;
    out << delim;
    s = out.str();
    return s;
}
#endif

inline int emi_rnn(uint* test_input){	
	ll h[hiddenDims_u] = {0};
	ll out_wRank_u[wRank_u] = {0};
	ll out_uRank_u[uRank_u] = {0};
	ll out_hiddenDims_u[hiddenDims_u] = {0};
	ll out_numClasses_u[numClasses_u] = {0};
	for(int t=0; t<timeSteps_u; t++){
#ifdef MOTE_PROFILE
		// Profile latency per timestep			
		//hal_printf("b");
		CPU_GPIO_SetPinState(0, true);
#endif
		uint x_int[inputDims_u] = {0};
		util_slice2D(test_input, x_int, t, inputDims_u);

		ll x[inputDims_u] = {};

		copyUIntVecToLL(x_int, x, inputDims_u);
#ifdef DBG
		cout << "Current input array in ll" << endl;
		util_printVec(x, inputDims_u);
#endif	
		stdScaleInput(x, inputDims_u, x);
#ifdef DBG	
		cout << "Post-standardization input" << endl;
		util_printVec(x, inputDims_u);
#endif	
		// Precompute
		ll pre[hiddenDims_u] = {0};
		mulMatVec((ll*)qW1_transp_u, x, wRank_u, inputDims_u, out_wRank_u);
		mulMatVec((ll*)qW2_transp_u, out_wRank_u, hiddenDims_u, wRank_u, pre);

		mulMatVec((ll*)qU1_transp_u, h, uRank_u, hiddenDims_u, out_uRank_u);
		mulMatVec((ll*)qU2_transp_u, out_uRank_u, hiddenDims_u, uRank_u, out_hiddenDims_u);

		addVecs(pre, out_hiddenDims_u, hiddenDims_u, pre);

		divVecScal(pre, q_u, hiddenDims_u, pre);

#ifdef DBG
		cout << "Pre at t=" << t << endl;
		util_printVec(pre, hiddenDims_u);
#endif

		// Create h_, z
		ll h_[hiddenDims_u] = {0};
		ll z[hiddenDims_u] = {0};

		addVecs(pre, (ll*)qB_h_u, hiddenDims_u, h_);
		addVecs(pre, (ll*)qB_g_u, hiddenDims_u, z);

		UPDATE_NL(h_, hiddenDims_u, q_times_I_u, h_);
		divVecScal(h_, q_u, hiddenDims_u, h_);

		GATE_NL(z, hiddenDims_u, q_times_I_u, z);
		divVecScal(z, q_u, hiddenDims_u, z);
#ifdef DBG
		cout << "h_ at t=" << t << endl;
		util_printVec(h_, hiddenDims_u);

		cout << "z at t=" << t << endl;
		util_printVec(z, hiddenDims_u);
#endif
		// Create new h
		mulVecs(z, h, hiddenDims_u, h);

		subVecs((ll*)I_u_vec, z, hiddenDims_u, out_hiddenDims_u);
		mulVecScal((ll*)out_hiddenDims_u, I_times_zeta_u, hiddenDims_u, out_hiddenDims_u);
		addVecs(out_hiddenDims_u, (ll*)I_squared_times_nu_u_vec, hiddenDims_u, out_hiddenDims_u);
		mulVecs(out_hiddenDims_u, h_, hiddenDims_u, out_hiddenDims_u);
		divVecScal(out_hiddenDims_u, I_u, hiddenDims_u, out_hiddenDims_u);
		addVecs(h, out_hiddenDims_u, hiddenDims_u, h);
		divVecScal(h, I_u, hiddenDims_u, h);
#ifdef MOTE_PROFILE
		//hal_printf("e");
		CPU_GPIO_SetPinState(0, false);
#endif
#ifdef DBG
		cout << "h at t=" << t << endl;
		util_printVec(h, hiddenDims_u);
#endif
	}

	// Classify
	mulMatVec((ll*)qFC_Weight_u, h, numClasses_u, hiddenDims_u, out_numClasses_u);
	addVecs(out_numClasses_u, (ll*)qFC_Bias_u, numClasses_u, out_numClasses_u);
#ifdef DBG
	cout << "Classification output:" << endl;
	util_printVec(out_numClasses_u, numClasses_u);
#endif
#ifdef MOTE
	if(out_numClasses_u[0]>out_numClasses_u[1])
		return 0;
	else
		return 1;
#endif
#ifndef MOTE
	//Print decision to csv file
	string outstr;
	for(int c = 0; c < numClasses_u -1 ; c++)
		outstr += strBuild(out_numClasses_u[c], ',');
	outstr += strBuild(out_numClasses_u[numClasses_u -1], '\n');
	outfile << outstr;
#endif
}

/*bool emi_driver(uint* data){
	// Reshape data
	//int (&data2D)[orig_num_steps][inputDims_u] = *reinterpret_cast<int (*)[orig_num_steps][inputDims_u]>(&data);
	int maxconsectargets = 0;
	bool detect_in_bag = false;
	// Create instances and run EMI
	for(int start = 0, i=0; i<numInstances; start += instStride, i++){
		int end;
		if(i==numInstances-1)
			// Correction for last iteration
			end = orig_num_steps;
		else
			end = start + timeSteps;

		uint next_inst[timeSteps][inputDims_u] = {0};
		extract_instance(data, (uint*)next_inst, start, end, inputDims_u);
		
		// Call emi_rnn
		int inst_dec = emi_rnn((uint*)next_inst);
		//hal_printf("%d", inst_dec);
		// Bag-level detection logic
		if(inst_dec==0)
			maxconsectargets = 0;
		else
			maxconsectargets++;
		if(maxconsectargets>=k)
			return true;
	}
	return false;
}*/

void run_test(){
	int size = sizeof(qW1_transp_u) + sizeof(qFC_Bias_u) + sizeof(qW2_transp_u) + sizeof(qU2_transp_u) + sizeof(qFC_Weight_u) + sizeof(qU1_transp_u) + sizeof(qB_g_u) + sizeof(qB_h_u) + sizeof(q_u) + sizeof(I_u) + sizeof(mean_u) + sizeof(stdev_u) + sizeof(I_u_vec) + sizeof(q_times_I_u) + sizeof(I_squared_times_nu_u_vec) + sizeof(I_times_zeta_u);
	
#ifndef MOTE
	cout << "Model size: " << size/1000 << " KB" << endl << endl;
#endif
#ifdef MOTE_PROFILE
	CPU_GPIO_EnableOutputPin(0, false); //J11-3
	CPU_GPIO_EnableOutputPin(1, false); //J11-4
#endif
#ifndef MOTE
	// Initialize output file
	outfile.open("out_c++.csv");
#endif
	for(int d=0; d < numData; d ++)
	{
		uint test_input[timeSteps_u][inputDims_u] = {0};
		util_slice3D((uint*) test_inputs, (uint*) test_input, d, timeSteps_u, inputDims_u);

#ifdef MOTE_PROFILE
		// Profile latency per bag (second in V1)
		if(d%numInstances==0)
			CPU_GPIO_SetPinState(1, true);
#endif

#ifdef DBG
		util_printMatrix((uint*) test_input, timeSteps, inputDims_u);
#endif
		// Call emi_rnn
		emi_rnn((uint*)test_input);

#ifdef MOTE_PROFILE
		if(d%numInstances==numInstances-1)
			CPU_GPIO_SetPinState(1, false);
#endif
	}
#ifndef MOTE
	outfile.close();
#else
	hal_printf("Test complete.");
#endif
}
#ifndef MOTE
int main(){
	run_test();
}
#endif