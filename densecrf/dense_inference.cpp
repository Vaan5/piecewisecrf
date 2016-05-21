/*
    Copyright (c) 2011, Philipp Krähenbühl
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
        * Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
        * Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.
        * Neither the name of the Stanford University nor the
        names of its contributors may be used to endorse or promote products
        derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY Philipp Krähenbühl ''AS IS'' AND ANY
    EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL Philipp Krähenbühl BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#include "densecrf.h"
#include <cstdio>
#include <cmath>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <limits>
#include "util.h"
#include "probimage.h"

// Store the colors we read, so that we can write them again.
int nColors = 0;
int colors[255];
unsigned int getColor(const unsigned char * c){
	return c[0] + 256 * c[1] + 256 * 256 * c[2];
}

// used for experiments
unsigned int getLabel(const unsigned char* c) {
	unsigned char r = c[0], g = c[1], b = c[2];
	if(r == 128 && g == 0 && b == 0) {
		return 1;
	} 
	if(r == 64 && g == 64 && b == 0) {
		return 9;
	} 
	if(r == 128 && g == 128 && b == 0) {
		return 5;
	} 
	if(r == 64 && g == 0 && b == 128) {
		return 7;
	} 
	if(r == 64 && g == 64 && b == 128) {
		return 4;
	} 
	if(r == 128 && g == 64 && b == 128) {
		return 2;
	} 
	if(r == 128 && g == 128 && b == 128) {
		return 0;
	} 
	if(r == 192 && g == 192 && b == 128) {
		return 6;
	} 
	if(r == 0 && g == 0 && b == 192) {
		return 3;
	} 
	if(r == 192 && g == 128 && b == 128) {
		return 8;
	} 
	if(r == 0 && g == 128 && b == 192) {
		return 10;
	} 
	if (r == 0 && g == 0 && b == 0) {
		return 11;
	}
	return -1;
}

void putColor(unsigned char * c, unsigned int cc){
	c[0] = cc & 0xff; c[1] = (cc >> 8) & 0xff; c[2] = (cc >> 16) & 0xff;
}
// Produce a color image from a bunch of labels
unsigned char * colorize(short* map, int W, int H){
	unsigned char * res = new unsigned char[W*H * 3];
	for (int k = 0; k<W*H; k++){
		int r, g, b;
		get_color(map[k], r, g, b);
		res[3 * k] = (unsigned char)r;
		res[3 * k + 1] = (unsigned char)g;
		res[3 * k + 2] = (unsigned char)b;
		//int c = colors[ map[k] ];
		//putColor( r+3*k, c );
	}
	return res;
}

// Certainty that the groundtruth is correct
const float GT_PROB = 0.5;

// used for experiments
float * classifyCompressedNoLOG(const ProbImage& prob_im, int W, int H, int M, short* map)
{
	float * res = new float[W*H*M];
	float epsilon = 1e-10;
	for (int k = 0; k<W*H; k++)
	{
		float * r = res + k*M;
		float mx = prob_im(k%W, k / W, 0);
		int imx = 0;
		for (int j = 0; j<M; j++)
		{
			float prob = prob_im(k%W, k/W, j);
			r[j] = -log( prob + epsilon);
			if( mx < prob )
			{
			mx = prob;
			imx = j;
			}
			/*float boost_energy = prob_im(k%W, k / W, j);
			r[j] = -boost_energy;
			if (mx < boost_energy)
			{
				mx = boost_energy;
				imx = j;
			}*/
		}
		map[k] = (short)imx;
	}
	return res;
}

float * classify(const ProbImage& prob_im, int W, int H, int M, short* map)
{
	return classifyCompressedNoLOG(prob_im, W, H, M, map);
}

// Simple classifier that is 50% certain that the annotation is correct
// used for experiments
float * classify( const unsigned char * im, int W, int H, int M ){
	const float u_energy = -log( 1.0f / M );
	const float n_energy = -log( (1.0f - GT_PROB) / (M-1) );
	const float p_energy = -log( GT_PROB );
	float * res = new float[W*H*M];
	for( int k=0; k<W*H; k++ ){
		// Map the color to a label
		int c = getLabel( im + 3*k );
		int i;
		for( i=0;i<nColors && c!=colors[i]; i++ );
		if (c && i==nColors){
			if (i<M)
				colors[nColors++] = c;
			else
				c=0;
		}
		
		// Set the energy
		float * r = res + k*M;
		if (c != 11){
			for( int j=0; j<M; j++ )
				//r[j] = std::numeric_limits<float>::max();
				r[j] = n_energy;
			r[c] = p_energy;
		}
		else{
			for( int j=0; j<M; j++ )
				r[j] = u_energy;
		}
	}
	return res;
}

int main(int argc, char* argv[]) {
	if (argc < 4) {
		printf("Usage: %s dataset input_image compressed_unary output_img sigmaSmoothness weightSmoothness positionSigmaBi colorSigmaBi weightBi\n", argv[0]);
		return 1;
	}

	int M;
	if (argv[1] == "kitti") {
		M = 11;
	} else {
		M = 19;
	}

	int W, H, GW, GH;

	unsigned char *im = readPPM(argv[2], W, H);
	if (!im) {
		printf("Failed to load image!\n");
		return 1;
	}

	short *map = new short[W*H];
	float * features = new float[W * H * M];

	// read unary potentials (from binary file)
	FILE* file1 = fopen(argv[3], "rb");
	unsigned int ndim1, trash;
	fread(&ndim1, sizeof(unsigned int), 1, file1);
	for (unsigned int i = 0; i < ndim1; i++) {
		fread(&trash, sizeof(unsigned int), 1, file1);
	}
	fread(features, sizeof(float), W*H*M, file1);
	float *unary = features;

	// construct dense crf graph
	DenseCRF2D crf(W, H, M);
	crf.setUnaryEnergy(unary);
	crf.addPairwiseGaussian(atof(argv[5]), atof(argv[5]), atof(argv[6]));
	crf.addPairwiseBilateral(atof(argv[7]), atof(argv[7]), atof(argv[8]), atof(argv[8]), atof(argv[8]), im, atof(argv[9]));

	// inference
	crf.map(10, map);

	// save results
	FILE* fp = fopen(argv[4], "wb");
	if (!fp)
	{
		printf("Failed to open file '%s'!\n", argv[3]);
	}

	unsigned int ndim = 2;
	unsigned int uH = H;
	unsigned int uW = W;
	fwrite(&ndim, sizeof(unsigned int), 1, fp);
	fwrite(&uH, sizeof(unsigned int), 1, fp);
	fwrite(&uW, sizeof(unsigned int), 1, fp);
	fwrite(map, sizeof(short), W*H, fp);
	fclose(fp);

	delete[] map;
	delete[] im;
}
