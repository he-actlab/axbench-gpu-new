/*
 * Copyright 1993-2014 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */



/*
 * This sample demonstrates two adaptive image denoising technqiues:
 * KNN and NLM, based on computation of both geometric and color distance
 * between texels. While both techniques are already implemented in the
 * DirectX SDK using shaders, massively speeded up variation
 * of the latter techique, taking advantage of shared memory, is implemented
 * in addition to DirectX counterparts.
 * See supplied whitepaper for more explanations.
 */


// OpenGL Graphics includes
// #include <GL/glew.h>
// #if defined(__APPLE__) || defined(MACOSX)
// #include <GLUT/glut.h>
// #else
// #include <GL/freeglut.h>
// #endif

// CUDA utilities and system includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// Includes
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "imageBinarization.h"

// includes, project
#include <helper_functions.h> // includes for helper utility functions
//#include <helper_cuda.h>      // includes for cuda error checking and initialization

const char *sSDKsample = "CUDA ImageBinarization";

const char *filterMode[] =
{
    "Passthrough",
    "KNN method",
    "NLM method",
    "Quick NLM(NLM2) method",
    NULL
};

// Define the files that are to be save and the reference images for validation
const char *sOriginal[] =
{
    "image_passthru.ppm",
    "image_knn.ppm",
    "image_nlm.ppm",
    "image_nlm2.ppm",
    NULL
};

const char *sReference[] =
{
    "ref_passthru.ppm",
    "ref_knn.ppm",
    "ref_nlm.ppm",
    "ref_nlm2.ppm",
    NULL
};

////////////////////////////////////////////////////////////////////////////////
// Global data handlers and parameters
////////////////////////////////////////////////////////////////////////////////
//OpenGL PBO and texture "names"
GLuint gl_PBO, gl_Tex;
struct cudaGraphicsResource *cuda_pbo_resource; // handles OpenGL-CUDA exchange
//Source image on the host side
uchar4 *h_Src;
int imageW, imageH;
GLuint shader;

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int  g_Kernel = 0;
bool    g_FPS = false;
bool   g_Diag = false;
StopWatchInterface *timer = NULL;

//Algorithms global parameters
const float noiseStep = 0.025f;
const float  lerpStep = 0.025f;
static float knnNoise = 0.32f;
static float nlmNoise = 1.45f;
static float    lerpC = 0.2f;


const int frameN = 24;
int frameCounter = 0;


#define BUFFER_DATA(i) ((char *)0 + i)

// Auto-Verification Code
const int frameCheckNumber = 4;
int fpsCount = 0;        // FPS count for averaging
int fpsLimit = 1;        // FPS limit for sampling
unsigned int frameCount = 0;
unsigned int g_TotalErrors = 0;

int *pArgc   = NULL;
char **pArgv = NULL;

#define MAX_EPSILON_ERROR 5
#define REFRESH_DELAY     10 //ms



void runImageFilters(unsigned char *d_dst)
{  
    //printf("Hello!\n");

    cuda_imageBinarization(d_dst, imageW, imageH);

    
}



void shutDown(unsigned char k, int /*x*/, int /*y*/)
{
    switch (k)
    {
        case '\033':
        case 'q':
        case 'Q':
            printf("Shutting down...\n");

            sdkStopTimer(&timer);
            sdkDeleteTimer(&timer);

            CUDA_FreeArray();
            free(h_Src);

            exit(EXIT_SUCCESS);
            break;

        case '1':
            printf("Passthrough.\n");
            g_Kernel = 0;
            break;

        case '2':
            printf("KNN method \n");
            g_Kernel = 1;
            break;

        case '3':
            printf("NLM method\n");
            g_Kernel = 2;
            break;

        case '4':
            printf("Quick NLM(NLM2) method\n");
            g_Kernel = 3;
            break;

        case '*':
            printf(g_Diag ? "LERP highlighting mode.\n" : "Normal mode.\n");
            g_Diag = !g_Diag;
            break;

        case 'n':
            printf("Decrease noise level.\n");
            knnNoise -= noiseStep;
            nlmNoise -= noiseStep;
            break;

        case 'N':
            printf("Increase noise level.\n");
            knnNoise += noiseStep;
            nlmNoise += noiseStep;
            break;

        case 'l':
            printf("Decrease LERP quotent.\n");
            lerpC = MAX(lerpC - lerpStep, 0.0f);
            break;

        case 'L':
            printf("Increase LERP quotent.\n");
            lerpC = MIN(lerpC + lerpStep, 1.0f);
            break;

        case 'f' :
        case 'F':
            g_FPS = true;
            break;

        case '?':
            printf("lerpC = %5.5f\n", lerpC);
            printf("knnNoise = %5.5f\n", knnNoise);
            printf("nlmNoise = %5.5f\n", nlmNoise);
            break;
    }
}

// shader for displaying floating-point texture
static const char *shader_code =
    "!!ARBfp1.0\n"
    "TEX result.color, fragment.texcoord, texture[0], 2D; \n"
    "END";

//void runAutoTest(int argc, char **argv, const char *filename, int kernel_param)
void runAutoTest(int argc, char **argv)
{


    LoadBMPFile(&h_Src, &imageW, &imageH, argv[1]);
    //printf("Data init done.\n");

    CUDA_MallocArray(&h_Src, imageW, imageH);


    unsigned char *d_dst = NULL;
    unsigned char *h_dst = NULL;
    cudaMalloc((void **)&d_dst, imageW*imageH*sizeof(unsigned char));
    h_dst = (unsigned char *)malloc(imageH*imageW);

    {
        CUDA_Bind2TextureArray();
        runImageFilters(d_dst);
        CUDA_UnbindTexture();
        cudaDeviceSynchronize();

        cudaMemcpy(h_dst, d_dst, imageW*imageH*sizeof(unsigned char), cudaMemcpyDeviceToHost);
        //sdkSavePPM4ub(argv[2], h_dst, imageW, imageH);
        sdkSavePGM(argv[2], h_dst, imageW, imageH);
    }

    CUDA_FreeArray();
    free(h_Src);

    cudaFree(d_dst);
    free(h_dst);

    // flushed before the application exits
    cudaDeviceReset();
    exit(g_TotalErrors == 0 ? EXIT_SUCCESS : EXIT_FAILURE);
}


int main(int argc, char **argv)
{
    char *dump_file = NULL;


    pArgc = &argc;
    pArgv = argv;

    //printf("%s Starting...\n\n", sSDKsample);

    runAutoTest(argc, argv); // main function for performing image binarization

    cudaDeviceReset();
    exit(EXIT_SUCCESS);
}
