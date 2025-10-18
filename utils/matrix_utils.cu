#include <iostream>
#include <cmath>
#include <random>
#include <fstream>
#include <stdio.h>
using namespace std;

typedef struct
{
    int height;
    int width;
    int stride;
    float *arr;
} Matrix;

__host__ __device__ float getElementMatrix(Matrix mat, int row, int col)
{
    int index = row * mat.stride + col;
    return mat.arr[index];
}

__host__ __device__ void setElementMatrix(Matrix mat, int row, int col, float value)
{
    int index = row * mat.stride + col;
    mat.arr[index] = value;
}

void printMatrix(Matrix A)
{
    for (int i = 0; i < A.height; i++)
    {
        for (int j = 0; j < A.width; j++)
        {
            int index = i * A.stride + j;
            cout << A.arr[index] << " ";
        }
        cout << endl;
    }
    cout << endl;
}

bool compare2Matrix(Matrix A, Matrix B)
{
    if ((A.height != B.height) || (A.width != B.width))
    {
        return false;
    }

    bool isSimilar = true;

    for (int i = 0; i < A.height; i++)
    {
        for (int j = 0; j < A.width; j++)
        {
            if (A.arr[i * A.stride + j] != B.arr[i * B.stride + j])
            {
                isSimilar = false;
                break;
            }
        }
    }
    return isSimilar;
}

void initMatrixHost(Matrix *mat, int height, int width, float defaultValue)
{
    mat->height = height;
    mat->width = width;
    mat->stride = width;
    mat->arr = (float *)malloc(width * height * sizeof(float));

    for (int i = 0; i < mat->height; i++)
    {
        for (int j = 0; j < mat->width; j++)
        {
            mat->arr[i * mat->stride + j] = defaultValue;
        }
    }
}

void matmulHost(Matrix A, Matrix B, Matrix C)
{
    for (int i = 0; i < C.height; i++)
    {
        for (int j = 0; j < C.width; j++)
        {
            float value = 0;
            for (int k = 0; k < A.width; k++)
            {
                value += getElementMatrix(A, i, k) * getElementMatrix(B, k, j);
            }

            setElementMatrix(C, i, j, value);
        }
    }
}