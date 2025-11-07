#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <inttypes.h>
#include "common.h"


void usage(int argc, char** argv);
void verify(int* sol, int* ans, int n);
void prefix_sum(int* src, int* prefix, int n);
void prefix_sum_p1(int* src, int* prefix, int n);
void prefix_sum_p2(int* src, int* prefix, int n);


int main(int argc, char** argv)
{
    // get inputs
    uint32_t n = 1048576;
    unsigned int seed = time(NULL);

    int m = 1;
    while (m < n) m <<= 1;

    if(argc > 2) {
        n = atoi(argv[1]); 
        seed = atoi(argv[2]);
    } else {
        usage(argc, argv);
        printf("using %"PRIu32" elements and time as seed\n", n);
    }

    // set up data 
    int* prefix_array = (int*) AlignedMalloc(sizeof(int) * n);  
    int* input_array = (int*) AlignedMalloc(sizeof(int) * n);
    srand(seed);
    for(int i = 0; i < n; i++) {
        input_array[i] = rand() % 100;
    }

    // set up timers
    uint64_t start_t;
    uint64_t end_t;
    InitTSC();

    // execute serial prefix sum and use it as ground truth
    start_t = ReadTSC();
    prefix_sum(input_array, prefix_array, n);
    end_t = ReadTSC();
    printf("Time to do O(N-1) prefix sum on a %"PRIu32" elements: %g (s)\n", 
           n, ElapsedTime(end_t - start_t));
    
    // printf("Elements of the array for prefix: ");
    // for (int i = 0; i < n; i++) {
    //     printf("%d ", prefix_array[i]); // Print each element followed by a space
    // }
    // printf("\n");


    // execute parallel prefix sum which uses a NlogN algorithm
    int* input_array1 = (int*) AlignedMalloc(sizeof(int) * m);  
    int* prefix_array1 = (int*) AlignedMalloc(sizeof(int) * m);  
    memcpy(input_array1, input_array, sizeof(int) * n);
    start_t = ReadTSC();
    prefix_sum_p1(input_array1, prefix_array1, n);
    end_t = ReadTSC();
    printf("Time to do O(NlogN) //prefix sum on a %"PRIu32" elements: %g (s)\n",
           n, ElapsedTime(end_t - start_t));
    verify(prefix_array, prefix_array1, n);

    
    // execute parallel prefix sum which uses a 2(N-1) algorithm
    memcpy(input_array1, input_array, sizeof(int));
    memset(prefix_array1, 0, sizeof(int));
    start_t = ReadTSC();
    prefix_sum_p2(input_array1, prefix_array1, n);
    end_t = ReadTSC();
    printf("Time to do 2(N-1) //prefix sum on a %"PRIu32" elements: %g (s)\n", 
           n, ElapsedTime(end_t - start_t));
    verify(prefix_array, prefix_array1, n);


    // free memory
    AlignedFree(prefix_array);
    AlignedFree(input_array);
    AlignedFree(input_array1);
    AlignedFree(prefix_array1);

    return 0;
}

void usage(int argc, char** argv)
{
    fprintf(stderr, "usage: %s <# elements> <rand seed>\n", argv[0]);
}


void verify(int* sol, int* ans, int n)
{
    int err = 0;
    for(int i = 0; i < n; i++) {
        if(sol[i] != ans[i]) {
            err++;
        }
    }
    if(err != 0) {
        fprintf(stderr, "There was an error: %d\n", err);
    } else {
        fprintf(stdout, "Pass\n");
    }
}

void prefix_sum(int* src, int* prefix, int n)
{
    prefix[0] = src[0];
    for(int i = 1; i < n; i++) {
        prefix[i] = src[i] + prefix[i - 1];
    }
}

void prefix_sum_p1(int* src, int* prefix, int n)
{
    int* temp = (int*)malloc(n * sizeof(int));
    int* current = prefix;
    int* next = temp;
    
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        current[i] = src[i];
    }

    int steps = (int)ceil(log2(n));

    for (int d = 0; d < steps; d++) {
        int offset = 1 << d;
        
        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            if (i >= offset)
                next[i] = current[i] + current[i - offset];
            else
                next[i] = current[i];
        }
        
        int* swap = current;
        current = next;
        next = swap;
    }
    
    if (current != prefix) {
        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            prefix[i] = current[i];
        }
    }
    
    free(temp);
}

void prefix_sum_p2(int* src, int* prefix, int n)
{
    if (n == 0) return;
    if (n == 1) { prefix[0] = 0; return; }

    // Find next power of 2
    int m = 1;
    while (m < n) m <<= 1;

    // Initialize prefix array
    for (int i = 0; i < n; i++)
        prefix[i] = src[i];
    for (int i = n; i < m; i++)
        prefix[i] = 0;  // Pad with zeros

    
    // ----- Up-sweep -----
    for (int d = 0; d < (int)log2(m); d++) {
        int stride = 1 << (d + 1);
        #pragma omp parallel for
        for (int j = 0; j < m; j += stride) {
            int a = j + (stride / 2) - 1;
            int b = j + stride - 1;
            prefix[b] += prefix[a];
        }
    }

    // Set root to zero
    prefix[m - 1] = 0;

    // ----- Down-sweep -----
    for (int d = (int)log2(m) - 1; d >= 0; d--) {
        int stride = 1 << (d + 1);
        #pragma omp parallel for
        for (int j = 0; j < m; j += stride) {
            int left  = j + (stride / 2) - 1;
            int right = j + stride - 1;
            if (right < m && left < m) {  // Add bounds check
                int t = prefix[left];
                prefix[left] = prefix[right];
                prefix[right] += t;
            }
        }
    }

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        prefix[i] += src[i];
    }

}



